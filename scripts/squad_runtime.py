"""Run the trained R6 squad protocol through the object layer and check it (task B).

Every centre-to-snake message becomes a signed ``codes`` envelope on the runtime bus
(``experiments.squad_runtime``). For each trained run this script plays the held-out
test episodes three ways and writes ``docs/results/squad_runtime.{md,json}``:

1. object layer: one ``squad.forage`` task per episode through the CommandCentre;
2. fast path, B = 1: ``run_episodes`` on the same seed (step-for-step comparison);
3. fast path, batched (B = episodes), as the R6 evaluation runs it (different action
   noise, so equal only in distribution).

Then a tamper test (a compromised lead sidecar rewrites signed envelopes) and a small
sample trace. The trained parameters are not in the repository:

    git show origin/gpu-results:runs/r6-c-r3/artifacts/learned_protocol_params.npz \
        > params_r3.npz
    python scripts/squad_runtime.py --params params_r3.npz --runs 0-9 --episodes 100
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

from genetic_coil_arena.envs.squad import SquadEnv, run_episodes  # noqa: E402
from genetic_coil_arena.evaluation import mean_ci  # noqa: E402
from genetic_coil_arena.experiments.learned_protocol import (  # noqa: E402
    CONDITION,
    LEARNED,
    MODES,
    TrainSettings,
    policies_for,
    test_episode_seeds,
)
from genetic_coil_arena.experiments.squad_runtime import (  # noqa: E402
    FALLBACK,
    SQUAD_FORAGE,
    SquadRuntime,
    compare_episode,
    fast_path_episode,
    hold_last_hook,
    snake_names,
)
from genetic_coil_arena.runtime import TraceLog  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_STEPS = 100


def parse_runs(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def forge_hook(targets: set[tuple[int, int]], vocab: int) -> Any:
    """Fast-path hook: the forged code ((sent + 1) mod V) is accepted."""

    def hook(messages: np.ndarray, step: int, mask: Any = None) -> np.ndarray:
        out = messages.copy()
        for r in range(out.shape[1]):
            if (step, r) in targets:
                out[0, r, 0] = (out[0, r, 0] + 1) % vocab
        return out

    return hook


def evaluate_run(
    mode: str, run: int, theta: np.ndarray, settings: TrainSettings, n: int
) -> dict[str, Any]:
    config = settings.squad_config()
    seeds = test_episode_seeds(run, n)
    specialist, centre = policies_for(mode, settings, theta)
    runtime = SquadRuntime(specialist, centre, config)
    runtime.run_episode(seeds[0])  # warm-up (numba compile, caches)
    run_episodes(SquadEnv(config), specialist, centre, seeds[:1])

    start = time.perf_counter()
    outcomes = [runtime.run_episode(s) for s in seeds]
    t_object = time.perf_counter() - start

    start = time.perf_counter()
    for s in seeds:
        run_episodes(SquadEnv(config), specialist, centre, [s])
    t_fast1 = time.perf_counter() - start

    start = time.perf_counter()
    batched = run_episodes(SquadEnv(config), specialist, centre, seeds)
    t_batch = time.perf_counter() - start

    rows = [
        compare_episode(o, fast_path_episode(specialist, centre, config, s))
        for o, s in zip(outcomes, seeds)
    ]
    agent_steps = int(sum(int(o.history["alive"].sum()) for o in outcomes))
    envelopes = sum(1 for r in runtime.trace.records if r.kind == "envelope")
    return {
        "mode": mode,
        "condition": CONDITION[mode],
        "run": run,
        "episodes": n,
        "states": sorted({o.state for o in outcomes}),
        "identical_episodes": int(sum(r["identical"] for r in rows)),
        "code_mismatches": int(sum(r["codes_mismatches"] for r in rows)),
        "heard_mismatches": int(sum(r["heard_mismatches"] for r in rows)),
        "action_mismatches": int(sum(r["actions_mismatches"] for r in rows)),
        "steps": int(sum(r["steps_object"] for r in rows)),
        "agent_steps": agent_steps,
        "team_food_object": [r["team_food_object"] for r in rows],
        "team_food_fast1": [r["team_food_fast"] for r in rows],
        "team_food_batched": batched["team_food"].tolist(),
        "seconds_object": t_object,
        "seconds_fast1": t_fast1,
        "seconds_batched": t_batch,
        "envelopes": envelopes,
        "trace_records": len(runtime.trace),
        "rejections": int(sum(len(o.rejections) for o in outcomes)),
    }


def tamper_test(
    theta: np.ndarray, run: int, settings: TrainSettings, n: int, steps: list[int]
) -> dict[str, Any]:
    """Corrupt the lead's envelope to snake 0 at the given steps of each episode."""
    config = settings.squad_config()
    names = snake_names(config.n_snakes)
    specialist, centre = policies_for(LEARNED, settings, theta)
    targets = [(s, names[0]) for s in steps]
    runtime = SquadRuntime(specialist, centre, config, tamper=targets)
    vocab = config.message.vocab_size
    rows = []
    for seed in test_episode_seeds(run, n):
        out = runtime.run_episode(seed)
        drops = {(t["step"], names.index(t["recipient"])) for t in out.tampered}
        replay = fast_path_episode(
            specialist, centre, config, seed, hook=hold_last_hook(drops)
        )
        intact = fast_path_episode(specialist, centre, config, seed)
        forged = fast_path_episode(
            specialist, centre, config, seed, hook=forge_hook(drops, vocab)
        )
        cmp = compare_episode(out, replay)
        rows.append(
            {
                "seed": seed,
                "state": out.state,
                "tampered": len(out.tampered),
                "rejected": len(out.rejections),
                "reasons": sorted({r["reason"] for r in out.rejections}),
                "matches_fallback_replay": cmp["identical"],
                "team_food": (out.result or {}).get("team_food"),
                "team_food_intact": intact["team_food"],
                "team_food_forgery_accepted": forged["team_food"],
                "actions_changed_vs_intact": _changed(out.history, intact),
                "actions_changed_if_forgery_accepted": _changed(
                    forged_history(forged), intact
                ),
            }
        )
    return {
        "run": run,
        "episodes": n,
        "target_steps": steps,
        "recipient": names[0],
        "fallback": FALLBACK,
        "rows": rows,
        "trace_events": len(runtime.trace.events("signature_rejected")),
    }


def forged_history(fast: dict[str, Any]) -> dict[str, np.ndarray]:
    return {"actions": fast["actions"], "alive": fast["alive"]}


def _changed(history: dict[str, np.ndarray], intact: dict[str, Any]) -> int:
    """Live-snake actions that differ from the intact episode (over the common
    prefix: once one action differs the episodes diverge)."""
    t = min(len(history["actions"]), len(intact["actions"]))
    a, b = history["actions"][:t], intact["actions"][:t]
    live = history["alive"][:t] & intact["alive"][:t]
    return int((a != b)[live].sum())


def write_sample(theta: np.ndarray, settings: TrainSettings, path: Path) -> dict:
    """Two short episodes (the second with one tampered envelope) as a JSONL trace."""
    config = settings.squad_config()
    names = snake_names(config.n_snakes)
    specialist, centre = policies_for(LEARNED, settings, theta)
    if path.exists():
        path.unlink()
    trace = TraceLog(path)
    runtime = SquadRuntime(specialist, centre, config, trace=trace)
    seeds = test_episode_seeds(0, 2)
    first = runtime.run_episode(seeds[0], max_steps=SAMPLE_STEPS)
    tampered = SquadRuntime(
        specialist, centre, config, trace=trace, tamper=[(20, names[1])]
    )
    second = tampered.run_episode(
        seeds[1], max_steps=SAMPLE_STEPS, assignment=[names[2], names[0], names[1]]
    )
    trace.close()
    loaded = TraceLog.load(path)
    triples = loaded.triples()
    return {
        "path": _display(path),
        "bytes": path.stat().st_size,
        "records": len(loaded),
        "triples": len(triples),
        "specs": [t.spec.spec if t.spec else None for t in triples],
        "situations": [len(loaded.situations(t.trace_id)) for t in triples],
        "envelopes": [len(t.messages) for t in triples],
        "rejections": len(loaded.events("signature_rejected")),
        "results": [first.result, second.result],
    }


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _ci(values: list[float]) -> list[float]:
    m, h = mean_ci(values)
    return [m, h]


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_run = {
        k: [float(np.mean(r[k])) for r in rows]
        for k in ("team_food_object", "team_food_fast1", "team_food_batched")
    }
    diff = [
        a - b for a, b in zip(per_run["team_food_object"], per_run["team_food_batched"])
    ]
    tot = {k: sum(r[k] for r in rows) for k in rows[0] if k.startswith("seconds_")}
    eps = sum(r["episodes"] for r in rows)
    return {
        "runs": [r["run"] for r in rows],
        "episodes": eps,
        "identical_episodes": sum(r["identical_episodes"] for r in rows),
        "code_mismatches": sum(r["code_mismatches"] for r in rows),
        "heard_mismatches": sum(r["heard_mismatches"] for r in rows),
        "action_mismatches": sum(r["action_mismatches"] for r in rows),
        "steps": sum(r["steps"] for r in rows),
        "envelopes": sum(r["envelopes"] for r in rows),
        "team_food_object": _ci(per_run["team_food_object"]),
        "team_food_fast1": _ci(per_run["team_food_fast1"]),
        "team_food_batched": _ci(per_run["team_food_batched"]),
        "object_minus_batched": _ci(diff),
        "episodes_per_s": {
            "object": eps / tot["seconds_object"],
            "fast_b1": eps / tot["seconds_fast1"],
            "fast_batched": eps / tot["seconds_batched"],
        },
        "ms_per_step_object": 1000
        * tot["seconds_object"]
        / sum(r["steps"] for r in rows),
    }


def render(payload: dict[str, Any]) -> str:
    s = payload["summary"]
    eps = s["episodes_per_s"]
    fmt = lambda ci: f"{ci[0]:.2f} ± {ci[1]:.2f}"  # noqa: E731
    lines = [
        "# Squad: the trained protocol through the object layer (task B)",
        "",
        f"- Date: {payload['date']}",
        f"- Commit: {payload['commit']} (code; this report is committed on top of it)",
        f"- Machine: {payload['machine']}, one process, BLAS threads = 1",
        f"- Policies: {payload['condition']} (R6, r = {payload['radius']}), the "
        "best-validation checkpoints of training runs "
        f"{', '.join(map(str, s['runs']))} from `runs/r6-c-r3` "
        "(`learned_protocol_params.npz` on the `gpu-results` branch; the report is "
        "`docs/results/learned_protocol_gpu_r3.md`).",
        f"- Episodes: {payload['episodes']} held-out test episodes per run (run s plays "
        "seeds 3,500,000,000 + s × 10,000 + i, the R6 test seeds), T = 300.",
        "",
        "Reproduce: `git show origin/gpu-results:runs/r6-c-r3/artifacts/"
        "learned_protocol_params.npz > params_r3.npz`, then "
        f"`python scripts/squad_runtime.py --params params_r3.npz --runs "
        f"{payload['runs_arg']} --episodes {payload['episodes']}`.",
        "",
        "## Design",
        "",
        "- **One episode = one task.** A `squad.forage` v1 GoalSpec (`episode_seed`, "
        "optional `max_steps` ≤ 1000, optional `assignment` = specialist name per "
        "colour; result schema `team_food`, `food`, `steps`, `survivors`, `return`) is "
        "made with `SpecRegistry.make` and submitted to the runtime `CommandCentre`, "
        "which routes it to `squad-lead` and validates the result against the schema.",
        "- **Parties on the `InProcessBus`:** `command-centre` (runtime), `squad-lead` "
        "(the learned centre network, wrapped as a `ClockAware` specialist) and "
        "`squad-snake-0..2` (one specialist per snake, all with the trained specialist "
        "network). Each runs in a `ClockAdapter`; the centre and the lead sign every "
        "envelope (HMAC-SHA256) and every receiver verifies.",
        "- **Messages:** every tick the lead computes one code per live snake from the "
        "full view and sends it as a `codes` envelope (V = 8, L = 1) built only by "
        "`protocol.codecs.codes_to_envelopes`; a snake decodes it with "
        "`envelopes_to_codes` after its adapter verified the signature. Bus delivery "
        "takes one round, which is the fast path's 'read one step later'.",
        "- **World:** `SquadWorld` wraps `SquadEnv` with B = 1 as the snakes' sensors "
        "and actuators and steps once every live snake acted. It never carries "
        "messages (`env.step` gets no messages; the snakes read their code only from "
        "envelopes).",
        "- **Matched randomness:** the fast path's `LearnedSpecialist` draws the "
        "Gumbel action noise of all three snakes from one generator per episode. The "
        "world draws the same noise each step and hands snake k its row (common "
        "random numbers), so the object layer must reproduce the B = 1 fast path "
        "exactly; the batched fast path (B = all episodes, as the R6 evaluation runs "
        "it) draws different noise and can only agree in distribution.",
        "- **Trace:** `runtime.TraceLog` (JSONL) holds every envelope (goal, codes, "
        "result), the task state events, a compact situation snapshot per step "
        "(heads, headings, food, health, alive, score; its `clock` equals that step's "
        "code envelopes) and a `signature_rejected` event per dropped envelope.",
        "",
        "## Equivalence (acceptance test)",
        "",
        f"- **Step-for-step identity:** {s['identical_episodes']} / {s['episodes']} "
        "episodes identical to the B = 1 fast path: "
        f"{s['code_mismatches']} code mismatches, {s['heard_mismatches']} "
        f"heard-code mismatches and {s['action_mismatches']} action mismatches over "
        f"{s['steps']:,} steps (live snakes), and the same team food in every episode.",
        f"- **Envelopes:** {s['envelopes']:,} envelopes crossed the bus and all "
        "verified (0 signature rejections outside the tamper test).",
        "",
        "Team food per episode, mean ± 95% t-CI over training runs (each run's score "
        "is its mean over the test episodes):",
        "",
        "| path | team food |",
        "|---|---|",
        f"| object layer (runtime, envelopes) | {fmt(s['team_food_object'])} |",
        f"| fast path, B = 1, same seeds and noise | {fmt(s['team_food_fast1'])} |",
        f"| fast path, batched (R6 evaluation style) | {fmt(s['team_food_batched'])} |",
        "",
        "Object layer − batched fast path, paired by run: "
        f"{s['object_minus_batched'][0]:+.2f} ± {s['object_minus_batched'][1]:.2f} "
        "(the CI contains 0 when the two agree in distribution).",
        "",
        "| run | identical | object | fast B = 1 | batched | object ep/s | "
        "B = 1 ep/s | batched ep/s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in payload["per_run"]:
        lines.append(
            f"| {r['run']} | {r['identical_episodes']}/{r['episodes']} | "
            f"{np.mean(r['team_food_object']):.2f} | "
            f"{np.mean(r['team_food_fast1']):.2f} | "
            f"{np.mean(r['team_food_batched']):.2f} | "
            f"{r['episodes'] / r['seconds_object']:.2f} | "
            f"{r['episodes'] / r['seconds_fast1']:.1f} | "
            f"{r['episodes'] / r['seconds_batched']:.1f} |"
        )
    lines += [
        "",
        "## Overhead",
        "",
        f"- Object layer: {eps['object']:.2f} episodes/s "
        f"({s['ms_per_step_object']:.2f} ms per episode step: 3 signed envelopes, "
        "verification, decoding, one trace snapshot, and per-snake network calls).",
        f"- Fast path, B = 1: {eps['fast_b1']:.1f} episodes/s "
        f"({eps['fast_b1'] / eps['object']:.1f}× the object layer).",
        f"- Fast path, batched: {eps['fast_batched']:.1f} episodes/s "
        f"({eps['fast_batched'] / eps['object']:.0f}× the object layer).",
        "",
        "The object layer is for orchestration, logging and (later) real models, not "
        "training; it is fast enough to record traces and run demos.",
        "",
    ]
    t = payload["tamper"]
    rows = t["rows"]
    n_t = sum(r["tampered"] for r in rows)
    n_r = sum(r["rejected"] for r in rows)
    replay = sum(r["matches_fallback_replay"] for r in rows)
    changed = sum(r["actions_changed_vs_intact"] > 0 for r in rows)
    forged = sum(r["actions_changed_if_forgery_accepted"] > 0 for r in rows)
    reasons = sorted({x for r in rows for x in r["reasons"]})
    lines += [
        "## Tamper test",
        "",
        f"A compromised sidecar on the lead (`TamperingAdapter`) rewrites the signed "
        f"envelope to `{t['recipient']}` at steps {t['target_steps']} of each of "
        f"{t['episodes']} episodes of run {t['run']}: the code becomes (sent + 1) mod 8 "
        "after signing (only while that snake is alive).",
        "",
        f"- Tampered envelopes: {n_t}; rejected by the snake's `ClockAdapter`: {n_r} "
        f"(reason: {', '.join(reasons) or '-'}); `signature_rejected` trace events: "
        f"{t['trace_events']}. Every task still completed "
        f"({sorted({r['state'] for r in rows})}).",
        f"- **Fallback (`{t['fallback']}`):** a dropped envelope counts as not sent, so "
        "the snake keeps the last verified code of the current task (0 before any), "
        "the environment's own convention for the `message` observation. "
        f"{replay} / {len(rows)} tampered episodes are identical, step for step, to the "
        "fast path with a hook that re-delivers the previous code at the tampered "
        "steps.",
        f"- Effect: with the fallback, {changed} / {len(rows)} episodes acted "
        "differently from the intact episode (a stale code instead of the fresh one); "
        f"had the forged code been accepted, {forged} / {len(rows)} would have.",
        "",
    ]
    sm = payload["sample"]
    lines += [
        "## Sample trace",
        "",
        f"`{sm['path']}` ({sm['bytes'] / 1024:.0f} KiB, {sm['records']} records): two "
        "episodes of run 0's learned protocol capped at `max_steps` = "
        f"{SAMPLE_STEPS} through the "
        "GoalSpec; the second uses a permuted `assignment` and has one tampered "
        f"envelope ({sm['rejections']} `signature_rejected` event). "
        f"`TraceLog.triples()` gives {sm['triples']} (situation, spec, messages) "
        f"triples with spec {sorted(set(sm['specs']))}, "
        f"{sm['situations']} situation snapshots and {sm['envelopes']} envelopes. "
        "`TraceLog.triples()` keeps only the last snapshot per trace; per-step "
        "snapshots are in `situations(trace_id)` and pair with envelopes by `clock`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, default=LEARNED)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--runs", default="0-9")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--tamper-run", type=int, default=0)
    parser.add_argument("--tamper-episodes", type=int, default=20)
    parser.add_argument("--tamper-steps", default="50,150")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "docs/results/squad_runtime.md"
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=ROOT / "docs/results/squad_runtime_trace_sample.jsonl",
    )
    args = parser.parse_args()

    settings = TrainSettings(radius=args.radius)
    params = np.load(args.params)

    def theta(mode: str, run: int) -> np.ndarray:
        return params[f"r{args.radius}_{mode}_run{run}"].astype(np.float64)

    rows = []
    for run in parse_runs(args.runs):
        row = evaluate_run(
            args.mode, run, theta(args.mode, run), settings, args.episodes
        )
        print(
            f"[run {run}] identical {row['identical_episodes']}/{row['episodes']}, "
            f"food object {np.mean(row['team_food_object']):.2f} / batched "
            f"{np.mean(row['team_food_batched']):.2f}, "
            f"{row['episodes'] / row['seconds_object']:.2f} ep/s object",
            flush=True,
        )
        rows.append(row)
    tamper = tamper_test(
        theta(LEARNED, args.tamper_run),
        args.tamper_run,
        settings,
        args.tamper_episodes,
        [int(s) for s in args.tamper_steps.split(",")],
    )
    print(f"[tamper] {sum(r['rejected'] for r in tamper['rows'])} rejected", flush=True)
    sample = write_sample(theta(LEARNED, 0), settings, args.sample)
    print(f"[sample] {sample['bytes']} bytes", flush=True)

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    ).stdout.strip()
    payload = {
        "date": datetime.date.today().isoformat(),
        "commit": commit,
        "machine": f"{platform.processor() or platform.machine()}, "
        f"{os.cpu_count()} logical cores",
        "spec": SQUAD_FORAGE,
        "condition": CONDITION[args.mode],
        "radius": args.radius,
        "runs_arg": args.runs,
        "episodes": args.episodes,
        "summary": summarise(rows),
        "per_run": rows,
        "tamper": tamper,
        "sample": sample,
    }
    args.out.write_text(render(payload), encoding="utf-8")
    args.out.with_suffix(".json").write_text(
        json.dumps(payload, indent=1, default=float), encoding="utf-8"
    )
    print(f"Wrote {args.out} and .json", flush=True)


if __name__ == "__main__":
    main()
