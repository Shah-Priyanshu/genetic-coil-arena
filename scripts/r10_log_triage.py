"""R10 step 1: log triage with forced specialisation (goal G4).

    # one slice of the evaluation (what the runner jobs do)
    python scripts/r10_log_triage.py run --model hf:Qwen/Qwen3-1.7B \\
        --conditions single_head,single_tail,single_selected,oracle_selected --systems SS,OB

    # tiny local smoke run
    python scripts/r10_log_triage.py run --limit 10 --split dev --out outputs/r10_smoke

    # merge the slices into docs/results/r10_log_triage.{md,json}
    python scripts/r10_log_triage.py report --from "runs:r10-*" --out docs/results/r10_log_triage.md

Needs the RCAEval cache (``python scripts/fetch_rcaeval.py``). Records go to
``<out>/results.jsonl`` (``$GCA_RUN_DIR/artifacts`` under the runner) and a killed run
resumes from them. ``hf:`` models need torch, transformers and peft; ``ollama:`` models
need a running Ollama server.
"""

import argparse
import datetime
import glob
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parent.parent


def log(message: str) -> None:
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def out_dir(arg: Optional[str]) -> Path:
    if arg:
        return Path(arg)
    run_dir = os.environ.get("GCA_RUN_DIR")
    return (
        (Path(run_dir) / "artifacts")
        if run_dir
        else REPO / "outputs" / "r10_log_triage"
    )


def pick(files: list[Path], limit: Optional[int]) -> list[Path]:
    """``limit`` files spread evenly over the sorted list (so a smoke run mixes faults)."""
    if not limit or limit >= len(files):
        return files
    step = len(files) / limit
    return [files[int(i * step)] for i in range(limit)]


def start_ollama(backend: Any) -> "subprocess.Popen[bytes]":
    """Start ``ollama serve`` (the caller stops it) and wait until it answers."""
    local = os.environ.get("LOCALAPPDATA", "")
    exe = Path(local) / "Programs" / "Ollama" / "ollama.exe"
    if not exe.exists():
        raise SystemExit("Ollama is unreachable and no ollama.exe was found")
    log(f"starting `{exe} serve`")
    proc = subprocess.Popen(
        [str(exe), "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    for _ in range(120):
        if backend.is_up():
            return proc
        time.sleep(0.5)
    proc.terminate()
    raise SystemExit("started `ollama serve` but it never answered")


def build_reasoner(spec: str, name: Optional[str], threads: int) -> tuple[Any, Any]:
    """(reasoner, closer) for ``hf:<id>`` or ``ollama:<tag>``."""
    from genetic_coil_arena.experiments import log_triage as lt

    kind, _, ident = spec.partition(":")
    if kind == "hf":
        from genetic_coil_arena.backends import HFPeftBackend

        backend = HFPeftBackend(ident, memory_fraction=0.85, num_threads=threads)
        backend.load()
        short = name or ident.split("/")[-1].lower()
        return lt.HFReasoner(backend, short), (lambda: None)
    if kind == "ollama":
        from genetic_coil_arena.backends import OllamaBackend

        backend = OllamaBackend(ident, keep_alive="10m", timeout=1800.0)
        proc = None
        if not backend.is_up():
            proc = start_ollama(backend)
        counter = None
        try:  # a real tokenizer for budgets when one is installed
            from transformers import AutoTokenizer

            from genetic_coil_arena.envs.logtriage.views import HFCounter

            counter = HFCounter(
                AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B"), "qwen3-tok"
            )
        except Exception:  # noqa: BLE001 - fall back to the character estimate
            counter = None
        reasoner = lt.OllamaReasoner(backend, counter)
        reasoner.name = name or f"ollama:{ident}"

        def close() -> None:
            try:
                backend.unload()
            except Exception:  # noqa: BLE001 - best effort, the server may be gone
                pass
            if proc is not None:
                proc.terminate()

        return reasoner, close
    raise SystemExit(f"--model must start with hf: or ollama:, got {spec!r}")


def cmd_run(args: argparse.Namespace) -> int:
    from genetic_coil_arena.envs.logtriage import rcaeval
    from genetic_coil_arena.envs.logtriage.views import ApproxCounter
    from genetic_coil_arena.experiments import log_triage as lt

    systems = tuple(s for s in args.systems.split(",") if s)
    files = rcaeval.list_incidents(args.cache_dir, systems)
    if not files:
        raise SystemExit(
            "no RCAEval incidents cached; run `python scripts/fetch_rcaeval.py` first"
        )
    out = out_dir(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.jsonl"
    cfg = lt.Settings(
        ctx_tokens=args.ctx,
        spec_ctx_tokens=args.spec_ctx,
        message_tokens=args.message_tokens,
        oracle_ctx_tokens=args.oracle_ctx,
        use_metrics=not args.no_metrics,
    )
    conditions = [c for c in args.conditions.split(",") if c] or list(lt.ALL_CONDITIONS)
    for c in conditions:
        if c not in lt.ALL_CONDITIONS:
            raise SystemExit(
                f"unknown condition {c!r}; choose from {lt.ALL_CONDITIONS}"
            )
    needs_model = any(c not in lt.HEURISTIC_CONDITIONS for c in conditions)
    threads = int(os.environ.get("GCA_WORKERS", "4"))
    reasoner, closer = (None, lambda: None)
    if needs_model:
        reasoner, closer = build_reasoner(args.model, args.model_name, threads)
    model_name = reasoner.name if reasoner else "none"
    counter = reasoner.counter if reasoner else ApproxCounter()

    # incidents of the chosen split
    selected = []
    for path in files:
        inc = rcaeval.load_incident(path)
        if args.split != "all" and rcaeval.split_of(inc) != args.split:
            continue
        selected.append(path)
    selected = pick(selected, args.limit)
    log(
        f"{len(selected)} incidents ({args.split}), conditions {conditions}, model {model_name}"
    )

    done = {
        (r.get("model"), r.get("condition"), r.get("incident"))
        for r in lt.read_jsonl(results_path)
        if r.get("kind") != "overflow"
    }
    overflow_done = {
        r["incident"]
        for r in lt.read_jsonl(results_path)
        if r.get("kind") == "overflow"
    }
    registry = lt.msgs.triage_registry()
    started = time.time()

    def have(model: str, cond: str, inc_id: str) -> bool:
        return (model, cond, inc_id) in done

    def emit(rows: list[dict[str, Any]]) -> None:
        lt.append_jsonl(results_path, rows)
        for r in rows:
            if r.get("kind") != "overflow":
                done.add((r["model"], r["condition"], r["incident"]))

    # ---- non-team conditions, one pass over the incidents
    simple = [c for c in conditions if c not in lt.TEAM_CONDITIONS]
    for k, path in enumerate(selected):
        inc = rcaeval.load_incident(path)
        rows: list[dict[str, Any]] = []
        if (
            not args.no_overflow
            and inc.id not in overflow_done
            and "single_head" in simple
        ):
            rows.append(lt.overflow_stats(inc, counter, cfg.ctx_tokens))
            overflow_done.add(inc.id)
        for cond in simple:
            m = "none" if cond in lt.HEURISTIC_CONDITIONS else model_name
            if have(m, cond, inc.id):
                continue
            lt.msgs.make_goal(
                registry,
                inc,
                cfg.oracle_ctx_tokens if cond == "oracle_selected" else cfg.ctx_tokens,
            )
            if cond in lt.HEURISTIC_CONDITIONS:
                rows.append(lt.run_heuristic(inc, cond).to_dict())
            else:
                try:
                    rows.append(lt.run_single(inc, cond, reasoner, cfg).to_dict())
                except RuntimeError as exc:  # CUDA out of memory at a long context
                    if "out of memory" not in str(exc).lower():
                        raise
                    log(f"{inc.id} {cond}: out of memory, recorded as a miss")
                    import torch

                    torch.cuda.empty_cache()
                    rec = lt.score_record(inc, model_name, cond, [], "unknown", valid=0)
                    rec.extra["error"] = "oom"
                    rows.append(rec.to_dict())
        emit(rows)
        if (k + 1) % 5 == 0 or k == len(selected) - 1:
            log(
                f"simple conditions: {k + 1}/{len(selected)} incidents ({time.time() - started:.0f}s)"
            )

    # ---- team conditions: messages for all incidents first, then decisions + ablations
    for cond in [c for c in conditions if c in lt.TEAM_CONDITIONS]:
        kind = {
            "team_typed": "typed",
            "team_free": "free",
            "team_typed_heur": "typed_heur",
        }[cond]
        msg_path = (
            out
            / f"messages_{model_name.replace(':', '_').replace('/', '_')}_{kind}.jsonl"
        )
        cached = {r["incident"]: r for r in lt.read_jsonl(msg_path)}
        loaded = {}
        for k, path in enumerate(selected):
            inc = rcaeval.load_incident(path)
            loaded[inc.id] = path
            if inc.id in cached:
                continue
            messages, info = lt.specialist_messages(inc, kind, reasoner, cfg)
            row = {"incident": inc.id, "messages": messages, "info": info}
            lt.append_jsonl(msg_path, [row])
            cached[inc.id] = row
            if (k + 1) % 5 == 0:
                log(
                    f"{cond} messages: {k + 1}/{len(selected)} ({time.time() - started:.0f}s)"
                )
        pool: dict[str, list[Any]] = {}
        for inc_id, path in loaded.items():
            inc = rcaeval.load_incident(path)
            for name, m in zip(inc.services, cached[inc_id]["messages"]):
                pool.setdefault(f"{kind}|{name}", []).append(m)
        model_key = "none" if kind == "typed_heur" and reasoner is None else model_name
        for k, (inc_id, path) in enumerate(loaded.items()):
            if have(model_key, cond, inc_id):
                continue
            inc = rcaeval.load_incident(path)
            lt.msgs.make_goal(registry, inc, cfg.ctx_tokens, cfg.message_tokens)
            row = cached[inc_id]
            recs = lt.run_team(
                inc,
                kind,
                reasoner,
                cfg,
                pool=pool,
                ablate=not args.no_ablate,
                precomputed=(row["messages"], row["info"]),
            )
            emit([r.to_dict() for r in recs])
            if (k + 1) % 5 == 0 or k == len(loaded) - 1:
                log(
                    f"{cond} decisions: {k + 1}/{len(loaded)} ({time.time() - started:.0f}s)"
                )

    closer()
    summary = {
        "model": args.model,
        "model_name": model_name,
        "conditions": conditions,
        "systems": list(systems),
        "split": args.split,
        "n_incidents": len(selected),
        "settings": cfg.__dict__,
        "counter": counter.name,
        "wall_s": round(time.time() - started, 1),
        "python": platform.python_version(),
        "commit": subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "finished": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    (out / f"summary_{model_name.replace(':', '_').replace('/', '_')}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    log(f"done in {summary['wall_s']}s; records in {results_path}")
    return 0


def resolve_sources(tokens: list[str]) -> list[Path]:
    """Files, folders, globs and ``runs:<pattern>`` (sibling jobs' ``artifacts``)."""
    found: list[Path] = []
    for tok in tokens:
        if tok.startswith("runs:"):
            run_dir = os.environ.get("GCA_RUN_DIR")
            if not run_dir:
                raise SystemExit("'runs:' needs GCA_RUN_DIR (set by the runner)")
            runs = Path(run_dir).resolve().parents[0]
            for job in sorted(runs.parent.glob(tok[len("runs:") :])):
                found.append(job / "artifacts")
        elif any(c in tok for c in "*?["):
            found += [Path(p) for p in sorted(glob.glob(tok))]
        else:
            found.append(Path(tok))
    return [p for p in found if p.exists()]


def cmd_report(args: argparse.Namespace) -> int:
    from genetic_coil_arena.experiments import log_triage as lt
    from genetic_coil_arena.experiments import log_triage_report as rp

    paths: list[Path] = []
    for src in resolve_sources(args.sources):
        paths += sorted(src.rglob("results*.jsonl")) if src.is_dir() else [src]
    if not paths:
        raise SystemExit("no results files found")
    records: list[dict[str, Any]] = []
    for p in paths:
        records += lt.read_jsonl(p)
    out = Path(args.out)
    analysis = rp.carried_analysis(out)
    report, payload = rp.build_report(records, analysis=analysis, note=args.note)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(report)
    log(f"wrote {out} and {out.with_suffix('.json')} from {len(paths)} file(s)")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run conditions on RCAEval incidents")
    run.add_argument("--model", default="hf:Qwen/Qwen3-1.7B")
    run.add_argument("--model-name", default=None)
    run.add_argument(
        "--conditions",
        default=",".join(
            [
                "heur_baro_logs",
                "heur_surprise_logs",
                "heur_baro_metrics",
                "single_head",
                "single_tail",
                "single_selected",
                "oracle_selected",
                "team_typed",
                "team_free",
                "team_typed_heur",
            ]
        ),
    )
    run.add_argument("--systems", default="SS,OB")
    run.add_argument("--split", choices=["dev", "test", "all"], default="test")
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--out", default=None)
    run.add_argument("--cache-dir", type=Path, default=None)
    run.add_argument("--ctx", type=int, default=4096)
    run.add_argument("--spec-ctx", type=int, default=3072)
    run.add_argument("--message-tokens", type=int, default=128)
    run.add_argument("--oracle-ctx", type=int, default=32768)
    run.add_argument("--no-metrics", action="store_true", help="logs only")
    run.add_argument("--no-ablate", action="store_true")
    run.add_argument("--no-overflow", action="store_true")
    rep = sub.add_parser("report", help="merge result files into the report")
    rep.add_argument("--from", dest="sources", nargs="+", required=True)
    rep.add_argument(
        "--out", default=str(REPO / "docs" / "results" / "r10_log_triage.md")
    )
    rep.add_argument("--note", default=None, help="status line shown under the title")
    args = parser.parse_args(argv)
    return cmd_run(args) if args.cmd == "run" else cmd_report(args)


if __name__ == "__main__":
    sys.exit(main())
