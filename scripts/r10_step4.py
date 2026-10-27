"""R10 step 4 (goal G4d): LLM log specialists on RCAEval RE3, where the signal is in the logs.

    # data, non-LLM systems and the log-signal gate (CPU, no model)
    python scripts/r10_step4.py gate --out outputs/r10s4_gate

    # a model as log specialist (+ rule centre, LLM centre, single-model conditions)
    python scripts/r10_step4.py run --model ollama:qwen2.5-coder:14b --shard 0/3 \\
        --conditions team4_rule,team4_llm,single_all,single_logs
    python scripts/r10_step4.py run --model hf:Qwen/Qwen3-1.7B --conditions team4_rule --shard 0/2

    # merge everything (gate.json + results*.jsonl) into docs/results/r10_step4.{md,json}
    python scripts/r10_step4.py report --from "runs:r10s4-*" --out docs/results/r10_step4.md

Records go to ``<out>/results.jsonl`` (``$GCA_RUN_DIR/artifacts`` under the runner) and a
killed run resumes from them; the log specialists' reports are cached in
``<out>/logmsgs_<model>.jsonl``. Needs the RCAEval RE3 cache
(``python scripts/fetch_rcaeval.py --systems RE3-SS,RE3-OB,RE3-TT``).
"""

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import r10_log_triage as s1  # noqa: E402  (model loading, file helpers)

log = s1.log


def safe(name: str) -> str:
    return name.replace(":", "_").replace("/", "_")


def parse_shard(text: Optional[str]) -> Optional[tuple[int, int]]:
    if not text:
        return None
    k, _, n = text.partition("/")
    shard = (int(k), int(n))
    if not (0 <= shard[0] < shard[1]):
        raise SystemExit(f"bad --shard {text!r}; use i/n with 0 <= i < n")
    return shard


def cmd_gate(args: argparse.Namespace) -> int:
    from genetic_coil_arena.experiments import log_triage4 as l4
    from genetic_coil_arena.experiments import log_triage4_report as rp4

    started = time.time()
    dev, test = l4.load_incidents(args.cache_dir, limit=args.limit)
    incs = dev + test
    cache = l4.all_parts(incs)
    fusion = l4.pick_fusion(dev, cache)
    records = l4.non_llm_records(incs, cache, fusion)
    payload: dict[str, Any] = {
        "kind": "gate",
        "data": l4.describe(incs),
        "gate": l4.gate_summary(l4.gate_rows(incs)),
        "gate_test": l4.gate_summary(l4.gate_rows(test)),
        "fusion": fusion,
        "fusion_label": l4.fusion_label(fusion),
        "n_dev": len(dev),
        "n_test": len(test),
        "seconds": round(time.time() - started, 1),
    }
    out = s1.out_dir(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate.json").write_text(
        json.dumps(payload, indent=1, default=float), encoding="utf-8"
    )
    with open(out / "results_nonllm.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), default=float) + "\n")
    print(rp4.gate_text(payload))
    log(f"fusion {payload['fusion_label']}; wrote {out}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from genetic_coil_arena.experiments import log_triage as lt
    from genetic_coil_arena.experiments import log_triage4 as l4
    from genetic_coil_arena.experiments import log_triage4_teams as t4x

    conditions = [c for c in args.conditions.split(",") if c]
    for c in conditions:
        if c not in (*t4x.TEAM4, *t4x.SINGLE4):
            raise SystemExit(f"unknown condition {c!r}")
    want_rule = "team4_rule" in conditions
    want_centre = "team4_llm" in conditions
    singles = [c for c in conditions if c in t4x.SINGLE4]
    kind = args.model.partition(":")[0]
    if (want_centre or singles) and kind != "ollama":
        raise SystemExit("team4_llm and the single conditions need an ollama: model")
    out = s1.out_dir(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.jsonl"
    cfg = t4x.Settings4(log_view_tokens=args.log_view_tokens, ctx_tokens=args.ctx)
    threads = int(os.environ.get("GCA_WORKERS", "4"))
    reasoner, closer = s1.build_reasoner(args.model, args.model_name, threads)
    model_name = reasoner.name

    systems = tuple(s for s in args.systems.split(",") if s)
    dev, test = l4.load_incidents(
        args.cache_dir, systems, args.limit, parse_shard(args.shard)
    )
    if args.split != "all":
        dev, test = (dev, []) if args.split == "dev" else ([], test)
    incs = sorted(dev + test, key=lambda i: i.id)
    log(f"{len(incs)} incidents, conditions {conditions}, model {model_name}")
    done = {
        (r.get("model"), r.get("condition"), r.get("incident"))
        for r in lt.read_jsonl(results_path)
    }
    started = time.time()

    # ---- pass 1: metric messages (code) and log reports (the model, cached)
    own_path = out / f"logmsgs_{safe(model_name)}.jsonl"
    cached = {row["incident"]: row for row in lt.read_jsonl(own_path)}
    per_inc: dict[str, tuple[Any, Any, Any]] = {}
    infos: dict[str, dict[str, Any]] = {}
    if want_rule or want_centre:
        for k, inc in enumerate(incs):
            mm = t4x.metric_messages(inc)
            row = cached.get(inc.id)
            if row is None:
                lm, info = t4x.log_messages4(inc, mm, reasoner, cfg)
                row = {
                    "model": model_name,
                    "incident": inc.id,
                    "messages": lm,
                    "info": info,
                }
                lt.append_jsonl(own_path, [row])
            per_inc[inc.id] = (list(inc.services), mm, row["messages"])
            infos[inc.id] = row["info"]
            log(
                f"reports {k + 1}/{len(incs)} {inc.id} valid "
                f"{row['info']['valid']}/{inc.n_services} ({time.time() - started:.0f}s)"
            )
    pool = (
        t4x.build_pool4(per_inc, {i.id: i.root_service for i in incs})
        if per_inc
        else {}
    )

    # ---- pass 2: conditions
    for k, inc in enumerate(incs):
        rows: list[dict[str, Any]] = []
        if per_inc and (
            (model_name, "team4_rule", inc.id) not in done
            or (model_name, "team4_llm", inc.id) not in done
        ):
            _, mm, lm = per_inc[inc.id]
            need_rule = want_rule and (model_name, "team4_rule", inc.id) not in done
            need_centre = want_centre and (model_name, "team4_llm", inc.id) not in done
            if need_rule or need_centre:
                recs = t4x.team_records4(
                    inc, model_name, mm, lm, infos[inc.id],
                    reasoner if need_centre else None, cfg, pool=pool,
                    ablate=not args.no_ablate, rule=need_rule,
                )  # fmt: skip
                rows += [r.to_dict() for r in recs]
        for cond in singles:
            if (model_name, cond, inc.id) not in done:
                rows.append(t4x.run_single4(inc, cond, reasoner, cfg).to_dict())
        lt.append_jsonl(results_path, rows)
        for r in rows:
            done.add((r["model"], r["condition"], r["incident"]))
        log(f"conditions {k + 1}/{len(incs)} {inc.id} ({time.time() - started:.0f}s)")
    closer()
    summary = {
        "model": args.model,
        "model_name": model_name,
        "conditions": conditions,
        "systems": args.systems,
        "shard": args.shard,
        "n_incidents": len(incs),
        "settings": cfg.__dict__,
        "wall_s": round(time.time() - started, 1),
        "python": platform.python_version(),
        "commit": subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "finished": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    tag = safe(f"{model_name}_{args.shard or 'all'}")
    (out / f"summary_{tag}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    log(f"done in {summary['wall_s']}s; records in {results_path}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from genetic_coil_arena.experiments import log_triage as lt
    from genetic_coil_arena.experiments import log_triage4_report as rp4
    from genetic_coil_arena.experiments import log_triage_report as rp

    result_files: list[Path] = []
    gate_files: list[Path] = []
    for src in s1.resolve_sources(args.sources):
        if src.is_dir():
            result_files += sorted(src.rglob("results*.jsonl"))
            gate_files += sorted(src.rglob("gate.json"))
        elif src.name == "gate.json":
            gate_files.append(src)
        else:
            result_files.append(src)
    if not result_files:
        raise SystemExit("no results files found")
    records: list[dict[str, Any]] = []
    for p in result_files:
        records += lt.read_jsonl(p)
    gate = json.loads(gate_files[0].read_text(encoding="utf-8")) if gate_files else None
    out = Path(args.out)
    report, payload = rp4.build_report(
        records, gate=gate, analysis=rp.carried_analysis(out), note=args.note
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps(payload, indent=1, default=float), encoding="utf-8"
    )
    print(report)
    log(f"wrote {out} and {out.with_suffix('.json')} from {len(result_files)} file(s)")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate", help="data description, non-LLM systems and the gate")
    g.add_argument("--out", default=None)
    g.add_argument("--limit", type=int, default=None)
    g.add_argument("--cache-dir", type=Path, default=None)
    run = sub.add_parser("run", help="LLM log specialists, centres and single models")
    run.add_argument("--model", required=True, help="hf:<id> or ollama:<tag>")
    run.add_argument("--model-name", default=None)
    run.add_argument("--systems", default="RE3-SS,RE3-OB,RE3-TT")
    run.add_argument("--split", choices=["dev", "test", "all"], default="all")
    run.add_argument(
        "--shard", default=None, help="i/n: every n-th incident by id hash"
    )
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--out", default=None)
    run.add_argument("--cache-dir", type=Path, default=None)
    run.add_argument("--conditions", default="team4_rule")
    run.add_argument("--no-ablate", action="store_true")
    run.add_argument("--ctx", type=int, default=6144)
    run.add_argument("--log-view-tokens", type=int, default=1500)
    rep = sub.add_parser("report", help="merge result files into the report")
    rep.add_argument("--from", dest="sources", nargs="+", required=True)
    rep.add_argument("--out", default=str(REPO / "docs" / "results" / "r10_step4.md"))
    rep.add_argument("--note", default=None)
    args = parser.parse_args(argv)
    return {"gate": cmd_gate, "run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
