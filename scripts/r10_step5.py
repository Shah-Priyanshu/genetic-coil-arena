"""R10 step 5 (goal G4e): an LLM final reader over the non-LLM fusion's top-3 on RCAEval RE3.

    # one model reads the fusion's top-3 with evidence blocks (+ shuffle ablations)
    python scripts/r10_step5.py run --model ollama:qwen2.5-coder:14b --split all

    # merge the models' result files into docs/results/r10_step5.{md,json}
    python scripts/r10_step5.py report --from runs:r10s5-o14 runs:r10s5-g4b runs:r10s5-q30 \\
        --out docs/results/r10_step5.md

The fusion weights are fitted on dev (repetition 1) exactly as in step 4. Records go to
``<out>/results.jsonl`` (``$GCA_RUN_DIR/artifacts`` under the runner) and a killed run
resumes from them; the fusion's own records go to ``results_nonllm.jsonl``. With
``--split all`` the dev split runs first and the test split is skipped if fewer than half of
the dev replies are valid (a broken prompt must not spend the test run).
Needs the RCAEval RE3 cache (``python scripts/fetch_rcaeval.py --systems RE3-SS,RE3-OB,RE3-TT``).
"""

import argparse
import datetime
import json
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
MIN_DEV_VALID = 0.5


def run_split(
    incs: list[Any],
    cond_list: list[str],
    ctx: dict[str, Any],
    done: set,
    started: float,
    label: str,
) -> float:
    """Run every missing (condition, incident); returns the valid-reply rate of this call."""
    from genetic_coil_arena.experiments import log_triage as lt
    from genetic_coil_arena.experiments import log_triage5 as l5

    valid: list[int] = []
    for k, inc in enumerate(incs):
        order, scores = l5.fusion_order(inc, ctx["cache"][inc.id], ctx["fusion"])
        mm = l5.metric_messages(inc)
        rows = []
        for cond in cond_list:
            if (ctx["name"], cond, inc.id) in done:
                continue
            rec = l5.read_shortlist(
                inc, ctx["name"], cond, order, scores, mm, ctx["reasoner"], ctx["cfg"]
            )
            rows.append(rec.to_dict())
            valid.append(rec.valid)
        lt.append_jsonl(ctx["results_path"], rows)
        for r in rows:
            done.add((r["model"], r["condition"], r["incident"]))
        log(f"{label} {k + 1}/{len(incs)} {inc.id} ({time.time() - started:.0f}s)")
    return sum(valid) / len(valid) if valid else 1.0


def cmd_run(args: argparse.Namespace) -> int:
    from genetic_coil_arena.experiments import log_triage as lt
    from genetic_coil_arena.experiments import log_triage4 as l4
    from genetic_coil_arena.experiments import log_triage5 as l5

    cond_list = [c for c in args.conditions.split(",") if c]
    for c in cond_list:
        if c not in l5.CONDITIONS5:
            raise SystemExit(f"unknown condition {c!r}; choose from {l5.CONDITIONS5}")
    if not args.model.startswith("ollama:"):
        raise SystemExit("--model must be ollama:<tag>")
    out = s1.out_dir(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.jsonl"
    started = time.time()

    systems = tuple(s for s in args.systems.split(",") if s)
    dev, test = l4.load_incidents(args.cache_dir, systems, args.limit)
    cache = l4.all_parts(dev + test)
    fusion = l4.pick_fusion(dev, cache)
    log(
        f"{len(dev)} dev / {len(test)} test incidents; fusion {l4.fusion_label(fusion)}"
    )
    fusion_rows = [
        r.to_dict()
        for r in l4.non_llm_records(dev + test, cache, fusion)
        if r.condition == "nonllm_fusion"
    ]
    with open(out / "results_nonllm.jsonl", "w", encoding="utf-8") as f:
        for r in fusion_rows:
            f.write(json.dumps(r, default=float) + "\n")

    reasoner, closer = s1.build_reasoner(args.model, args.model_name, 1)
    done = {
        (r.get("model"), r.get("condition"), r.get("incident"))
        for r in lt.read_jsonl(results_path)
    }
    ctx = {
        "name": reasoner.name,
        "reasoner": reasoner,
        "cfg": l5.Settings5(evidence_tokens=args.evidence_tokens),
        "cache": cache,
        "fusion": fusion,
        "results_path": results_path,
    }
    dev_valid = 1.0
    try:
        if args.split in ("dev", "all"):
            dev_valid = run_split(dev, cond_list, ctx, done, started, "dev")
            log(f"dev valid-reply rate {dev_valid:.2f}")
        if args.split in ("test", "all"):
            if args.split == "all" and dev_valid < MIN_DEV_VALID:
                raise SystemExit(
                    f"dev valid-reply rate {dev_valid:.2f} < {MIN_DEV_VALID}: "
                    "not spending the test split on a broken prompt"
                )
            run_split(test, cond_list, ctx, done, started, "test")
    finally:
        closer()  # unloads the Ollama model (keep_alive 0)
    summary = {
        "model": args.model,
        "model_name": reasoner.name,
        "conditions": cond_list,
        "split": args.split,
        "fusion": l4.fusion_label(fusion),
        "n_dev": len(dev),
        "n_test": len(test),
        "settings": ctx["cfg"].__dict__,
        "wall_s": round(time.time() - started, 1),
        "python": platform.python_version(),
        "commit": subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "finished": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    tag = reasoner.name.replace(":", "_").replace("/", "_")
    (out / f"summary_{tag}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    log(f"done in {summary['wall_s']}s; records in {results_path}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from genetic_coil_arena.experiments import log_triage as lt
    from genetic_coil_arena.experiments import log_triage5_report as rp5
    from genetic_coil_arena.experiments import log_triage_report as rp

    result_files: list[Path] = []
    for src in s1.resolve_sources(args.sources):
        if src.is_dir():
            result_files += sorted(src.rglob("results*.jsonl"))
        else:
            result_files.append(src)
    if not result_files:
        raise SystemExit("no results files found")
    records: list[dict[str, Any]] = []
    for p in result_files:
        records += lt.read_jsonl(p)
    out = Path(args.out)
    report, payload = rp5.build_report(
        records, analysis=rp.carried_analysis(out), note=args.note
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
    run = sub.add_parser("run", help="an LLM reads the fusion's top-3 shortlist")
    run.add_argument("--model", required=True, help="ollama:<tag>")
    run.add_argument("--model-name", default=None)
    run.add_argument("--systems", default="RE3-SS,RE3-OB,RE3-TT")
    run.add_argument("--split", choices=["dev", "test", "all"], default="all")
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--out", default=None)
    run.add_argument("--cache-dir", type=Path, default=None)
    run.add_argument(
        "--conditions",
        default=",".join(("rerank3", "rerank3:shuffle", "rerank3:shuffle_blind")),
    )
    run.add_argument("--evidence-tokens", type=int, default=1500)
    rep = sub.add_parser("report", help="merge result files into the report")
    rep.add_argument("--from", dest="sources", nargs="+", required=True)
    rep.add_argument("--out", default=str(REPO / "docs" / "results" / "r10_step5.md"))
    rep.add_argument("--note", default=None)
    args = parser.parse_args(argv)
    return {"run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
