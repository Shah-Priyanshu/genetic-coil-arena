"""R10 step 2: a heterogeneous team (metric + log specialists) on RCAEval incidents (goal G4b).

    # CPU only: BARO and the metric-only team with the rule centre
    python scripts/r10_step2.py run --conditions baro_metrics,team_metric_rule --systems SS

    # a log-specialist model: writes its messages, runs the hybrid rule team and the singles
    python scripts/r10_step2.py run --model ollama:qwen2.5-coder:14b \\
        --conditions team_hybrid_rule,single_selected,oracle_selected --oracle-ctx 8192

    # LLM centre over messages written by other jobs
    python scripts/r10_step2.py run --model ollama:qwen2.5-coder:14b --no-generate \\
        --messages-from "runs:r10s2-q3-spec-*" "runs:r10s2-o14-spec-*" \\
        --conditions team_metric_llm,team_hybrid_llm

    # the hybrid team through CommandCentre / envelopes / TraceLog on a few incidents
    python scripts/r10_step2.py runtime --model ollama:qwen2.5-coder:14b --limit 10

    # merge everything into docs/results/r10_step2.{md,json}
    python scripts/r10_step2.py report --from "runs:r10s2-*" --out docs/results/r10_step2.md

Records go to ``<out>/results.jsonl`` (``$GCA_RUN_DIR`` under the runner) and a killed run
resumes from them. Log-specialist messages are cached in ``<out>/logmsgs_<model>.jsonl``.
Needs the RCAEval cache (``python scripts/fetch_rcaeval.py``).
"""

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import r10_log_triage as s1  # noqa: E402  (step 1's model loading and file helpers)

log = s1.log


def safe(name: str) -> str:
    return name.replace(":", "_").replace("/", "_")


def load_logmsgs(paths: list[Path]) -> dict[str, dict[str, dict[str, Any]]]:
    """model -> incident -> {messages, info} from ``logmsgs_*.jsonl`` files."""
    from genetic_coil_arena.experiments import log_triage as lt

    out: dict[str, dict[str, dict[str, Any]]] = {}
    for src in s1.resolve_sources([str(p) for p in paths]):
        files = sorted(src.rglob("logmsgs_*.jsonl")) if src.is_dir() else [src]
        for f in files:
            for row in lt.read_jsonl(f):
                out.setdefault(row["model"], {})[row["incident"]] = row
    return out


def select_incidents(args: argparse.Namespace) -> list[Path]:
    from genetic_coil_arena.envs.logtriage import rcaeval

    systems = tuple(s for s in args.systems.split(",") if s)
    files = rcaeval.list_incidents(args.cache_dir, systems)
    if not files:
        raise SystemExit(
            "no RCAEval incidents cached; run `python scripts/fetch_rcaeval.py` first"
        )
    chosen = []
    for path in files:
        inc = rcaeval.load_incident(path)
        if args.split != "all" and rcaeval.split_of(inc) != args.split:
            continue
        chosen.append(path)
    if args.limit and args.limit < len(chosen):
        # a fixed pseudo-random subset: the sorted list is grouped by fault type
        chosen = sorted(chosen, key=lambda q: zlib.crc32(q.name.encode()))[: args.limit]
        chosen.sort()
    return chosen


def cmd_run(args: argparse.Namespace) -> int:
    from genetic_coil_arena.envs.logtriage import rcaeval
    from genetic_coil_arena.experiments import log_triage as lt
    from genetic_coil_arena.experiments import log_triage2 as t2

    conditions = [c for c in args.conditions.split(",") if c]
    for c in conditions:
        if c not in t2.CONDITIONS2:
            raise SystemExit(f"unknown condition {c!r}; choose from {t2.CONDITIONS2}")
    hybrid = [c for c in conditions if c.startswith("team_hybrid")]
    llm_centre = [c for c in conditions if c.endswith("_llm")]
    singles = [c for c in conditions if c in ("single_selected", "oracle_selected")]
    out = s1.out_dir(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.jsonl"
    cfg = t2.Settings2(
        policy=args.policy,
        log_view_tokens=args.log_view_tokens,
        llm_ablate_dev=args.llm_ablate_dev,
    )
    cfg1 = lt.Settings(oracle_ctx_tokens=args.oracle_ctx)
    threads = int(os.environ.get("GCA_WORKERS", "4"))

    reasoner: Any = None
    closer = lambda: None  # noqa: E731
    if args.model:
        reasoner, closer = s1.build_reasoner(args.model, args.model_name, threads)
    if (
        singles or llm_centre or (hybrid and not args.no_generate)
    ) and reasoner is None:
        raise SystemExit("this selection of conditions needs --model")
    model_name = reasoner.name if reasoner else "none"

    selected = select_incidents(args)
    log(
        f"{len(selected)} incidents ({args.split}), conditions {conditions}, model {model_name}"
    )
    done = {
        (r.get("model"), r.get("condition"), r.get("incident"))
        for r in lt.read_jsonl(results_path)
    }
    started = time.time()

    # ---- pass 1: metric messages (code) and log messages (own model, cached; others loaded)
    spec_msgs = load_logmsgs([Path(p) if not p.startswith("runs:") else p for p in args.messages_from])  # type: ignore[misc]
    own_path = out / f"logmsgs_{safe(model_name)}.jsonl"
    if hybrid and not args.no_generate and reasoner is not None:
        cache = spec_msgs.setdefault(model_name, {})
        for row in lt.read_jsonl(own_path):
            cache.setdefault(row["incident"], row)
    mm_by_inc: dict[str, tuple[list[str], list[dict[str, Any]]]] = {}
    for k, path in enumerate(selected):
        inc = rcaeval.load_incident(path)
        mm = t2.metric_messages(inc)
        mm_by_inc[inc.id] = (list(inc.services), mm)
        if hybrid and not args.no_generate and reasoner is not None:
            cache = spec_msgs.setdefault(model_name, {})
            if inc.id not in cache:
                lm, info = t2.log_messages(inc, mm, reasoner, cfg)
                row = {
                    "model": model_name,
                    "incident": inc.id,
                    "messages": lm,
                    "info": info,
                }
                lt.append_jsonl(own_path, [row])
                cache[inc.id] = row
        if (k + 1) % 5 == 0 or k == len(selected) - 1:
            log(f"messages: {k + 1}/{len(selected)} ({time.time() - started:.0f}s)")

    def pool_for(model: Optional[str]) -> Any:
        per = {}
        for iid, (services, mm) in mm_by_inc.items():
            lm = None
            if model is not None:
                row = spec_msgs.get(model, {}).get(iid)
                if row is None:
                    continue
                lm = row["messages"]
            per[iid] = (services, mm, lm)
        return t2.build_pool(per)

    pools: dict[Optional[str], Any] = {None: pool_for(None)}
    for m in spec_msgs:
        pools[m] = pool_for(m)

    # ---- pass 2: conditions
    for k, path in enumerate(selected):
        inc = rcaeval.load_incident(path)
        services, mm = mm_by_inc[inc.id]
        rows: list[dict[str, Any]] = []
        add = lambda recs, out=rows: out.extend(r.to_dict() for r in recs)  # noqa: E731

        for cond in conditions:
            if cond == "baro_metrics" and ("none", cond, inc.id) not in done:
                add([t2.run_baro(inc)])
            elif cond in singles and (model_name, cond, inc.id) not in done:
                try:
                    add([lt.run_single(inc, cond, reasoner, cfg1)])
                except RuntimeError as exc:  # CUDA out of memory at a long context
                    if "out of memory" not in str(exc).lower():
                        raise
                    log(f"{inc.id} {cond}: out of memory, recorded as a miss")
                    import torch

                    torch.cuda.empty_cache()
                    rec = lt.score_record(inc, model_name, cond, [], "unknown", valid=0)
                    rec.extra["error"] = "oom"
                    add([rec])
            elif cond == "team_metric_rule" and ("none", cond, inc.id) not in done:
                add(
                    t2.team_records(
                        inc, cond, mm, None, "none", None, None, cfg,
                        pool=pools[None], ablate=not args.no_ablate,
                    )
                )  # fmt: skip
            elif cond == "team_metric_llm" and (model_name, cond, inc.id) not in done:
                add(
                    t2.team_records(
                        inc, cond, mm, None, model_name, None, reasoner, cfg,
                        pool=pools[None], ablate=not args.no_ablate,
                    )
                )  # fmt: skip
            elif cond in hybrid:
                for spec_model, per_inc in spec_msgs.items():
                    row = per_inc.get(inc.id)
                    if row is None or (spec_model, cond, inc.id) in done:
                        continue
                    add(
                        t2.team_records(
                            inc, cond, mm, row["messages"], spec_model, row["info"],
                            reasoner if cond.endswith("_llm") else None, cfg,
                            pool=pools[spec_model], ablate=not args.no_ablate,
                        )
                    )  # fmt: skip
        lt.append_jsonl(results_path, rows)
        for r in rows:
            done.add((r["model"], r["condition"], r["incident"]))
        if (k + 1) % 5 == 0 or k == len(selected) - 1:
            log(f"conditions: {k + 1}/{len(selected)} ({time.time() - started:.0f}s)")

    closer()
    summary = {
        "model": args.model,
        "model_name": model_name,
        "conditions": conditions,
        "systems": args.systems,
        "split": args.split,
        "n_incidents": len(selected),
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
    (out / f"summary_{safe(model_name)}_{safe(args.conditions)[:60]}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    log(f"done in {summary['wall_s']}s; records in {results_path}")
    return 0


def cmd_runtime(args: argparse.Namespace) -> int:
    """The hybrid team through CommandCentre + envelopes + TraceLog on a few incidents."""
    from genetic_coil_arena.envs.logtriage import rcaeval
    from genetic_coil_arena.envs.logtriage import team as tm
    from genetic_coil_arena.experiments import log_triage2 as t2
    from genetic_coil_arena.experiments.log_triage2_runtime import RuntimeTeam

    out = s1.out_dir(args.out)
    out.mkdir(parents=True, exist_ok=True)
    threads = int(os.environ.get("GCA_WORKERS", "4"))
    reasoner, closer = s1.build_reasoner(args.model, args.model_name, threads)
    incidents = [rcaeval.load_incident(p) for p in select_incidents(args)]
    trace_path = out / "trace_hybrid.jsonl"
    if trace_path.exists():
        trace_path.unlink()
    team = RuntimeTeam(
        incidents, reasoner, t2.Settings2(policy=args.policy),
        centre="rule", trace_path=trace_path,
    )  # fmt: skip
    rows = []
    for inc in incidents:
        res = team.triage(inc, hybrid=True)
        # the batch path's rule centre on the very same reports must agree
        ranking, faults = tm.rule_decide(
            inc.services, res["metric_reports"], res["log_reports"], args.policy
        )
        res["batch_ranking_equal"] = ranking == res["ranking"]
        res["batch_fault_equal"] = faults["chosen"] == res["fault_type"]
        res["top1"] = int(res["ranking"][0] == inc.root_service)
        res["fault_ok"] = int(res["fault_type"] == inc.fault_type)
        res["root"], res["fault"] = inc.root_service, inc.fault_type
        res.pop("metric_reports"), res.pop("log_reports")
        rows.append(res)
        log(
            f"{inc.id}: top1 {res['top1']} fault {res['fault_ok']} {res['n_envelopes']} envelopes"
        )
    team.close()
    closer()
    n = max(len(rows), 1)
    summary = {
        "model": reasoner.name,
        "n_incidents": len(rows),
        "top1": sum(r["top1"] for r in rows) / n,
        "fault_ok": sum(r["fault_ok"] for r in rows) / n,
        "joint": sum(r["top1"] * r["fault_ok"] for r in rows) / n,
        "batch_ranking_equal": all(r["batch_ranking_equal"] for r in rows),
        "batch_fault_equal": all(r["batch_fault_equal"] for r in rows),
        "envelopes_per_incident": sum(r["n_envelopes"] for r in rows) / n,
        "envelope_bytes_per_incident": sum(r["envelope_bytes"] for r in rows) / n,
        "report_bytes_per_incident": sum(r["report_bytes"] for r in rows) / n,
        "wall_s_per_incident": sum(r["wall_s"] for r in rows) / n,
        "trace": trace_path.name,
        "rows": rows,
    }
    (out / "runtime_summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8"
    )
    log(
        f"runtime check: {json.dumps({k: v for k, v in summary.items() if k != 'rows'})}"
    )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from genetic_coil_arena.experiments import log_triage as lt
    from genetic_coil_arena.experiments import log_triage2_report as rp2
    from genetic_coil_arena.experiments import log_triage_report as rp

    result_files: list[Path] = []
    runtime_files: list[Path] = []
    for src in s1.resolve_sources(args.sources):
        if src.is_dir():
            result_files += sorted(src.rglob("results*.jsonl"))
            runtime_files += sorted(src.rglob("runtime_summary.json"))
        elif src.name.startswith("runtime_summary"):
            runtime_files.append(src)
        else:
            result_files.append(src)
    if not result_files:
        raise SystemExit("no results files found")
    records: list[dict[str, Any]] = []
    for p in result_files:
        records += lt.read_jsonl(p)
    out = Path(args.out)
    report, payload = rp2.build_report(
        records, analysis=rp.carried_analysis(out), note=args.note
    )
    runtime = [json.loads(p.read_text(encoding="utf-8")) for p in runtime_files]
    if runtime:
        payload["runtime"] = [
            {k: v for k, v in r.items() if k != "rows"} for r in runtime
        ]
        r0 = payload["runtime"][0]
        report += (
            "\n## Object layer check\n\n"
            f"The hybrid team with the rule centre ran through `CommandCentre`, envelopes and `TraceLog` "
            f"on {r0['n_incidents']} test incidents with {r0['model']}: top-1 {100 * r0['top1']:.0f}%, "
            f"joint {100 * r0['joint']:.0f}% (a smoke-sized subset, not a result). "
            f"{r0['envelopes_per_incident']:.0f} envelopes per incident, {r0['envelope_bytes_per_incident']:,.0f} bytes "
            f"on the wire, of which {r0['report_bytes_per_incident']:,.0f} bytes are the typed reports. "
            f"The batch path's rule centre on the same reports gave the same ranking "
            f"({r0['batch_ranking_equal']}) and fault ({r0['batch_fault_equal']}).\n"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(report)
    log(f"wrote {out} and {out.with_suffix('.json')} from {len(result_files)} file(s)")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", default=None, help="hf:<id> or ollama:<tag>")
        p.add_argument("--model-name", default=None)
        p.add_argument("--systems", default="SS,OB")
        p.add_argument("--split", choices=["dev", "test", "all"], default="all")
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--out", default=None)
        p.add_argument("--cache-dir", type=Path, default=None)
        p.add_argument("--policy", default="log_c50")

    run = sub.add_parser("run", help="run conditions on RCAEval incidents")
    common(run)
    run.add_argument("--conditions", default="baro_metrics,team_metric_rule")
    run.add_argument("--messages-from", nargs="*", default=[])
    run.add_argument("--no-generate", action="store_true")
    run.add_argument("--no-ablate", action="store_true")
    run.add_argument("--llm-ablate-dev", action="store_true")
    run.add_argument("--oracle-ctx", type=int, default=32768)
    run.add_argument("--log-view-tokens", type=int, default=1500)
    rt = sub.add_parser("runtime", help="hybrid team through the object layer")
    common(rt)
    rep = sub.add_parser("report", help="merge result files into the report")
    rep.add_argument("--from", dest="sources", nargs="+", required=True)
    rep.add_argument("--out", default=str(REPO / "docs" / "results" / "r10_step2.md"))
    rep.add_argument("--note", default=None)
    args = parser.parse_args(argv)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "runtime":
        if not args.model:
            raise SystemExit("runtime needs --model")
        args.split = "test" if args.split == "all" else args.split
        return cmd_runtime(args)
    return cmd_report(args)


if __name__ == "__main__":
    sys.exit(main())
