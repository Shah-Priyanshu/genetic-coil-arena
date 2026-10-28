"""R12c: the desk checks the translator's literals against the request (goal G10b).

Desk v4 is the G10 desk (same prompts, model and spec v3) plus a deterministic literal check
(``envs/logtriage/literal_check.py``). One translator run per model; every item is scored twice
from the same raw translation, check off (the G10 desk) and check on (v4).

Sub-commands (each is a runner job):

    run       GPU (ollama:<tag>) or CPU (keyword, stub): the sealed (or dev) request set through
              the chain, paired. Writes items_<tag>.jsonl (the check-on record with the
              check-off record under ``off``, the raw and checked params, the rules fired) and
              summary_<tag>.json to $GCA_RUN_DIR.
    report    merge the summaries into docs/results/r12c_literal_check.{md,json}; ``--posthoc``
              also applies the check to the stored raw outputs of earlier sealed runs.

    python scripts/r12c_desk.py run --translator ollama:qwen2.5-coder:14b --split dev --team-from runs:r12-team
    python scripts/r12c_desk.py run --translator keyword --team-from runs:r12-team
    python scripts/r12c_desk.py report --from "runs:r12c-*" --posthoc runs:r12-o14-sealed runs:r11b-o14-sealed

Real models run only in the queued jobs (the Ollama model is unloaded when the job ends).
``--smoke``, ``--limit`` and ``--translator stub`` run a handful of items without any model
(smoke results are never a verdict).
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

import argparse  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Optional  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import r12_growth as g  # noqa: E402
from r7a_vertical_slice import OllamaServer, VramMonitor, clean, log  # noqa: E402

from genetic_coil_arena.backends.ollama import DEFAULT_HOST, OllamaBackend  # noqa: E402
from genetic_coil_arena.envs.logtriage import literal_check as lc  # noqa: E402
from genetic_coil_arena.envs.logtriage.catalogue import (  # noqa: E402
    DESK_NOW,
    DESK_NOW_R12,
)
from genetic_coil_arena.experiments import r11_incident as inc  # noqa: E402
from genetic_coil_arena.experiments import r11_pipeline as pipe  # noqa: E402
from genetic_coil_arena.experiments import r12_incident as inc3  # noqa: E402
from genetic_coil_arena.experiments import r12_team as jt  # noqa: E402
from genetic_coil_arena.experiments import r12c_desk as desk4  # noqa: E402
from genetic_coil_arena.experiments import r12c_report as rep4  # noqa: E402
from genetic_coil_arena.experiments import r12c_sealed as sealed  # noqa: E402
from genetic_coil_arena.experiments.log_triage_report import (  # noqa: E402
    carried_analysis,
)
from genetic_coil_arena.experiments.r7a_domains import build_domains  # noqa: E402
from genetic_coil_arena.experiments.r7a_policy import with_policy_packs  # noqa: E402
from genetic_coil_arena.experiments.r7a_step2 import (  # noqa: E402
    resolve_runs,
)
from genetic_coil_arena.translator import ReverseTranslator  # noqa: E402

POSTHOC_FILE = "items_qwen2.5-coder_14b.jsonl"


def split_path(split: str) -> Path:
    return sealed.SEALED_PATH if split == "sealed" else sealed.DEV_PATH


def cmd_run(args: argparse.Namespace) -> int:
    out = g.run_dir(args.out)
    kind = args.translator.split(":", 1)[0]
    tag = g.tag_of(args.translator).replace(":", "_").replace("/", "_")
    smoke = bool(args.smoke or args.limit or kind == "stub")
    items = sealed.load_sealed() if args.split == "sealed" else sealed.load_dev()
    if args.limit:
        items = items[: args.limit]
    elif args.smoke:
        items = items[:6]
    state = g.load_team_state(args)
    rule = jt.CHOSEN_RULE or args.rule
    if rule is None:
        raise SystemExit(
            "no pinned rule: set r12_team.CHOSEN_RULE or pass --rule (smoke)"
        )
    pinned = (state.get("g9") or {}).get("chosen")
    if pinned and pinned != rule:
        raise SystemExit(f"team.json says rule {pinned!r}, the code says {rule!r}")
    catalogue = inc3.load_catalogue_r12(args.cache_dir)
    _, joined = g.build_joined(state, rule)
    executor = jt.JoinedExecutor(catalogue, joined)
    registry = inc3.build_registry_r12()
    domains = with_policy_packs(build_domains(None, 8, 5))
    domains["incident"] = inc3.incident_domain_v3(executor)

    items_path = out / f"items_{tag}.jsonl"
    done = {r["id"] for r in g.read_jsonl(items_path)}
    started = time.time()
    last_log = [started]
    raw_cache: dict[Any, Any] = {}

    def finish(inner: Any, summariser: Optional[ReverseTranslator], vram: dict) -> None:
        with items_path.open("a", encoding="utf-8") as fh:
            if summariser is not None:
                for text in g.WARM_TEXTS:  # warm the model, the team and the executor
                    inner.handle(text)
            for k, item in enumerate(items):
                if item.id not in done:
                    row = desk4.process_pair(
                        inner, DESK_NOW_R12, item, summariser, raw_cache
                    )
                    fh.write(json.dumps(clean(row), ensure_ascii=False) + "\n")
                    fh.flush()
                if time.time() - last_log[0] > 60 or k + 1 == len(items):
                    log(
                        f"{tag} {args.split}: {k + 1}/{len(items)} ({time.time() - started:.0f}s)"
                    )
                    last_log[0] = time.time()
        write_summary(out, tag, args, state, rule, catalogue, items_path, vram, smoke)

    vram: dict[str, Any] = {}
    if kind == "keyword":
        finish(pipe.KeywordDesk(registry, domains, catalogue), None, vram)
        return 0
    if kind == "stub":
        inner = pipe.Desk(
            pipe.KeywordBackend(inc3.keyword_incident_v3),
            registry,
            domains,
            catalogue,
            route_system=inc3.ROUTE_SYSTEM_V3,
        )
        finish(inner, ReverseTranslator(inner.translator), vram)
        return 0
    if kind != "ollama":
        raise SystemExit(f"unknown translator {args.translator!r}")
    model = args.translator.split(":", 1)[1]
    monitor = VramMonitor()
    with OllamaServer(args.host) as server:
        server.require(model)
        server.unload_all()
        time.sleep(1.0)
        base_mib = monitor.read()
        monitor.start()
        backend = OllamaBackend(model, host=args.host, keep_alive=g.KEEP, options=g.CTX)
        try:
            inner = pipe.Desk(
                backend,
                registry,
                domains,
                catalogue,
                keep_alive=g.KEEP,
                route_system=inc3.ROUTE_SYSTEM_V3,
            )
            summariser = ReverseTranslator(inner.translator, keep_alive=g.KEEP)
            vram = {"baseline_smi_mib": base_mib}
            try:
                finish(inner, summariser, vram)
            finally:
                peak = monitor.stop()
            vram.update(
                peak_smi_mib=peak if monitor.available else None,
                model_smi_mib=(
                    (peak - base_mib)
                    if (monitor.available and base_mib is not None)
                    else None
                ),
            )
            write_summary(
                out, tag, args, state, rule, catalogue, items_path, vram, smoke
            )
        finally:
            try:
                backend.unload()  # keep_alive 0
            except Exception as exc:  # noqa: BLE001
                log(f"unload failed: {exc}")
    return 0


def write_summary(
    out: Path,
    tag: str,
    args: argparse.Namespace,
    state: dict[str, Any],
    rule: str,
    catalogue: Any,
    items_path: Path,
    vram: dict[str, Any],
    smoke: bool,
) -> None:
    rows = g.read_jsonl(items_path)
    path = split_path(args.split)
    resolves, traps = sealed.load_resolves_to(path), sealed.load_traps(path)
    mib = vram.get("peak_smi_mib")
    summary = {
        "tag": tag,
        "model": g.tag_of(args.translator),
        "translator": args.translator,
        "split": args.split,
        "smoke": smoke,
        "records": rows,
        "metrics": rep4.g10b_metrics(rows, resolves, traps),
        "vram": vram,
        "vram_gb": None if mib is None else mib / 1024.0,
        "spec_version": 3,
        "sealed_sha256": (
            sealed.file_sha256(sealed.SEALED_PATH) if args.split == "sealed" else None
        ),
        "catalogue": {"n": len(catalogue), "fingerprint": catalogue.fingerprint()},
        "team": {"rule": rule, "g9": state.get("g9")},
        "prompt_audit": inc3.prompt_diff(),
        "rules": list(lc.RULES),
        "commit": g.commit(),
        "python": platform.python_version(),
        "finished": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    g.write_json(out / f"summary_{tag}.json", summary)
    m = summary["metrics"]
    log(
        f"{tag}: exact off {m['off']['exact']['p']} on {m['on']['exact']['p']} "
        f"(n={m['on']['exact']['n']}), fixed {len(m['paired']['fixed'])} broke "
        f"{len(m['paired']['broke'])}, silent wrong off {len(m['off']['silent_wrong'])} "
        f"on {len(m['on']['silent_wrong'])}"
    )


# ------------------------------------------------------------------------------ report


def posthoc_rows(sources: list[str]) -> dict[str, Any]:
    registry = inc3.build_registry_r12()
    schema = registry.get(inc.INVESTIGATE).params_schema
    out: dict[str, Any] = {}
    for src in resolve_runs(sources):
        files = sorted(src.rglob(POSTHOC_FILE)) if src.is_dir() else [src]
        for f in files:
            name = src.parent.name if src.is_dir() else f.stem
            now = DESK_NOW if "r11b" in name else DESK_NOW_R12
            out[name] = rep4.posthoc(
                g.read_jsonl(f), registry, inc.INVESTIGATE, now, schema
            )
    return out


def cmd_report(args: argparse.Namespace) -> int:
    paths: list[Path] = []
    for src in resolve_runs(args.sources):
        paths += sorted(src.rglob("summary_*.json")) if src.is_dir() else [src]
    summaries = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    summaries = [
        s for s in summaries if s.get("split") == "sealed" and not s.get("smoke")
    ]
    if not summaries:
        raise SystemExit("nothing to report: no full sealed summaries")
    for s in summaries:
        if s["sealed_sha256"] != sealed.SEALED_SHA256:
            raise SystemExit(
                f"{s['tag']}: ran on a sealed file that is not the pinned one"
            )
    order = {g.PRIMARY: 0, g.SMALL: 1, "keyword": 2}
    summaries.sort(key=lambda s: order.get(s["model"], 9))
    resolves, traps = sealed.load_resolves_to(), sealed.load_traps()
    systems: dict[str, Any] = {}
    for s in summaries:
        metrics = rep4.g10b_metrics(s["records"], resolves, traps)  # from the records
        systems[s["tag"]] = {
            "label": g.LABELS.get(s["model"], s["model"]),
            "model": s["model"],
            "metrics": metrics,
            "verdict": rep4.verdict(metrics),
            "records": s["records"],
            "resolves_to": resolves,
            "vram": s.get("vram"),
        }
    first = summaries[0]
    report: dict[str, Any] = {
        "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "g10b": {
            "n_sealed": len(first["records"]),
            "n_traps": len(traps),
            "sealed_sha256": first["sealed_sha256"],
            "catalogue": first["catalogue"],
            "rules_text": lc.RULES_TEXT,
            "systems": systems,
            "commit": first["commit"],
        },
    }
    if args.posthoc:
        report["posthoc"] = posthoc_rows(args.posthoc)
    out = Path(args.out or REPO / "docs" / "results" / "r12c_literal_check.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rep4.render(report, carried_analysis(out)), encoding="utf-8")
    for (
        s
    ) in systems.values():  # the records are in the md/json only once, in compact form
        s["records"] = [
            {k: v for k, v in r.items() if k not in ("summary", "facts", "off")}
            | {"off_status": r["off"]["status"], "off_exact": r["off"].get("exact")}
            for r in s["records"]
        ]
        s.pop("resolves_to")
    payload = json.loads(json.dumps(clean(report), allow_nan=False))
    out.with_suffix(".json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    log(f"wrote {out} and {out.with_suffix('.json')}")
    for name, s in systems.items():
        log(f"{name}: {s['verdict']['verdict']}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", default=None)
    r.add_argument("--cache-dir", type=Path, default=None)
    r.add_argument("--limit-train", type=int, default=None)
    r.add_argument("--team-from", nargs="*", default=[])
    r.add_argument(
        "--rule", default=None, help="smoke only: a rule when none is pinned"
    )
    r.add_argument("--translator", required=True, help="ollama:<tag>, keyword or stub")
    r.add_argument("--split", choices=["sealed", "dev"], default="sealed")
    r.add_argument("--host", default=DEFAULT_HOST)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument(
        "--smoke", action="store_true", help="a handful of items (not a result)"
    )
    p = sub.add_parser("report")
    p.add_argument("--from", dest="sources", nargs="+", required=True)
    p.add_argument(
        "--posthoc", nargs="*", default=[], help="earlier sealed runs (runs:<id>)"
    )
    p.add_argument(
        "--out", default=None, help="default: docs/results/r12c_literal_check.md"
    )
    args = parser.parse_args(argv)
    return {"run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
