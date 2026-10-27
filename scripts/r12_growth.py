"""R12: growth on real incidents (goals G9 and G10, see ``docs/goals.md``).

G9: a log-novelty specialist joins the G4c incident team without retraining the rest. G10: the
incident desk grows to Train Ticket and the RE3 incidents without prompt changes, with the
joined team behind it.

Sub-commands (each is a runner job):

    team      CPU: G9. Load the frozen G4c state (``--team-from runs:r11-team``), join the
              newcomer, run the old and the joined team on the dev split (RE2 SS/OB and RE3
              repetition 1, every candidate rule), check that the dev selection agrees with the
              rule pinned in ``r12_team.CHOSEN_RULE``, then on the test split (the pinned rule
              only). Writes records_dev.jsonl, records_test.jsonl, g9.json and team.json (the
              frozen state, for G10's ``--team-from runs:r12-team``) to $GCA_RUN_DIR.
    run       GPU (ollama:<tag>) or CPU (keyword, stub): G10. The grown desk on the sealed (or
              dev) request set, the R7a sealed items for the routing check; per-item JSONL and
              summary_<tag>.json.
    report    merge the G9 records and the G10 summaries into docs/results/r12_growth.{md,json}.

    python scripts/r12_growth.py team --team-from runs:r11-team
    python scripts/r12_growth.py run --translator ollama:qwen2.5-coder:14b --team-from runs:r12-team
    python scripts/r12_growth.py run --translator keyword --team-from runs:r12-team
    python scripts/r12_growth.py report --from "runs:r12-*"

Real models run only in the queued jobs. The Ollama model is unloaded (keep_alive 0) when the
job ends. ``--smoke``, ``--limit`` and ``--translator stub`` run a handful of items without any
model (smoke results are never a verdict).
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

import argparse  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Optional  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from r7a_vertical_slice import OllamaServer, VramMonitor, clean, log  # noqa: E402

from genetic_coil_arena.backends.ollama import DEFAULT_HOST, OllamaBackend  # noqa: E402
from genetic_coil_arena.envs.logtriage import novelty as nv  # noqa: E402
from genetic_coil_arena.envs.logtriage import rcaeval  # noqa: E402
from genetic_coil_arena.experiments import r7a_sealed  # noqa: E402
from genetic_coil_arena.experiments import r11_incident as inc  # noqa: E402
from genetic_coil_arena.experiments import r11_pipeline as pipe  # noqa: E402
from genetic_coil_arena.experiments import r11_report as rep  # noqa: E402
from genetic_coil_arena.experiments import r11_team as team  # noqa: E402
from genetic_coil_arena.experiments import r12_incident as inc3  # noqa: E402
from genetic_coil_arena.experiments import r12_report as rep12  # noqa: E402
from genetic_coil_arena.experiments import r12_sealed as sealed  # noqa: E402
from genetic_coil_arena.experiments import r12_team as jt  # noqa: E402
from genetic_coil_arena.experiments.log_triage_report import (  # noqa: E402
    carried_analysis,
)
from genetic_coil_arena.experiments.r7a_domains import build_domains  # noqa: E402
from genetic_coil_arena.experiments.r7a_policy import with_policy_packs  # noqa: E402
from genetic_coil_arena.experiments.r7a_step2 import (  # noqa: E402
    find_file,
    resolve_runs,
)
from genetic_coil_arena.translator import ReverseTranslator  # noqa: E402

PRIMARY = "qwen2.5-coder:14b"
SMALL = "gemma4:e4b"
LABELS = {
    PRIMARY: "qwen2.5-coder:14b",
    SMALL: "gemma4:e4b",
    "keyword": "keyword/regex parser",
}
CTX = {
    "num_ctx": 4096
}  # the incident prompt is long: Ollama would cut the start silently
KEEP = "30m"
WARM_TEXTS = (
    "Please investigate INC-2010 on the online boutique.",
    "sock shop 22 nov 06:08 top 2",
)
ALL_SERVICES = (
    set(inc.SOCK_SHOP_SERVICES)
    | set(inc.ONLINE_BOUTIQUE_SERVICES)
    | set(inc3.TRAIN_TICKET_SERVICES)
)


def run_dir(arg: Optional[str]) -> Path:
    path = Path(arg or os.environ.get("GCA_RUN_DIR") or "outputs/r12")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(clean(obj), indent=1, allow_nan=False), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def commit() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()


def tag_of(translator: str) -> str:
    return (
        translator.split(":", 1)[1] if translator.startswith("ollama:") else translator
    )


def load_team_state(args: argparse.Namespace) -> dict[str, Any]:
    if args.team_from:
        src = resolve_runs(args.team_from)
        return dict(json.loads(find_file(src, "team.json").read_text(encoding="utf-8")))
    log("no --team-from: training the specialist inline (smoke only)")
    return team.train_metric_specialist(
        args.cache_dir, 0, args.limit_train or 40, log=log
    )


def build_joined(
    state: dict[str, Any], rule: str
) -> tuple[team.IncidentTeam, jt.JoinedTeam]:
    specialist = team.load_specialist(state)
    base = team.IncidentTeam(specialist, ALL_SERVICES)
    return base, jt.JoinedTeam(base, ALL_SERVICES, rule)


def pick(files: list[Path], limit: Optional[int]) -> list[Path]:
    """At most ``limit`` files, chosen by a hash of the name (not the file order)."""
    if not limit or limit >= len(files):
        return files
    return sorted(files, key=lambda p: zlib.crc32(p.name.encode()))[:limit]


# ------------------------------------------------------------------------------ team (G9)


def cmd_team(args: argparse.Namespace) -> int:
    out = run_dir(args.out)
    state = load_team_state(args)
    splits = [s for s in args.splits.split(",") if s]
    base, joined = build_joined(state, nv.OLD_RULE)
    hash_before = joined.frozen
    log(f"old team state hash {hash_before}")
    recs: dict[str, list[dict[str, Any]]] = {}
    started = time.time()
    for split in ("dev", "test"):
        path = out / f"records_{split}.jsonl"
        recs[split] = read_jsonl(path)
        if split not in splits:
            continue
        if split == "test":
            chosen = jt.CHOSEN_RULE or args.rule
            if chosen is None:
                raise SystemExit(
                    "the test split needs the pinned rule (r12_team.CHOSEN_RULE)"
                )
            if args.splits_dev_check and recs["dev"]:
                sel = jt.select_rule(recs["dev"])["chosen"]
                if sel != chosen:
                    raise SystemExit(
                        f"dev selection says {sel!r}, the pinned rule is {chosen!r}"
                    )
            rules: list[str] = [nv.OLD_RULE, chosen]
        else:
            rules = list(nv.ALL_RULES)
        files = pick(
            jt.load_incident_files(args.cache_dir, (split,)), args.limit_incidents
        )
        done = {r["id"] for r in recs[split]}
        with path.open("a", encoding="utf-8") as fh:
            for k, f in enumerate(files):
                key = f"{f.parent.name}/{f.stem.rsplit('__', 1)[0]}/{f.stem.rsplit('__', 1)[1]}"
                if key in done:
                    continue
                incident = jt.prepare(rcaeval.load_incident(f))
                row = jt.run_incident(base, joined, incident, rules)
                fh.write(json.dumps(clean(row), ensure_ascii=False) + "\n")
                fh.flush()
                recs[split].append(row)
                if k % 10 == 0 or k + 1 == len(files):
                    log(f"{split}: {k + 1}/{len(files)} ({time.time() - started:.0f}s)")
    joined.assert_frozen()
    hash_after = jt.old_state_hash(base)
    g9: dict[str, Any] = {
        "hash_before": hash_before,
        "hash_after": hash_after,
        "pinned_rule": jt.CHOSEN_RULE,
        "n_dev": len(recs["dev"]),
        "n_test": len(recs["test"]),
        "commit": commit(),
        "python": platform.python_version(),
        "finished": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    if recs["dev"]:
        g9["dev"] = jt.select_rule(recs["dev"])
        log(
            "dev table (pooled top-1): "
            + ", ".join(f"{t['rule']} {t['top1']:.3f}" for t in g9["dev"]["table"])
        )
        log(f"dev selection: {g9['dev']['chosen']} (pinned: {jt.CHOSEN_RULE})")
    write_json(out / "g9.json", g9)
    team_state = dict(state)
    team_state["g9"] = {k: v for k, v in g9.items() if k != "dev"}
    team_state["g9"]["chosen"] = jt.CHOSEN_RULE or args.rule
    write_json(out / "team.json", team_state)
    return 0


# ------------------------------------------------------------------------------ run (G10)


def sealed_items(split: str) -> list[Any]:
    return sealed.load_sealed() if split == "sealed" else sealed.load_dev()


def cmd_run(args: argparse.Namespace) -> int:
    out = run_dir(args.out)
    kind = args.translator.split(":", 1)[0]
    tag = tag_of(args.translator).replace(":", "_").replace("/", "_")
    smoke = bool(args.smoke or args.limit or kind == "stub")
    items = sealed_items(args.split)
    r7a_items = (
        r7a_sealed.load_sealed() if args.split == "sealed" else r7a_sealed.load_dev()
    )
    r7a_items = [i for i in r7a_items if i.expected in ("spec", "clarify")]
    if args.limit:
        items, r7a_items = items[: args.limit], r7a_items[: max(2, args.limit // 2)]
    if args.smoke and not args.limit:
        items, r7a_items = items[:6], r7a_items[:4]

    state = load_team_state(args)
    rule = jt.CHOSEN_RULE or args.rule
    if rule is None:
        raise SystemExit(
            "no pinned rule: set r12_team.CHOSEN_RULE or pass --rule (smoke)"
        )
    pinned = (state.get("g9") or {}).get("chosen")
    if pinned and pinned != rule:
        raise SystemExit(f"team.json says rule {pinned!r}, the code says {rule!r}")
    catalogue = inc3.load_catalogue_r12(args.cache_dir)
    _, joined = build_joined(state, rule)
    executor = jt.JoinedExecutor(catalogue, joined)
    registry = inc3.build_registry_r12()
    domains = with_policy_packs(build_domains(None, 8, 5))
    domains["incident"] = inc3.incident_domain_v3(executor)

    items_path = out / f"items_{tag}.jsonl"
    routes_path = out / f"routes_{tag}.jsonl"
    done = {r["id"] for r in read_jsonl(items_path)}
    done_routes = {r["id"] for r in read_jsonl(routes_path)}
    started = time.time()
    last_log = [started]

    def progress(label: str, k: int, n: int) -> None:
        if time.time() - last_log[0] > 60 or k == n:
            log(f"{label}: {k}/{n} ({time.time() - started:.0f}s)")
            last_log[0] = time.time()

    def finish(
        desk: Any, summariser: Optional[ReverseTranslator], vram: dict[str, Any]
    ) -> None:
        with (
            items_path.open("a", encoding="utf-8") as fh,
            routes_path.open("a", encoding="utf-8") as rh,
        ):
            if summariser is not None:
                for (
                    text
                ) in WARM_TEXTS:  # warm the model, the team and the executor cache
                    desk.handle(text)
                for text in WARM_TEXTS:
                    desk.route(text)
            for k, item in enumerate(items):
                if item.id not in done:
                    row = pipe.process_incident_item(desk, item, summariser).record
                    fh.write(json.dumps(clean(row), ensure_ascii=False) + "\n")
                    fh.flush()
                progress(f"{tag} {args.split}", k + 1, len(items))
            for k, it in enumerate(r7a_items):
                if it.id not in done_routes:
                    row = pipe.route_only(desk, it)
                    rh.write(json.dumps(clean(row), ensure_ascii=False) + "\n")
                    rh.flush()
                progress(f"{tag} routing (R7a items)", k + 1, len(r7a_items))
        write_summary(
            out, tag, args, state, rule, catalogue, items_path, routes_path, vram, smoke
        )

    vram: dict[str, Any] = {}
    if kind == "keyword":
        finish(pipe.KeywordDesk(registry, domains, catalogue), None, vram)
        return 0
    if kind == "stub":
        desk = pipe.Desk(
            pipe.KeywordBackend(inc3.keyword_incident_v3), registry, domains, catalogue
        )
        finish(desk, ReverseTranslator(desk.translator), vram)
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
        backend = OllamaBackend(model, host=args.host, keep_alive=KEEP, options=CTX)
        try:
            desk = pipe.Desk(backend, registry, domains, catalogue, keep_alive=KEEP)
            summariser = ReverseTranslator(desk.translator, keep_alive=KEEP)
            try:
                vram = {"baseline_smi_mib": base_mib}
                finish(desk, summariser, vram)
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
                out,
                tag,
                args,
                state,
                rule,
                catalogue,
                items_path,
                routes_path,
                vram,
                smoke,
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
    routes_path: Path,
    vram: dict[str, Any],
    smoke: bool,
) -> None:
    records = read_jsonl(items_path)
    routes = read_jsonl(routes_path)
    mib = vram.get("peak_smi_mib")
    resolves = sealed.load_resolves_to(
        sealed.SEALED_PATH if args.split == "sealed" else sealed.DEV_PATH
    )
    summary = {
        "tag": tag,
        "model": tag_of(args.translator),
        "translator": args.translator,
        "split": args.split,
        "smoke": smoke,
        "records": records,
        "route_records": routes,
        "metrics": rep12.g10_metrics(records, routes, resolves),
        "vram": vram,
        "vram_gb": None if mib is None else mib / 1024.0,
        "spec_version": 3,
        "sealed_sha256": (
            sealed.file_sha256(sealed.SEALED_PATH) if args.split == "sealed" else None
        ),
        "r7a_sealed_sha256": r7a_sealed.sealed_sha256(),
        "catalogue": {"n": len(catalogue), "fingerprint": catalogue.fingerprint()},
        "team": {
            "rule": rule,
            "chosen": state.get("chosen"),
            "n_train": state.get("n_train"),
            "g9": state.get("g9"),
            "recipe": state.get("recipe"),
        },
        "prompt_audit": inc3.prompt_diff(),
        "commit": commit(),
        "python": platform.python_version(),
        "finished": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    write_json(out / f"summary_{tag}.json", summary)
    ex = summary["metrics"]["all"]["exact"]
    log(
        f"{tag}: wrote summary; exact {ex['p']} (n={ex['n']}), routing {summary['metrics']['all']['routing']['p']}"
    )


# ------------------------------------------------------------------------------ report


def g9_report(sources: list[Path]) -> Optional[dict[str, Any]]:
    folders = [s for s in sources if s.is_dir() and (s / "g9.json").exists()]
    if not folders:
        return None
    folder = folders[0]
    g9 = json.loads((folder / "g9.json").read_text(encoding="utf-8"))
    dev = read_jsonl(folder / "records_dev.jsonl")
    test = read_jsonl(folder / "records_test.jsonl")
    rule = g9.get("pinned_rule")
    if not test or rule is None:
        return None
    return {
        "rule": rule,
        "hash_before": g9["hash_before"],
        "hash_after": g9["hash_after"],
        "dev": jt.select_rule(dev),
        "test": rep12.g9_summary(test, rule),
    }


def cmd_report(args: argparse.Namespace) -> int:
    paths: list[Path] = []
    for src in resolve_runs(args.sources):
        paths += sorted(src.rglob("summary_*.json")) if src.is_dir() else [src]
    summaries = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    summaries = [
        s
        for s in summaries
        if s.get("split") == "sealed"
        and not s.get("smoke")
        and int(s.get("spec_version", 1)) == 3
    ]
    for s in summaries:
        if s["sealed_sha256"] != sealed.SEALED_SHA256:
            raise SystemExit(
                f"{s['tag']}: ran on a sealed file that is not the pinned one"
            )
    report: dict[str, Any] = {
        "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    g9 = g9_report(resolve_runs(args.sources))
    if g9:
        report["g9"] = g9
    if summaries:
        order = {PRIMARY: 0, SMALL: 1, "keyword": 2}
        summaries.sort(key=lambda s: order.get(s["model"], 9))
        resolves = sealed.load_resolves_to()
        systems: dict[str, Any] = {}
        for s in summaries:
            # recompute from the records so the numbers always match the current code
            metrics = rep12.g10_metrics(s["records"], s["route_records"], resolves)
            systems[s["tag"]] = {
                "label": LABELS.get(s["model"], s["model"])
                + (" (refusal policy)" if s["model"] != "keyword" else ""),
                "model": s["model"],
                "metrics": metrics,
                "verdict": rep12.g10_verdict(metrics, s.get("vram_gb")),
                "failures": rep.failures(s["records"]),
                "vram": s.get("vram"),
                "records": s["records"],
                "route_records": s["route_records"],
            }
        first = summaries[0]
        t = first["team"]
        audit = first["prompt_audit"]
        report["g10"] = {
            "n_sealed": len(first["records"]),
            "sealed_sha256": first["sealed_sha256"],
            "r7a_sealed_sha256": first["r7a_sealed_sha256"],
            "catalogue": first["catalogue"],
            "prompt_audit": (
                f"v3 differs from v2 in {len(audit['system_lines_changed'])} system-prompt lines (the clock line and the generated list of allowed systems) and {len(audit['examples_changed'])} few-shot date (the 'yesterday' example, of {audit['n_examples']}); the summary pack is the v2 one: {audit['summary_pack_same']}."
            ),
            "team": (
                f"the G9 joined team (rule `{t['rule']}`) over the G4c trained metric fault specialist (config {t.get('chosen')}, {t.get('n_train')} training incidents)."
            ),
            "systems": systems,
            "commit": first["commit"],
        }
    if "g9" not in report and "g10" not in report:
        raise SystemExit("nothing to report: no g9.json and no full sealed summaries")
    out = Path(args.out or REPO / "docs" / "results" / "r12_growth.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rep12.render(report, carried_analysis(out)), encoding="utf-8")
    payload = json.loads(json.dumps(clean(report), allow_nan=False))
    out.with_suffix(".json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    log(f"wrote {out} and {out.with_suffix('.json')}")
    if "g9" in report:
        log(f"G9: {report['g9']['test']['verdict']['verdict']}")
    for name, s in report.get("g10", {}).get("systems", {}).items():
        log(f"{name}: {s['verdict']['verdict']}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--out", default=None)
        p.add_argument("--cache-dir", type=Path, default=None)
        p.add_argument("--limit-train", type=int, default=None)
        p.add_argument("--team-from", nargs="*", default=[])
        p.add_argument(
            "--rule", default=None, help="smoke only: a rule when none is pinned"
        )

    t = sub.add_parser("team")
    common(t)
    t.add_argument("--splits", default="dev,test", help="dev, test or dev,test")
    t.add_argument("--limit-incidents", type=int, default=None, help="smoke")
    t.add_argument(
        "--no-dev-check",
        dest="splits_dev_check",
        action="store_false",
        help="smoke: do not require the dev selection to equal the pinned rule",
    )
    r = sub.add_parser("run")
    common(r)
    r.add_argument("--translator", required=True, help="ollama:<tag>, keyword or stub")
    r.add_argument("--split", choices=["sealed", "dev"], default="sealed")
    r.add_argument("--host", default=DEFAULT_HOST)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument(
        "--smoke", action="store_true", help="a handful of items (not a result)"
    )
    p = sub.add_parser("report")
    p.add_argument("--from", dest="sources", nargs="+", required=True)
    p.add_argument("--out", default=None, help="default: docs/results/r12_growth.md")
    args = parser.parse_args(argv)
    return {"team": cmd_team, "run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
