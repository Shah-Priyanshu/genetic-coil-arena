"""R11: the ecosystem on real incidents (goal G8, see ``docs/goals.md``).

Natural language -> one shared translator (domain, then GoalSpec) -> command centre -> the G4c
trained team on a real RCAEval RE2 incident -> trace view -> reverse translator's summary ->
the summary routed and translated again.

Sub-commands (each is a runner job):

    team      CPU: train the G4c metric fault specialist (Train Ticket + SS/OB repetition 1,
              leave-one-system-out selection; the test split is never fitted on), save it to
              $GCA_RUN_DIR (team.json), write the incident catalogue and the team's own
              joint accuracy on the 120 catalogued incidents through the object layer
    run       GPU (ollama:<tag>) or CPU (keyword): the desk on the sealed (or dev) set, the
              R7a sealed items for the routing check; writes per-item JSONL and summary_<tag>.json
    report    merge summaries into docs/results/r11_ecosystem.{md,json} with the G8 verdict

    python scripts/r11_ecosystem.py team
    python scripts/r11_ecosystem.py run --translator ollama:qwen2.5-coder:14b --team-from runs:r11-team
    python scripts/r11_ecosystem.py run --translator keyword --team-from runs:r11-team
    python scripts/r11_ecosystem.py report --from "runs:r11-*"

R11 v2 (goal G8b, `docs/results/r11b_ecosystem.md`): add `--spec-version 2` to `run` and `report`.
It uses `rca.investigate` v2 (partial dates), the v2 prompts and the fresh sealed set
`r11b_sealed_test.jsonl`; v1 stays reproducible without the flag.

    python scripts/r11_ecosystem.py run --spec-version 2 --translator ollama:qwen2.5-coder:14b --team-from runs:r11-team
    python scripts/r11_ecosystem.py report --spec-version 2 --from runs:r11b-o14-sealed runs:r11b-gemma-sealed runs:r11b-keyword-sealed

Real models run only in the queued jobs. The Ollama model is unloaded (keep_alive 0) when the
job ends. ``--smoke`` and ``--translator stub`` run a handful of items without any model
(smoke results are never a verdict).
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
from pathlib import Path  # noqa: E402
from typing import Any, Optional  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from r7a_vertical_slice import OllamaServer, VramMonitor, clean, log  # noqa: E402

from genetic_coil_arena.backends.ollama import DEFAULT_HOST, OllamaBackend  # noqa: E402
from genetic_coil_arena.experiments import r7a_sealed  # noqa: E402
from genetic_coil_arena.experiments import r11_incident as inc  # noqa: E402
from genetic_coil_arena.experiments import r11_pipeline as pipe  # noqa: E402
from genetic_coil_arena.experiments import r11_report as rep  # noqa: E402
from genetic_coil_arena.experiments import r11_sealed as sealed  # noqa: E402
from genetic_coil_arena.experiments import r11_team as team  # noqa: E402
from genetic_coil_arena.experiments import r11b_incident as incb  # noqa: E402
from genetic_coil_arena.experiments import r11b_sealed as sealedb  # noqa: E402
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
    "Please investigate INC-1010 on the online boutique.",
    "sock shop 20 Jan 13:49 top 2",
)


def run_dir(arg: Optional[str]) -> Path:
    path = Path(arg or os.environ.get("GCA_RUN_DIR") or "outputs/r11")
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


def sealed_module(version: int) -> Any:
    """The sealed/dev loader module of a spec version (1: R11 v1, 2: R11 v2)."""
    return sealedb if version == 2 else sealed


def tag_of(translator: str) -> str:
    return (
        translator.split(":", 1)[1] if translator.startswith("ollama:") else translator
    )


# ------------------------------------------------------------------------------ team


def cmd_team(args: argparse.Namespace) -> int:
    out = run_dir(args.out)
    state = team.train_metric_specialist(
        args.cache_dir, args.seed, args.limit_train, log=log
    )
    catalogue = team.load_catalogue(args.cache_dir)
    specialist = team.load_specialist(state)
    services = set(inc.SOCK_SHOP_SERVICES) | set(inc.ONLINE_BOUTIQUE_SERVICES)
    executor = team.IncidentExecutor(catalogue, team.IncidentTeam(specialist, services))
    log(
        f"evaluating the team on the {len(catalogue)} catalogued incidents (report only)"
    )
    state["catalogue_eval"] = executor.evaluate_all()
    state["catalogue"] = {
        "n": len(catalogue),
        "fingerprint": catalogue.fingerprint(),
        "entries": [e.to_dict() for e in catalogue.entries],
    }
    state["commit"] = commit()
    log(
        f"chosen {state['chosen']}; leave-one-system-out {state['oof_per_system']}; "
        f"catalogue top-1 {state['catalogue_eval']['top1']:.3f}, joint {state['catalogue_eval']['joint']:.3f}"
    )
    write_json(out / "team.json", state)
    return 0


# ------------------------------------------------------------------------------ run


def load_team_state(args: argparse.Namespace) -> dict[str, Any]:
    if args.team_from:
        src = resolve_runs(args.team_from)
        return dict(json.loads(find_file(src, "team.json").read_text(encoding="utf-8")))
    log("no --team-from: training the specialist inline (smoke only)")
    return team.train_metric_specialist(
        args.cache_dir, 0, args.limit_train or 40, log=log
    )


def cmd_run(args: argparse.Namespace) -> int:
    out = run_dir(args.out)
    kind = args.translator.split(":", 1)[0]
    tag = tag_of(args.translator).replace(":", "_").replace("/", "_")
    smoke = bool(args.smoke or args.limit or kind == "stub")
    version = int(args.spec_version)
    mod = sealed_module(version)
    items = mod.load_sealed() if args.split == "sealed" else mod.load_dev()
    r7a_items = (
        r7a_sealed.load_sealed() if args.split == "sealed" else r7a_sealed.load_dev()
    )
    r7a_items = [i for i in r7a_items if i.expected in ("spec", "clarify")]
    if args.limit:
        items, r7a_items = items[: args.limit], r7a_items[: max(2, args.limit // 2)]
    if args.smoke and not args.limit:
        items, r7a_items = items[:6], r7a_items[:4]

    state = load_team_state(args)
    catalogue = team.load_catalogue(args.cache_dir)
    specialist = team.load_specialist(state)
    services = set(inc.SOCK_SHOP_SERVICES) | set(inc.ONLINE_BOUTIQUE_SERVICES)
    executor = team.IncidentExecutor(catalogue, team.IncidentTeam(specialist, services))
    registry = incb.build_registry_r11b() if version == 2 else inc.build_registry_r11()
    domains = with_policy_packs(build_domains(None, 8, 5))
    domains["incident"] = (
        incb.incident_domain_v2(executor)
        if version == 2
        else inc.incident_domain(executor)
    )

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
            out, tag, args, state, catalogue, items_path, routes_path, vram, smoke
        )

    vram: dict[str, Any] = {}
    if kind == "keyword":
        finish(pipe.KeywordDesk(registry, domains, catalogue), None, vram)
        return 0
    if kind == "stub":
        parser = incb.keyword_incident_v2 if version == 2 else None
        desk = pipe.Desk(pipe.KeywordBackend(parser), registry, domains, catalogue)
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
        base = monitor.read()
        monitor.start()
        backend = OllamaBackend(model, host=args.host, keep_alive=KEEP, options=CTX)
        try:
            desk = pipe.Desk(backend, registry, domains, catalogue, keep_alive=KEEP)
            summariser = ReverseTranslator(desk.translator, keep_alive=KEEP)
            peak = None
            try:
                vram = {"baseline_smi_mib": base}
                finish(desk, summariser, vram)
            finally:
                peak = monitor.stop()
            vram.update(
                peak_smi_mib=peak if monitor.available else None,
                model_smi_mib=(
                    (peak - base) if (monitor.available and base is not None) else None
                ),
            )
            write_summary(
                out, tag, args, state, catalogue, items_path, routes_path, vram, smoke
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
    catalogue: Any,
    items_path: Path,
    routes_path: Path,
    vram: dict[str, Any],
    smoke: bool,
) -> None:
    records = read_jsonl(items_path)
    routes = read_jsonl(routes_path)
    mib = vram.get("peak_smi_mib")
    summary = {
        "tag": tag,
        "model": tag_of(args.translator),
        "translator": args.translator,
        "split": args.split,
        "smoke": smoke,
        "records": records,
        "route_records": routes,
        "metrics": rep.system_metrics(records, routes),
        "vram": vram,
        "vram_gb": None if mib is None else mib / 1024.0,
        "spec_version": int(args.spec_version),
        "sealed_sha256": (
            sealed_module(int(args.spec_version)).file_sha256(
                sealed_module(int(args.spec_version)).SEALED_PATH
            )
            if args.split == "sealed"
            else None
        ),
        "r7a_sealed_sha256": r7a_sealed.sealed_sha256(),
        "catalogue": {"n": len(catalogue), "fingerprint": catalogue.fingerprint()},
        "team": {
            "chosen": state.get("chosen"),
            "oof_per_system": state.get("oof_per_system"),
            "n_train": state.get("n_train"),
            "catalogue_eval": state.get("catalogue_eval"),
            "recipe": state.get("recipe"),
        },
        "commit": commit(),
        "python": platform.python_version(),
        "finished": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    write_json(out / f"summary_{tag}.json", summary)
    ex = summary["metrics"]["exact"]
    log(
        f"{tag}: wrote summary; exact {ex['p']} (n={ex['n']}), routing {summary['metrics']['routing']['p']}"
    )


# ------------------------------------------------------------------------------ report


def cmd_report(args: argparse.Namespace) -> int:
    paths: list[Path] = []
    for src in resolve_runs(args.sources):
        paths += sorted(src.rglob("summary_*.json")) if src.is_dir() else [src]
    summaries = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    version = int(args.spec_version)
    summaries = [
        s
        for s in summaries
        if s.get("split") == "sealed"
        and not s.get("smoke")
        and int(s.get("spec_version", 1)) == version
    ]
    pinned = sealed_module(version).SEALED_SHA256
    for s in summaries:
        if s["sealed_sha256"] != pinned:
            raise SystemExit(
                f"{s['tag']}: ran on a sealed file that is not the pinned one"
            )
    if not summaries:
        raise SystemExit("no full sealed-set summaries found")
    order = {PRIMARY: 0, SMALL: 1, "keyword": 2}
    summaries.sort(key=lambda s: order.get(s["model"], 9))
    systems: dict[str, Any] = {}
    for s in summaries:
        # recompute from the records so the numbers always match the current code
        metrics = rep.system_metrics(s["records"], s["route_records"])
        systems[s["tag"]] = {
            "label": LABELS.get(s["model"], s["model"])
            + (" (refusal policy)" if s["model"] != "keyword" else ""),
            "model": s["model"],
            "metrics": metrics,
            "verdict": rep.verdict(metrics, s.get("vram_gb")),
            "failures": rep.failures(s["records"]),
            "vram": s.get("vram"),
            "records": s["records"],
            "route_records": s["route_records"],
        }
    first = summaries[0]
    team_info = first["team"]
    ce = team_info.get("catalogue_eval") or {}
    per_system = ", ".join(
        f"{k} {v:.1%}" for k, v in (team_info.get("oof_per_system") or {}).items()
    )
    team_text = (
        f"the G4c trained metric fault specialist (config {team_info.get('chosen')}, trained on "
        f"{team_info.get('n_train')} Train Ticket and SS/OB repetition-1 incidents; "
        f"leave-one-system-out accuracy {per_system})"
    )
    if ce.get("n"):
        team_text += (
            f"; through the object layer it gets top-1 {ce['top1']:.1%} and joint "
            f"{ce['joint']:.1%} on all {ce['n']} catalogued incidents (R10 step 3 reported 68.3 joint)"
        )
    team_text += "."
    report: dict[str, Any] = {
        "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_sealed": len(first["records"]),
        "sealed_sha256": first["sealed_sha256"],
        "r7a_sealed_sha256": first["r7a_sealed_sha256"],
        "catalogue": first["catalogue"],
        "team": team_text,
        "systems": systems,
        "commit": first["commit"],
    }
    if version == 2:
        report.update(
            goal="G8b",
            title="R11 v2: the ecosystem on real incidents, partial dates resolved by the desk",
            sealed_file="r11b_sealed_test.jsonl",
            command="python scripts/r11_ecosystem.py report --spec-version 2",
        )
    out = Path(
        args.out
        or REPO
        / "docs"
        / "results"
        / ("r11b_ecosystem.md" if version == 2 else "r11_ecosystem.md")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rep.render(report, carried_analysis(out)), encoding="utf-8")
    payload = json.loads(json.dumps(clean(report), allow_nan=False))
    out.with_suffix(".json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    log(f"wrote {out} and {out.with_suffix('.json')} from {len(summaries)} summar(ies)")
    for name, s in systems.items():
        log(f"{name}: {s['verdict']['verdict']}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--out", default=None)
        p.add_argument("--cache-dir", type=Path, default=None)
        p.add_argument("--limit-train", type=int, default=None)

    t = sub.add_parser("team")
    common(t)
    t.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("run")
    common(r)
    r.add_argument("--translator", required=True, help="ollama:<tag>, keyword or stub")
    r.add_argument("--split", choices=["sealed", "dev"], default="sealed")
    r.add_argument("--spec-version", type=int, choices=[1, 2], default=1)
    r.add_argument("--team-from", nargs="*", default=[])
    r.add_argument("--host", default=DEFAULT_HOST)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument(
        "--smoke", action="store_true", help="a handful of items (not a result)"
    )
    p = sub.add_parser("report")
    p.add_argument("--from", dest="sources", nargs="+", required=True)
    p.add_argument(
        "--out", default=None, help="default: docs/results/r11[b]_ecosystem.md"
    )
    p.add_argument("--spec-version", type=int, choices=[1, 2], default=1)
    args = parser.parse_args(argv)
    return {"team": cmd_team, "run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
