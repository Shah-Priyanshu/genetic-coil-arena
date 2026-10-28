"""R12b (goal G9b): the attributing newcomer joins the frozen incident team. CPU only, no LLM.

    python scripts/r12b_growth.py team --team-from runs:r11-team [--groups dev,tt,re2_test,re3_test]
    python scripts/r12b_growth.py report --from runs:r12b-team [--out docs/results/r12b_growth.md]

``team`` runs the old team and the v2 joined team (every candidate rule on dev; the reference
rows and the pinned rule elsewhere) over the requested groups, appends one record per incident to
``records_<group>.jsonl`` in the run directory (``$GCA_RUN_DIR``; a rerun skips finished
incidents) and, when every group is present, writes the report next to them. ``report`` renders
``docs/results/r12b_growth.{md,json}`` from the records (the analysis block is carried over).
The fresh and test groups need the rule pinned in ``r12b_team.CHOSEN_RULE_B``; the dev check
recomputes the selection from the dev records and refuses to go on if it disagrees.
"""

import argparse
import datetime
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import r12_growth as base  # noqa: E402

from genetic_coil_arena.envs.logtriage import novelty as nv  # noqa: E402
from genetic_coil_arena.envs.logtriage import novelty2 as n2  # noqa: E402
from genetic_coil_arena.envs.logtriage import rcaeval  # noqa: E402
from genetic_coil_arena.experiments import r11_team as team  # noqa: E402
from genetic_coil_arena.experiments import r12_team as jt  # noqa: E402
from genetic_coil_arena.experiments import r12b_report as rep  # noqa: E402
from genetic_coil_arena.experiments import r12b_team as bt  # noqa: E402
from genetic_coil_arena.experiments.log_triage_report import (  # noqa: E402
    carried_analysis,
)
from genetic_coil_arena.experiments.r7a_step2 import resolve_runs  # noqa: E402

log, clean = base.log, base.clean
ALL_GROUPS = ",".join(bt.GROUPS)


def build_joined_b(
    state: dict[str, Any],
) -> tuple[team.IncidentTeam, bt.JoinedTeamV2]:
    specialist = team.load_specialist(state)
    old = team.IncidentTeam(specialist, base.ALL_SERVICES)
    return old, bt.JoinedTeamV2(old, base.ALL_SERVICES, nv.OLD_RULE)


def cmd_team(args: argparse.Namespace) -> int:
    out = base.run_dir(args.out)
    state = base.load_team_state(args)
    groups = [g for g in args.groups.split(",") if g]
    old, joined = build_joined_b(state)
    hash_before = joined.frozen
    log(f"old team state hash {hash_before}")
    recs = {g: base.read_jsonl(out / f"records_{g}.jsonl") for g in bt.GROUPS}
    pinned = args.rule or bt.CHOSEN_RULE_B
    started = time.time()
    for group in bt.GROUPS:
        if group not in groups:
            continue
        if group == "dev":
            rules = list(n2.ALL_RULES2)
        else:
            if pinned is None:
                raise SystemExit("this group needs the pinned rule (CHOSEN_RULE_B)")
            if recs["dev"]:
                sel = bt.dev_table(recs["dev"])["chosen"]
                if sel != pinned:
                    raise SystemExit(
                        f"dev selection says {sel!r}, pinned is {pinned!r}"
                    )
            rules = [nv.OLD_RULE, n2.REF_RULE, n2.ALONE_RULE, pinned]
        files = base.pick(bt.load_files([group], args.cache_dir), args.limit_incidents)
        done = {r["id"] for r in recs[group]}
        path = out / f"records_{group}.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            for k, f in enumerate(files):
                case, rep_no = f.stem.rsplit("__", 1)
                if f"{f.parent.name}/{case}/{rep_no}" in done:
                    continue
                incident = bt.prepare(rcaeval.load_incident(f))
                row = bt.run_incident_b(old, joined, incident, rules)
                fh.write(json.dumps(clean(row), ensure_ascii=False) + "\n")
                fh.flush()
                recs[group].append(row)
                if k % 10 == 0 or k + 1 == len(files):
                    log(f"{group}: {k + 1}/{len(files)} ({time.time() - started:.0f}s)")
    joined.assert_frozen()
    hash_after = jt.old_state_hash(old)
    meta: dict[str, Any] = {
        "hash_before": hash_before,
        "hash_after": hash_after,
        "pinned_rule": pinned,
        "n": {g: len(v) for g, v in recs.items()},
        "commit": base.commit(),
        "python": platform.python_version(),
        "finished": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    if recs["dev"]:
        meta["dev"] = bt.dev_table(recs["dev"])
        log(
            "dev (pooled top-1): "
            + ", ".join(f"{t['rule']} {t['top1']:.3f}" for t in meta["dev"]["table"])
            + f"; chosen {meta['dev']['chosen']}"
        )
    base.write_json(out / "g9b.json", meta)
    if all(recs[g] for g in ("dev", "tt", "re2_test", "re3_test")):
        write_report(meta, recs, out / "r12b_growth.md")
    return 0


def write_report(meta: dict[str, Any], recs: dict[str, list[Any]], out: Path) -> None:
    rule = meta["pinned_rule"]
    test = [r for g in ("tt", "re2_test", "re3_test") for r in recs[g]]
    report = {
        "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "rule": rule,
        "hash_before": meta["hash_before"],
        "hash_after": meta["hash_after"],
        "dev": bt.dev_table(recs["dev"]),
        "summary": rep.summarise(test, rule),
        "commit": meta.get("commit"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rep.render(report, carried_analysis(out)), encoding="utf-8")
    payload = json.loads(json.dumps(clean(report), allow_nan=False))
    out.with_suffix(".json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    log(f"wrote {out} and {out.with_suffix('.json')}")
    log(f"G9b: {report['summary']['verdict']['verdict']}")


def cmd_report(args: argparse.Namespace) -> int:
    folders = [s for s in resolve_runs(args.sources) if s.is_dir()]
    folder = next((f for f in folders if (f / "g9b.json").exists()), None)
    if folder is None:
        for f in folders:
            hits = list(f.rglob("g9b.json"))
            if hits:
                folder = hits[0].parent
                break
    if folder is None:
        raise SystemExit("no g9b.json in the sources")
    meta = json.loads((folder / "g9b.json").read_text(encoding="utf-8"))
    recs = {g: base.read_jsonl(folder / f"records_{g}.jsonl") for g in bt.GROUPS}
    out = Path(args.out or base.REPO / "docs" / "results" / "r12b_growth.md")
    write_report(meta, recs, out)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("team")
    t.add_argument("--out", default=None)
    t.add_argument("--cache-dir", type=Path, default=None)
    t.add_argument("--limit-train", type=int, default=None)
    t.add_argument("--team-from", nargs="*", default=[])
    t.add_argument("--groups", default=ALL_GROUPS)
    t.add_argument("--limit-incidents", type=int, default=None, help="smoke")
    t.add_argument("--rule", default=None, help="smoke only: overrides the pinned rule")
    r = sub.add_parser("report")
    r.add_argument("--from", dest="sources", nargs="+", required=True)
    r.add_argument("--out", default=None, help="default: docs/results/r12b_growth.md")
    args = parser.parse_args(argv)
    return {"team": cmd_team, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
