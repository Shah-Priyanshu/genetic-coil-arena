"""R13 (goals G11a, G11b): the frozen G4c team on fresh RE1 incidents, and a learned ranking centre.

    replicate  G11a. The frozen G4c team (``--team-from runs:r11-team``: the trained fault
               specialist, the rule centre) on every RE1 incident against BARO; joint accuracy,
               paired bootstrap, per-fault confusion, metric-kind coverage. Fetches RE1 if absent.
    train      G11b. Leave-one-system-out cross-validation of the six ranker configs on RE1,
               pick one, fit it on all RE1 (``ranker.json``). Reads no RE2 incident.
    test       G11b. The trained ranker (``--train-from runs:r13-train``) once on RE2 (SS/OB
               repetitions 2-3, Train Ticket all repetitions) against the rule centre.
    report     Merge the jobs' JSON into docs/results/r13_real_centre.{md,json}.

    python scripts/r13_real_centre.py replicate --team-from runs:r11-team
    python scripts/r13_real_centre.py train --team-from runs:r11-team
    python scripts/r13_real_centre.py test --team-from runs:r11-team --train-from runs:r13-train
    python scripts/r13_real_centre.py report --from "runs:r13-*" --out docs/results/r13_real_centre.md

``--limit N`` runs N incidents per system (smoke; never a verdict). Artifacts go to
``--out`` or ``$GCA_RUN_DIR``.
"""

import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from genetic_coil_arena.envs.logtriage import fault_features as ff  # noqa: E402
from genetic_coil_arena.envs.logtriage.rcaeval import FAULT_TYPES  # noqa: E402
from genetic_coil_arena.experiments import r11_team as team  # noqa: E402
from genetic_coil_arena.experiments import r12_team as r12  # noqa: E402
from genetic_coil_arena.experiments import r13_real_centre as r13  # noqa: E402
from genetic_coil_arena.experiments.log_triage_report import (  # noqa: E402
    ANALYSIS_END,
    ANALYSIS_START,
    carried_analysis,
)
from genetic_coil_arena.experiments.r7a_step2 import (  # noqa: E402
    find_file,
    resolve_runs,
)

SYSTEMS = r13.RE1_SYS


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def clean(obj: Any) -> Any:
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.generic):
        return clean(obj.item())
    return obj


def out_dir(arg: Optional[str]) -> Path:
    path = Path(arg or os.environ.get("GCA_RUN_DIR") or "outputs/r13")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(clean(obj), indent=1, allow_nan=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def team_state(args: argparse.Namespace) -> dict[str, Any]:
    if not args.team_from:
        raise SystemExit("--team-from runs:r11-team (the frozen G4c state) is required")
    src = resolve_runs([args.team_from])
    return dict(read_json(find_file(src, "team.json")))


def cache(args: argparse.Namespace) -> Optional[Path]:
    return Path(args.cache_dir) if args.cache_dir else None


# ---------------------------------------------------------------------------- replicate


def cmd_replicate(args: argparse.Namespace) -> int:
    out = out_dir(args.out)
    r13.ensure_re1(cache(args), log)
    state = team_state(args)
    digest = r13.assert_frozen(state)
    specialist = team.load_specialist(state)
    incs = r13.load_re1(cache(args), args.limit)
    log(f"{len(incs)} RE1 incidents; frozen model {digest[:12]}")
    services = sorted({s for i in incs for s in i.services})
    tm_team = team.IncidentTeam(specialist, services)
    hash0 = r12.old_state_hash(tm_team)
    rows = []
    for k, inc in enumerate(incs):
        res = tm_team.triage(inc)
        b_rank, b_fault = r13.baro_decision(inc)
        rows.append(
            {
                "id": inc.id,
                "system": inc.system.split("-")[-1],
                "root": inc.root_service,
                "fault": inc.fault_type,
                "team_top": res["ranking"][0],
                "team_fault": res["fault_type"],
                "baro_top": b_rank[0],
                "baro_fault": b_fault,
            }
        )
        if (k + 1) % 25 == 0:
            log(f"{k + 1}/{len(incs)}")
    hash1 = r12.old_state_hash(tm_team)
    if hash0 != hash1 or r13.model_sha256(state) != digest:
        raise SystemExit("the frozen team changed during the run")
    tm_team.close()
    sysv = np.array([r["system"] for r in rows])

    def vec(prefix: str, what: str) -> np.ndarray:
        if what == "top1":
            return np.array([r[f"{prefix}_top"] == r["root"] for r in rows], float)
        if what == "fault":
            return np.array([r[f"{prefix}_fault"] == r["fault"] for r in rows], float)
        return vec(prefix, "top1") * vec(prefix, "fault")

    res: dict[str, Any] = {}
    for what in ("joint", "top1", "fault"):
        a, b = vec("team", what), vec("baro", what)
        res[what] = {"pooled": r13.paired(a, b)}
        for s in SYSTEMS:
            res[what][s] = r13.paired(a[sysv == s], b[sysv == s])
    true = [r["fault"] for r in rows]
    res["confusion"] = {
        "team": r13.confusion(true, [r["team_fault"] for r in rows]),
        "baro": r13.confusion(true, [r["baro_fault"] for r in rows]),
    }
    res["per_fault_joint"] = {
        f: {
            "n": int(sum(r["fault"] == f for r in rows)),
            "team": (
                float(
                    np.mean(
                        [
                            x
                            for x, r in zip(vec("team", "joint"), rows)
                            if r["fault"] == f
                        ]
                    )
                )
                if any(r["fault"] == f for r in rows)
                else None
            ),
            "baro": (
                float(
                    np.mean(
                        [
                            x
                            for x, r in zip(vec("baro", "joint"), rows)
                            if r["fault"] == f
                        ]
                    )
                )
                if any(r["fault"] == f for r in rows)
                else None
            ),
        }
        for f in FAULT_TYPES
    }
    res["same_top_as_baro"] = int(sum(r["team_top"] == r["baro_top"] for r in rows))
    res["verdict"] = r13.g11a_verdict(res)
    res.update(
        n=len(rows),
        smoke=bool(args.limit),
        model_sha256=digest,
        state_hash=hash0,
        re1=r13.re1_statistics(cache(args)),
        coverage_re1={
            s: r13.kind_coverage([i for i in incs if i.system.endswith(s)])
            for s in SYSTEMS
        },
        records=rows,
    )
    write_json(out / "g11a.json", res)
    v = res["verdict"]["verdict"]
    j = res["joint"]["pooled"]
    log(
        f"G11a {v}: team {j['a']:.3f} baro {j['b']:.3f} diff {j['diff']:+.3f} [{j['lo']:+.3f}, {j['hi']:+.3f}]"
    )
    return 0


# ------------------------------------------------------------------------------ train


def cmd_train(args: argparse.Namespace) -> int:
    out = out_dir(args.out)
    r13.ensure_re1(cache(args), log)
    state = team_state(args)
    digest = r13.assert_frozen(state)
    model = team.load_specialist(state).model
    incs = r13.load_re1(cache(args), args.limit)
    data = r13.RankData(incs, model)
    log(f"{len(incs)} RE1 incidents, {data.x.shape[2]} features")
    table = r13.loso_table(data)
    best = r13.choose_config(table)
    for i, row in enumerate(table):
        log(
            f"config {i} {row['features']} l2={row['l2']}: LOSO top-1 {row['mean_top1']:.3f}"
            + ("  <- chosen" if i == best else "")
        )
    cfg = table[best]
    pooled = r13.loso_pooled(data, cfg)
    ranker = r13.ConditionalLogit(cfg["features"], cfg["l2"]).fit(data).to_dict()
    ranker["fault_model_sha256"] = digest
    write_json(out / "ranker.json", {**ranker, "sha256": r13.ranker_sha256(ranker)})
    write_json(
        out / "g11b_loso.json",
        {
            "table": table,
            "chosen": best,
            "chosen_config": {"features": cfg["features"], "l2": cfg["l2"]},
            "chosen_loso": pooled,
            "n": len(incs),
            "smoke": bool(args.limit),
            "ranker_sha256": r13.ranker_sha256(ranker),
            "feature_names": list(r13.feature_names(cfg["features"])),
            "weights": dict(zip(r13.feature_names(cfg["features"]), ranker["w"])),
        },
    )
    p = pooled["pooled"]
    log(
        f"chosen LOSO pooled: learned {p['a']:.3f} baro {p['b']:.3f} diff {p['diff']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}]"
    )
    return 0


# ------------------------------------------------------------------------------- test


def cmd_test(args: argparse.Namespace) -> int:
    out = out_dir(args.out)
    state = team_state(args)
    digest = r13.assert_frozen(state)
    model = team.load_specialist(state).model
    src = resolve_runs([args.train_from])
    saved = read_json(find_file(src, "ranker.json"))
    claimed = saved.pop("sha256")
    if r13.ranker_sha256(saved) != claimed or saved["fault_model_sha256"] != digest:
        raise SystemExit(
            "ranker.json does not match its hash or the frozen fault model"
        )
    ranker = r13.ConditionalLogit.from_dict(saved)
    incs = r13.load_re2_test(cache(args), args.limit)
    log(f"{len(incs)} RE2 test incidents; ranker {claimed[:12]}")
    data = r13.RankData(incs, model)
    top = ranker.top(data)
    base = r13.baro_top(data)
    learned = (top == data.y).astype(float)
    rule = (base == data.y).astype(float)
    res: dict[str, Any] = {"top1": {"pooled": r13.paired(learned, rule)}}
    for s in SYSTEMS:
        sel = data.systems == s
        res["top1"][s] = r13.paired(learned[sel], rule[sel])
    # extra (not a criterion): joint accuracy with the specialist's fault for the chosen service
    fault_hit = np.array(
        [
            FAULT_TYPES[int(np.argmax(data.probs[k][top[k]]))]
            == data.incs[k].fault_type
            for k in range(len(top))
        ],
        float,
    )
    rule_hit = np.array(
        [
            FAULT_TYPES[int(np.argmax(data.probs[k][base[k]]))]
            == data.incs[k].fault_type
            for k in range(len(base))
        ],
        float,
    )
    res["joint"] = {"pooled": r13.paired(learned * fault_hit, rule * rule_hit)}
    res["verdict"] = r13.g11b_verdict(res)
    res.update(
        n=len(incs),
        smoke=bool(args.limit),
        counts={s: int((data.systems == s).sum()) for s in SYSTEMS},
        ranker_sha256=claimed,
        chosen_config={"features": saved["feature_set"], "l2": saved["l2"]},
        coverage_re2={
            s: r13.kind_coverage([i for i in incs if i.system == s]) for s in SYSTEMS
        },
        specialist_saw_tt=True,
        records=[
            {
                "id": data.incs[k].id,
                "system": str(data.systems[k]),
                "root": data.incs[k].root_service,
                "learned_top": data.incs[k].services[int(top[k])],
                "rule_top": data.incs[k].services[int(base[k])],
            }
            for k in range(len(top))
        ],
    )
    write_json(out / "g11b_test.json", res)
    p = res["top1"]["pooled"]
    log(
        f"G11b {res['verdict']['verdict']}: learned {p['a']:.3f} rule {p['b']:.3f} diff {p['diff']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}]"
    )
    return 0


# ----------------------------------------------------------------------------- report


def pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}"


def pd(r: dict[str, Any]) -> str:
    return f"{100 * r['diff']:+.1f} [{100 * r['lo']:+.1f}, {100 * r['hi']:+.1f}]"


def cmd_report(args: argparse.Namespace) -> int:
    srcs = resolve_runs(args.sources)

    def get(name: str) -> Optional[dict[str, Any]]:
        try:
            return dict(read_json(find_file(srcs, name)))
        except FileNotFoundError:
            return None

    a, tr, te = get("g11a.json"), get("g11b_loso.json"), get("g11b_test.json")
    out_md = Path(args.out)
    keep = carried_analysis(out_md)
    lines = [
        "# R13: the frozen G4c team on fresh incidents, and a learned centre (G11a, G11b)",
        "",
        f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M} by `scripts/r13_real_centre.py report`. "
        "RCAEval RE1 (metrics only, no logs; resource faults cpu, mem, disk, delay, loss), "
        "never used by an earlier goal. Differences are in percentage points with paired bootstrap "
        "95% CIs (10,000 resamples over incidents).",
        "",
    ]
    report: dict[str, Any] = {}
    if a:
        report["g11a"] = a
        v = a["verdict"]
        lines += [
            f"## G11a: does G4c replicate on RE1? **{v['verdict']}**",
            "",
            f"Frozen fault specialist SHA-256 `{a['model_sha256'][:16]}`, team state hash `{a['state_hash'][:16]}` "
            f"(asserted unchanged after the run). {a['n']} incidents"
            + (" (SMOKE: not a verdict)" if a["smoke"] else "")
            + ". Pass: pooled joint gain >= 5 points over BARO with CI above 0, and the same in each system. "
            "Kill: pooled CI includes 0 or lies below.",
            "",
            "| Metric | System | Team | BARO | Difference [95% CI] |",
            "|---|---|---|---|---|",
        ]
        for what, label in (
            ("joint", "Joint"),
            ("top1", "Root top-1"),
            ("fault", "Fault type"),
        ):
            for s in ("pooled", *SYSTEMS):
                r = a[what][s]
                lines.append(
                    f"| {label} | {s} | {pct(r['a'])} | {pct(r['b'])} | {pd(r)} |"
                )
        lines += [
            "",
            f"Team top service equals BARO's in {a['same_top_as_baro']} of {a['n']} incidents (by construction).",
            "",
            "### Joint accuracy per fault type",
            "",
            "| Fault | n | Team | BARO |",
            "|---|---|---|---|",
        ]
        for f, r in a["per_fault_joint"].items():
            if r["n"]:
                lines.append(
                    f"| {f} | {r['n']} | {pct(r['team'])} | {pct(r['baro'])} |"
                )
        for who in ("team", "baro"):
            lines += [
                "",
                f"### Fault confusion, {who} (rows: true, columns: predicted)",
                "",
                "| true | " + " | ".join(FAULT_TYPES) + " |",
                "|---|" + "---|" * len(FAULT_TYPES),
            ]
            for t, row in a["confusion"][who].items():
                if sum(row.values()):
                    lines.append(
                        f"| {t} | "
                        + " | ".join(str(row[p]) for p in FAULT_TYPES)
                        + " |"
                    )
        lines += [
            "",
            "### RE1 data",
            "",
            "| System | Incidents | Cases found | Parse failures | Metric kinds (incidents having each) |",
            "|---|---|---|---|---|",
        ]
        for s, r in a["re1"].items():
            lines.append(
                f"| {s} | {r['n_incidents']} | {r['n_cases_found']} | {r['n_failed']} | "
                + ", ".join(f"{k} {n}" for k, n in r["metric_kinds"].items())
                + " |"
            )
        lines += [
            "",
            "RE1 has no logs. Renamed kinds (Online Boutique): "
            + ", ".join(
                f"`{k}` as `{v}`"
                for k, v in next(iter(a["re1"].values()))["kind_alias"].items()
            )
            + ".",
        ]
    if tr:
        report["g11b_train"] = tr
        cov_re1 = (a or {}).get("coverage_re1")
        lines += [
            "",
            "## G11b: LOSO cross-validation on RE1",
            "",
            f"{tr['n']} RE1 incidents{' (SMOKE)' if tr['smoke'] else ''}. Mean held-out top-1 over the three left-out systems; selection rule: highest mean, ties to the compact set then the larger L2.",
            "",
            "| # | Features | L2 | "
            + " | ".join(f"{s} held out" for s in SYSTEMS)
            + " | Mean |",
            "|---|---|---|" + "---|" * (len(SYSTEMS) + 1),
        ]
        for i, row in enumerate(tr["table"]):
            lines.append(
                f"| {i}{' *' if i == tr['chosen'] else ''} | {row['features']} | {row['l2']} | "
                + " | ".join(pct(row["per_system"][s]["top1"]) for s in SYSTEMS)
                + f" | {pct(row['mean_top1'])} |"
            )
        lines.append(
            "| BARO | | | "
            + " | ".join(pct(tr["table"][0]["per_system"][s]["baro"]) for s in SYSTEMS)
            + " | |"
        )
        c = tr["chosen_loso"]
        lines += [
            "",
            "Chosen config (*), learned minus BARO top-1 on the held-out systems: "
            + "; ".join(f"{s} {pd(c[s])}" for s in ("pooled", *SYSTEMS))
            + ".",
        ]
        del cov_re1
    if te:
        report["g11b_test"] = te
        lines += [
            "",
            f"## G11b: the ranker on RE2 **{te['verdict']['verdict']}**",
            "",
            f"Ranker `{te['ranker_sha256'][:16]}` ({te['chosen_config']['features']}, L2 {te['chosen_config']['l2']}), trained on RE1 only, tested once on {te['n']} RE2 incidents "
            f"({', '.join(f'{s} {n}' for s, n in te['counts'].items())})"
            + (" (SMOKE)" if te["smoke"] else "")
            + ". These RE2 splits were used by earlier goals (G4c to G10b), but never to train or select this ranker. "
            "The frozen fault specialist was trained on RE2 Train Ticket, so its probabilities are optimistic there. "
            "Pass: pooled top-1 gain >= 3 points over the rule centre (BARO ranking) with CI above 0, and each system's CI lower bound above -5. Kill: pooled CI includes 0 or lies below.",
            "",
            "| System | Learned | Rule centre | Difference [95% CI] |",
            "|---|---|---|---|",
        ]
        for s in ("pooled", *SYSTEMS):
            r = te["top1"][s]
            lines.append(f"| {s} | {pct(r['a'])} | {pct(r['b'])} | {pd(r)} |")
        j = te["joint"]["pooled"]
        lines += [
            "",
            f"Not a criterion: joint accuracy with the specialist's fault for the chosen service, pooled: learned {pct(j['a'])}, rule {pct(j['b'])}, {pd(j)}.",
        ]
    if a and te:
        lines += [
            "",
            "## Metric-kind coverage, RE1 against RE2",
            "",
            "Share of incidents whose root service reports each kind.",
            "",
            "| Group | n | " + " | ".join(ff.KINDS) + " |",
            "|---|---|" + "---|" * len(ff.KINDS),
        ]
        groups = [(f"RE1 {s}", a["coverage_re1"][s]) for s in SYSTEMS] + [
            (f"RE2 {s}", te["coverage_re2"][s]) for s in SYSTEMS
        ]
        for name, cv in groups:
            lines.append(
                f"| {name} | {cv['n_incidents']} | "
                + " | ".join(pct(cv["root_has_kind"][k]) for k in ff.KINDS)
                + " |"
            )
        report["coverage"] = dict(groups)
    lines += [
        "",
        ANALYSIS_START,
        keep or "(analysis to be written after the results are in)",
        ANALYSIS_END,
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    write_json(out_md.with_suffix(".json"), report)
    log(f"wrote {out_md}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (
        ("replicate", cmd_replicate),
        ("train", cmd_train),
        ("test", cmd_test),
    ):
        p = sub.add_parser(name)
        p.add_argument("--team-from", default=None, help="runs:r11-team (team.json)")
        p.add_argument("--cache-dir", default=None)
        p.add_argument("--out", default=None)
        p.add_argument(
            "--limit", type=int, default=None, help="incidents per system (smoke)"
        )
        if name == "test":
            p.add_argument(
                "--train-from", required=True, help="runs:r13-train (ranker.json)"
            )
        p.set_defaults(fn=fn)
    p = sub.add_parser("report")
    p.add_argument("--from", dest="sources", nargs="+", required=True)
    p.add_argument("--out", default="docs/results/r13_real_centre.md")
    p.set_defaults(fn=cmd_report)
    args = ap.parse_args()
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
