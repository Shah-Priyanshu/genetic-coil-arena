"""R12 report: the G9 and G10 numbers, verdicts and markdown from per-item records.

**G9** (a specialist joins the incident team, ``docs/goals.md``), computed from the per-incident
records of ``r12_team.run_incident`` on the test split:

1. RE3: the joined team's top-1 beats the old team's by >= 10 points, the paired 95% bootstrap
   CI of the difference excluding 0;
2. RE2: the joined team is non-inferior to the old team on top-1 and on joint accuracy, the
   lower bound of that CI > -5 points.

Kill: the RE2 top-1 or joint CI upper bound < 0, or the RE3 CI includes 0.

**G10** (the desk grows to a new system), computed from ``r11_pipeline.process_incident_item``
records on the R12 sealed set, split into the RE3 group (66 spec items) and the RE2 regression
group (25) by the incident the gold resolves to:

1. exact match >= 90% in each group (kill < 85% in either); routing >= 98% on the mixed set;
2. NL task score (RE3 root top-1, RE2 joint) non-inferior to the gold run in each group, the
   lower bound of the paired CI of NL minus gold > -5 points (kill: upper bound < 0);
3. round trip >= 90% of the executed items; 4. summary fidelity >= 90% of facts.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np

from . import r11_report as rep
from .r12_sealed import group_of

REF = "old"
BASELINES = ("baro", "new_ext", "re3_fusion")
BASELINE_LABELS = {
    "baro": "BARO on metrics",
    "new_ext": "newcomer alone (new_ext)",
    "re3_fusion": "RE3-only fusion (baro 0.5 + new_ext 0.5)",
}
RE3_GAIN = 0.10
NONINF = 0.05


# ---------------------------------------------------------------------- G9


def _col(records: Sequence[Mapping[str, Any]], getter: Any) -> np.ndarray:
    return np.array([float(getter(r)) for r in records])


def _stat(values: np.ndarray) -> Optional[float]:
    return float(values.mean()) if values.size else None


def g9_dataset(
    records: Sequence[Mapping[str, Any]], dataset: str, rule: str
) -> dict[str, Any]:
    """Old vs joined (and the baselines) on one dataset's test records."""
    rs = [r for r in records if r["dataset"] == dataset]
    out: dict[str, Any] = {"n": len(rs)}
    if not rs:
        return out
    old1 = _col(rs, lambda r: r["old"]["top1"])
    new1 = _col(rs, lambda r: r["rules"][rule]["top1"])
    out["old_top1"], out["joined_top1"] = _stat(old1), _stat(new1)
    out["old_top3"] = _stat(_col(rs, lambda r: r["old"]["in_top3"]))
    out["joined_top3"] = _stat(_col(rs, lambda r: r["rules"][rule]["in_top3"]))
    out["diff_top1"] = rep.paired_bootstrap(new1 - old1)
    if dataset == "re2":
        oldj = _col(rs, lambda r: r["old"]["joint"])
        newj = _col(rs, lambda r: r["rules"][rule]["joint"])
        out["old_joint"], out["joined_joint"] = _stat(oldj), _stat(newj)
        out["diff_joint"] = rep.paired_bootstrap(newj - oldj)
    out["baselines"] = {}
    for name in BASELINES:
        b = _col(rs, lambda r, n=name: r["baselines"][n]["top1"])
        out["baselines"][name] = {
            "top1": _stat(b),
            "top3": _stat(_col(rs, lambda r, n=name: r["baselines"][n]["in_top3"])),
            "joined_minus": rep.paired_bootstrap(new1 - b),
        }
    used = _col(rs, lambda r: r["rules"][rule]["used_logs"])
    changed = _col(rs, lambda r: r["rules"][rule]["changed_top1"])
    fixed = int(((new1 == 1) & (old1 == 0)).sum())
    broke = int(((new1 == 0) & (old1 == 1)).sum())
    out["logs_used"] = _stat(used)
    out["top1_changed"] = _stat(changed)
    out["fixed"], out["broke"] = fixed, broke
    by_sys: dict[str, Any] = {}
    for system in sorted({r["system"] for r in rs}):
        sub = [r for r in rs if r["system"] == system]
        o = _col(sub, lambda r: r["old"]["top1"])
        n = _col(sub, lambda r: r["rules"][rule]["top1"])
        by_sys[system] = {
            "n": len(sub),
            "old_top1": _stat(o),
            "joined_top1": _stat(n),
            "logs_used": _stat(_col(sub, lambda r: r["rules"][rule]["used_logs"])),
        }
    out["by_system"] = by_sys
    return out


def g9_verdict(re2: Mapping[str, Any], re3: Mapping[str, Any]) -> dict[str, Any]:
    """``PASS``, ``KILL`` or ``NOT PASSED`` with the four criteria."""
    rows: list[dict[str, Any]] = []

    def row(name: str, target: str, measured: str, ok: Optional[bool]) -> None:
        rows.append(
            {
                "criterion": name,
                "target": target,
                "measured": measured,
                "result": "n/a" if ok is None else ("PASS" if ok else "not met"),
            }
        )

    def pts(d: Mapping[str, Any]) -> str:
        if d["mean"] is None:
            return "n/a"
        return f"{d['mean'] * 100:+.1f} [{d['lo'] * 100:+.1f}, {d['hi'] * 100:+.1f}] points (n={d['n']})"

    d3 = re3.get("diff_top1")
    row(
        "RE3 top-1, joined - old",
        f">= +{RE3_GAIN * 100:.0f} points, CI excludes 0",
        "n/a" if not d3 or d3["mean"] is None else pts(d3),
        (
            None
            if not d3 or d3["mean"] is None
            else (d3["mean"] >= RE3_GAIN and d3["lo"] > 0)
        ),
    )
    kills = []
    for key, label in (("diff_top1", "RE2 top-1"), ("diff_joint", "RE2 joint")):
        d = re2.get(key)
        row(
            f"{label}, joined - old",
            f"lower CI bound > -{NONINF * 100:.0f} points",
            "n/a" if not d or d["mean"] is None else pts(d),
            None if not d or d["mean"] is None else d["lo"] > -NONINF,
        )
        if d and d["hi"] is not None and d["hi"] < 0:
            kills.append(f"{label} significantly below the old team")
    if d3 and d3["lo"] is not None and d3["lo"] <= 0:
        kills.append("the RE3 gain's CI includes 0")
    results = [r["result"] for r in rows]
    label = (
        "KILL"
        if kills
        else "PASS" if all(r == "PASS" for r in results) else "NOT PASSED"
    )
    return {"verdict": label, "kill_reasons": kills, "criteria": rows}


def g9_summary(test_records: Sequence[Mapping[str, Any]], rule: str) -> dict[str, Any]:
    re2 = g9_dataset(test_records, "re2", rule)
    re3 = g9_dataset(test_records, "re3", rule)
    return {"rule": rule, "re2": re2, "re3": re3, "verdict": g9_verdict(re2, re3)}


def _p(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}"


def render_g9(g9: Mapping[str, Any]) -> list[str]:
    """Markdown of the G9 section."""
    dev = g9["dev"]
    lines = [
        "## G9: the log-novelty specialist joins the incident team",
        "",
        f"Joined rule (pinned before the test run): `{g9['rule']}`. "
        f"State hash of the old team before and after joining and running: `{g9['hash_before']}` / `{g9['hash_after']}` "
        f"({'identical' if g9['hash_before'] == g9['hash_after'] else 'DIFFERENT'}).",
        "",
        "### Dev table (pooled RE2 SS/OB repetition 1 + RE3 repetition 1; chooses the rule)",
        "",
        "| Rule | Pooled top-1 | RE2 top-1 | RE3 top-1 | RE2 joint | Log part used |",
        "|---|---|---|---|---|---|",
    ]
    ref = dev["reference"]
    lines.append(
        f"| `metric` (old team) | {_p(ref['top1'])} | {_p(ref['re2_top1'])} | {_p(ref['re3_top1'])} | {_p(ref['re2_joint'])} | - |"
    )
    for t in dev["table"]:
        star = " (chosen)" if t["rule"] == dev["chosen"] else ""
        lines.append(
            f"| `{t['rule']}`{star} | {_p(t['top1'])} | {_p(t['re2_top1'])} | {_p(t['re3_top1'])} | {_p(t['re2_joint'])} | {_p(t['used_logs'])} |"
        )
    lines += [
        "",
        "The dev incidents of RE2 repetition 1 are in the fault specialist's training data, so the dev joint accuracy is optimistic; the ranking does not depend on it.",
        "",
    ]
    s = g9["test"]
    v = s["verdict"]
    lines += [
        f"### Verdict on the test split: **{v['verdict']}**"
        + (f" ({'; '.join(v['kill_reasons'])})" if v["kill_reasons"] else ""),
        "",
        "| Criterion | Target | Measured | Result |",
        "|---|---|---|---|",
    ]
    for r in v["criteria"]:
        lines.append(
            f"| {r['criterion']} | {r['target']} | {r['measured']} | {r['result']} |"
        )
    for ds, label in (
        ("re2", "RE2 SS/OB (repetitions 2-3)"),
        ("re3", "RE3 (repetitions 2-4)"),
    ):
        d = s[ds]
        if not d.get("n"):
            continue
        lines += [
            "",
            f"### {label}, n = {d['n']}",
            "",
            "| System | Top-1 | Top-3 | Joint | Joined - this (top-1, paired 95% CI) |",
            "|---|---|---|---|---|",
            f"| old team | {_p(d['old_top1'])} | {_p(d['old_top3'])} | {_p(d.get('old_joint'))} | {_pts(d['diff_top1'])} |",
            f"| joined team (`{g9['rule']}`) | {_p(d['joined_top1'])} | {_p(d['joined_top3'])} | {_p(d.get('joined_joint'))} | - |",
        ]
        for name, b in d["baselines"].items():
            lines.append(
                f"| {BASELINE_LABELS[name]} | {_p(b['top1'])} | {_p(b['top3'])} | - | {_pts(b['joined_minus'])} |"
            )
        lines += [
            "",
            f"The log part took part in {_p(d['logs_used'])}% of incidents and changed the top-1 in {_p(d['top1_changed'])}% ({d['fixed']} fixed, {d['broke']} broken).",
            "",
            "| System | n | Old top-1 | Joined top-1 | Log part used (%) |",
            "|---|---|---|---|---|",
        ]
        for system, x in d["by_system"].items():
            lines.append(
                f"| {system} | {x['n']} | {_p(x['old_top1'])} | {_p(x['joined_top1'])} | {_p(x['logs_used'])} |"
            )
    return lines + [""]


def _pts(d: Mapping[str, Any]) -> str:
    if d["mean"] is None:
        return "n/a"
    return f"joined {d['mean'] * 100:+.1f} [{d['lo'] * 100:+.1f}, {d['hi'] * 100:+.1f}]"


# ---------------------------------------------------------------------- G10


def split_groups(
    records: Sequence[Mapping[str, Any]], resolves_to: Mapping[str, Optional[str]]
) -> dict[str, list[Mapping[str, Any]]]:
    out: dict[str, list[Mapping[str, Any]]] = {"re3": [], "re2": [], "none": []}
    for r in records:
        out[group_of(resolves_to.get(r["id"]))].append(r)
    return out


def g10_metrics(
    records: Sequence[Mapping[str, Any]],
    route_records: Sequence[Mapping[str, Any]],
    resolves_to: Mapping[str, Optional[str]],
) -> dict[str, Any]:
    """All G10 numbers of one translator."""
    groups = split_groups(records, resolves_to)
    out: dict[str, Any] = {
        "all": rep.system_metrics(records, route_records),
        "group": {},
    }
    for name in ("re3", "re2"):
        out["group"][name] = rep.system_metrics(groups[name], [])
    by_system: dict[str, Any] = {}
    for r in groups["re3"] + groups["re2"]:
        if r["kind"] != "spec" or r.get("gold") is None:
            continue
        key = f"{r['gold'].get('system', '?')} ({group_of(resolves_to.get(r['id']))})"
        by_system.setdefault(key, []).append(bool(r.get("exact")))
    out["exact_by_system"] = {
        k: rep.rate(sum(v), len(v)) for k, v in sorted(by_system.items())
    }
    return out


def criteria_g10(
    m: Mapping[str, Any], vram_gb: Optional[float]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def row(name: str, target: str, measured: str, ok: Optional[bool]) -> None:
        rows.append(
            {
                "criterion": name,
                "target": target,
                "measured": measured,
                "result": "n/a" if ok is None else ("PASS" if ok else "not met"),
            }
        )

    def ci(r: Mapping[str, Any]) -> str:
        if r["p"] is None:
            return "n/a"
        return f"{r['p']:.1%} (n={r['n']}, 95% CI {r['lo']:.1%}-{r['hi']:.1%})"

    for g, label in (
        ("re3", "RE3 (new system and incidents)"),
        ("re2", "RE2 (regression)"),
    ):
        ex = m["group"][g]["exact"]
        row(
            f"exact match, {label}",
            f">= {rep.EXACT_PASS:.0%} (kill < {rep.EXACT_KILL:.0%})",
            ci(ex),
            None if ex["p"] is None else ex["p"] >= rep.EXACT_PASS,
        )
    rt = m["all"]["routing"]
    row("domain routing", f">= {rep.ROUTING_PASS:.0%}", ci(rt),
        None if rt["p"] is None else rt["p"] >= rep.ROUTING_PASS)  # fmt: skip
    for g, label, what in (
        ("re3", "RE3", "root top-1"),
        ("re2", "RE2", "joint"),
    ):
        task = m["group"][g]["task"]
        d = task["diff"]
        if d["mean"] is None:
            row(
                f"task score ({what}), {label}, NL vs gold",
                f"lower CI bound > -{rep.NONINF_MARGIN:.0%}",
                "n/a",
                None,
            )
        else:
            row(
                f"task score ({what}), {label}, NL vs gold",
                f"lower CI bound of NL - gold > -{rep.NONINF_MARGIN * 100:.0f} points",
                f"NL {task['nl']:.1%} vs gold {task['gold']:.1%}; diff {d['mean'] * 100:+.1f} [{d['lo'] * 100:+.1f}, {d['hi'] * 100:+.1f}] points (n={d['n']})",
                d["lo"] > -rep.NONINF_MARGIN,
            )
    rtp = m["all"]["round_trip"]
    row("round trip", f">= {rep.ROUND_TRIP_PASS:.0%}", ci(rtp),
        None if rtp["p"] is None else rtp["p"] >= rep.ROUND_TRIP_PASS)  # fmt: skip
    fid = m["all"]["fidelity"]
    fv = fid["facts"]
    row("summary field fidelity", f">= {rep.FIDELITY_PASS:.0%} of facts",
        "n/a" if fv is None else f"{fv:.1%} of {fid['n_facts']} facts; all facts in {fid['all_facts']:.1%} of {fid['n_summaries']} summaries",
        None if fv is None else fv >= rep.FIDELITY_PASS)  # fmt: skip
    lat = m["all"]["latency"]
    p95 = lat["total_p95"]
    rows.append(
        {
            "criterion": "warm end-to-end p95 / peak VRAM (reported)",
            "target": "-",
            "measured": ("n/a" if p95 is None else f"{p95:.2f} s")
            + ("" if vram_gb is None else f", {vram_gb:.1f} GB"),
            "result": "-",
        }
    )
    return rows


def g10_verdict(m: Mapping[str, Any], vram_gb: Optional[float]) -> dict[str, Any]:
    rows = criteria_g10(m, vram_gb)
    kills = []
    for g, label in (("re3", "RE3"), ("re2", "RE2")):
        ex = m["group"][g]["exact"]
        if ex["p"] is not None and ex["p"] < rep.EXACT_KILL:
            kills.append(f"{label} exact match {ex['p']:.1%} < {rep.EXACT_KILL:.0%}")
        d = m["group"][g]["task"]["diff"]
        if d["hi"] is not None and d["hi"] < 0:
            kills.append(f"{label} NL task score significantly below the gold run")
    results = [r["result"] for r in rows if r["result"] != "-"]
    label = (
        "KILL"
        if kills
        else "PASS" if all(r == "PASS" for r in results) else "NOT PASSED"
    )
    return {"verdict": label, "kill_reasons": kills, "criteria": rows}


def render_g10_system(s: Mapping[str, Any]) -> list[str]:
    m, v = s["metrics"], s["verdict"]
    lines = [
        f"### {s['label']}",
        "",
        f"**{v['verdict']}**"
        + (f" ({'; '.join(v['kill_reasons'])})" if v["kill_reasons"] else ""),
        "",
        "| Criterion | Target | Measured | Result |",
        "|---|---|---|---|",
    ]
    for r in v["criteria"]:
        lines.append(
            f"| {r['criterion']} | {r['target']} | {r['measured']} | {r['result']} |"
        )
    lines += [
        "",
        "Routing by desk: "
        + "; ".join(
            f"{k} {rep.pct(x)}" for k, x in m["all"]["routing_by_domain"].items()
        )
        + ".",
        "",
        "Exact match by system: "
        + "; ".join(f"{k} {rep.pct(x)}" for k, x in m["exact_by_system"].items())
        + ".",
        "",
        f"Final decision on all {m['all']['n_items']} items: {rep.pct(m['all']['decision'])}. "
        + " ".join(
            f"{k}: {rep.pct(x)} {x['status']}."
            for k, x in m["all"]["decision_by_kind"].items()
        ),
        "",
        "| Category | Exact match, all spec items | Decision (all kinds) |",
        "|---|---|---|",
    ]
    cats = sorted(
        set(m["all"]["exact_by_category"]) | set(m["all"]["decision_by_category"])
    )
    for c in cats:
        e = m["all"]["exact_by_category"].get(c)
        lines.append(
            f"| {c} | {rep.pct(e) if e else '-'} | {rep.pct(m['all']['decision_by_category'][c])} |"
        )
    bf = m["all"]["fidelity"]["by_fact"]
    if bf:
        lines += [
            "",
            "Fact accuracy: " + ", ".join(f"{k} {x:.1%}" for k, x in bf.items()) + ".",
        ]
    return lines + [""]


def _task_diff(m: Mapping[str, Any], group: str) -> str:
    d = m["group"][group]["task"]["diff"]
    if d["mean"] is None:
        return "n/a"
    return f"{d['mean'] * 100:+.1f} [{d['lo'] * 100:+.1f}, {d['hi'] * 100:+.1f}]"


def render(report: Mapping[str, Any], analysis: Optional[str] = None) -> str:
    """The markdown of ``docs/results/r12_growth.md`` (``analysis`` is carried over)."""
    lines = [
        "# R12: growth on real incidents (goals G9 and G10)",
        "",
        f"Generated {report['generated']}. Command: `python scripts/r12_growth.py report`.",
        "",
    ]
    if report.get("g9"):
        lines += render_g9(report["g9"])
    g10 = report.get("g10")
    if g10:
        lines += [
            "## G10: the desk grows to Train Ticket and the RE3 incidents",
            "",
            f"- **Sealed set:** {g10['n_sealed']} hand-written requests (`experiments/data/r12_sealed_test.jsonl`, SHA-256 `{g10['sealed_sha256']}`), plus the R7a sealed spec and clarify items for the routing check (SHA-256 `{g10['r7a_sealed_sha256']}`). Each translator ran once.",
            f"- **Catalogue:** {g10['catalogue']['n']} test incidents (RE2 SS/OB repetitions 2-3 as `INC-1001`..`INC-1120`, RE3 repetitions 2-4 as `INC-2001`..`INC-2062`), fingerprint `{g10['catalogue']['fingerprint']}`; desk clock Wednesday 2024-12-11 09:00 UTC.",
            f"- **Prompt audit:** {g10['prompt_audit']}",
            f"- **Team:** {g10['team']}",
            "",
            "### Verdict",
            "",
            "| Translator | G10 verdict | Exact RE3 | Exact RE2 | Routing | NL - gold, RE3 | NL - gold, RE2 | Round trip | Fidelity |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for s in g10["systems"].values():
            m = s["metrics"]

            fid = m["all"]["fidelity"]["facts"]
            lines.append(
                f"| {s['label']} | **{s['verdict']['verdict']}** | {rep.pct(m['group']['re3']['exact'])} | {rep.pct(m['group']['re2']['exact'])} | {rep.pct(m['all']['routing'])} | {_task_diff(m, 're3')} | {_task_diff(m, 're2')} | {rep.pct(m['all']['round_trip'])} | {'n/a' if fid is None else f'{fid:.1%}'} |"
            )
        lines += ["", "### Per translator", ""]
        for s in g10["systems"].values():
            lines += render_g10_system(s)
            lines += rep.render_failures(s["label"], s["failures"])
    lines += [
        "## Analysis",
        "",
        "<!-- analysis:start -->",
        analysis or "*Analysis pending the full run.*",
        "<!-- analysis:end -->",
        "",
    ]
    return "\n".join(lines)
