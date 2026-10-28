"""R12b report (goal G9b): numbers, verdict and markdown from the per-incident records.

Groups (``r12b_team``): ``tt`` (RE2 Train Ticket, fresh: root top-1 only), ``re3_test`` and
``re2_test`` (both reused: G9 already ranked them). With *joined* = the chosen v2 rule and
*old* = the frozen G4c team, paired 95% bootstrap CIs (10,000 resamples) over incidents:

1. TT: joined - old top-1, lower bound > -5 points (non-inferior);
2. RE3 test: joined - old top-1 >= +10 points and the CI excludes 0;
3. RE2 SS/OB test: top-1 and joint, lower bound > -5 points.

Kill: TT or RE2 SS/OB top-1 CI wholly below 0 (upper bound < 0), or the RE3 gain's CI includes 0.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np

from ..envs.logtriage import novelty2 as n2
from . import r11_report as rep

RE3_GAIN = 0.10
NONINF = 0.05
BASELINES = ("baro", "new_ext", "re3_fusion")
BASELINE_LABELS = {
    "baro": "BARO on metrics",
    "new_ext": "G9 newcomer alone (new_ext)",
    "re3_fusion": "RE3-only fusion (baro 0.5 + new_ext 0.5)",
}
GROUP_LABELS = {
    "tt": "RE2 Train Ticket (fresh, never ranked in a test)",
    "re3_test": "RE3 test, repetitions 2-4 (REUSED: G4d, G4e and G9 used it)",
    "re2_test": "RE2 SS/OB test, repetitions 2-3 (REUSED: G9 used it)",
}


def _col(records: Sequence[Mapping[str, Any]], getter: Any) -> np.ndarray:
    return np.array([float(getter(r)) for r in records])


def _stat(values: np.ndarray) -> Optional[float]:
    return float(values.mean()) if values.size else None


def _p(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}"


def _pts(d: Optional[Mapping[str, Any]]) -> str:
    if not d or d["mean"] is None:
        return "n/a"
    return f"{d['mean'] * 100:+.1f} [{d['lo'] * 100:+.1f}, {d['hi'] * 100:+.1f}]"


def group_summary(
    records: Sequence[Mapping[str, Any]], group: str, rule: str
) -> dict[str, Any]:
    """Old vs joined (chosen rule), the G9 reference rule, newcomer v2 alone, baselines."""
    rs = [r for r in records if r["group"] == group]
    out: dict[str, Any] = {"group": group, "rule": rule, "n": len(rs)}
    if not rs:
        return out
    joint = group != "tt" and group != "re3_test"  # TT: root top-1 only; RE3: no joint
    old1 = _col(rs, lambda r: r["old"]["top1"])
    out["old_top1"] = _stat(old1)
    out["old_top3"] = _stat(_col(rs, lambda r: r["old"]["in_top3"]))
    rows: dict[str, Any] = {}
    for name in (rule, n2.REF_RULE, n2.ALONE_RULE):
        n1 = _col(rs, lambda r, n=name: r["rules"][n]["top1"])
        row: dict[str, Any] = {
            "top1": _stat(n1),
            "top3": _stat(_col(rs, lambda r, n=name: r["rules"][n]["in_top3"])),
            "diff_top1": rep.paired_bootstrap(n1 - old1),
            "fixed": int(((n1 == 1) & (old1 == 0)).sum()),
            "broke": int(((n1 == 0) & (old1 == 1)).sum()),
            "logs_used": _stat(_col(rs, lambda r, n=name: r["rules"][n]["used_logs"])),
            "top1_changed": _stat(
                _col(rs, lambda r, n=name: r["rules"][n]["changed_top1"])
            ),
        }
        if joint:
            oj = _col(rs, lambda r: r["old"]["joint"])
            nj = _col(rs, lambda r, n=name: r["rules"][n]["joint"])
            row["joint"] = _stat(nj)
            row["diff_joint"] = rep.paired_bootstrap(nj - oj)
        rows[name] = row
    out["rules"] = rows
    if joint:
        out["old_joint"] = _stat(_col(rs, lambda r: r["old"]["joint"]))
    out["baselines"] = {}
    j1 = _col(rs, lambda r: r["rules"][rule]["top1"])
    for name in BASELINES:
        b = _col(rs, lambda r, n=name: r["baselines"][n]["top1"])
        out["baselines"][name] = {
            "top1": _stat(b),
            "top3": _stat(_col(rs, lambda r, n=name: r["baselines"][n]["in_top3"])),
            "joined_minus": rep.paired_bootstrap(j1 - b),
        }
    moved = _col(rs, lambda r: r["attr"]["services_moved"] > 0)
    out["attribution"] = {
        "incidents_moved": _stat(moved),
        "templates_redirected": int(sum(r["attr"]["templates_redirected"] for r in rs)),
        "templates": int(sum(r["attr"]["templates"] for r in rs)),
    }
    out["by_system"] = _breakdown(rs, "system", rule)
    if group != "re3_test":
        out["by_fault"] = _breakdown(rs, "fault", rule)
    return out


def _breakdown(rs: Sequence[Mapping[str, Any]], key: str, rule: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for value in sorted({str(r[key]) for r in rs}):
        sub = [r for r in rs if str(r[key]) == value]
        o = _col(sub, lambda r: r["old"]["top1"])
        j = _col(sub, lambda r: r["rules"][rule]["top1"])
        g = _col(sub, lambda r: r["rules"][n2.REF_RULE]["top1"])
        out[value] = {
            "n": len(sub),
            "old_top1": _stat(o),
            "joined_top1": _stat(j),
            "g9_top1": _stat(g),
            "fixed": int(((j == 1) & (o == 0)).sum()),
            "broke": int(((j == 0) & (o == 1)).sum()),
        }
    return out


def verdict(groups: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """``PASS``, ``KILL`` or ``NOT PASSED`` with the criteria rows."""
    rows: list[dict[str, Any]] = []
    kills: list[str] = []

    def row(
        name: str, target: str, d: Optional[Mapping[str, Any]], ok: Optional[bool]
    ) -> None:
        rows.append(
            {
                "criterion": name,
                "target": target,
                "measured": _pts(d) + (f" points (n={d['n']})" if d and d["n"] else ""),
                "result": "n/a" if ok is None else ("PASS" if ok else "not met"),
            }
        )

    def diff(group: str, key: str) -> Optional[Mapping[str, Any]]:
        g = groups.get(group)
        if not g or not g.get("n"):
            return None
        d = g["rules"][g["rule"]].get(key)
        return d if d and d["mean"] is not None else None

    noninf = f"lower CI bound > -{NONINF * 100:.0f} points"
    d = diff("tt", "diff_top1")
    row(
        "(1) RE2 Train Ticket (fresh) top-1, joined - old",
        noninf,
        d,
        None if d is None else d["lo"] > -NONINF,
    )
    if d and d["hi"] < 0:
        kills.append("RE2 Train Ticket top-1 significantly below the old team")
    d = diff("re3_test", "diff_top1")
    row(
        "(2) RE3 test (reused) top-1, joined - old",
        f">= +{RE3_GAIN * 100:.0f} points, CI excludes 0",
        d,
        None if d is None else (d["mean"] >= RE3_GAIN and d["lo"] > 0),
    )
    if d and d["lo"] <= 0:
        kills.append("the RE3 gain's CI includes 0")
    for key, label in (("diff_top1", "top-1"), ("diff_joint", "joint")):
        d = diff("re2_test", key)
        row(
            f"(3) RE2 SS/OB test (reused) {label}, joined - old",
            noninf,
            d,
            None if d is None else d["lo"] > -NONINF,
        )
        if key == "diff_top1" and d and d["hi"] < 0:
            kills.append("RE2 SS/OB top-1 significantly below the old team")
    results = [r["result"] for r in rows]
    label = (
        "KILL"
        if kills
        else "PASS" if all(r == "PASS" for r in results) else "NOT PASSED"
    )
    return {"verdict": label, "kill_reasons": kills, "criteria": rows}


def summarise(records: Sequence[Mapping[str, Any]], rule: str) -> dict[str, Any]:
    groups = {
        g: group_summary(records, g, rule) for g in ("tt", "re3_test", "re2_test")
    }
    return {"rule": rule, "groups": groups, "verdict": verdict(groups)}


def render(report: Mapping[str, Any], analysis: Optional[str] = None) -> str:
    dev = report["dev"]
    lines = [
        "# R12b: a newcomer that reports what its errors are about (goal G9b)",
        "",
        f"Generated {report['generated']}. Joined rule (pinned before any fresh or test run): "
        f"`{report['rule']}`. State hash of the old team before and after joining and running: "
        f"`{report['hash_before']}` / `{report['hash_after']}` "
        f"({'identical' if report['hash_before'] == report['hash_after'] else 'DIFFERENT'}).",
        "",
        "The RE3 and RE2 SS/OB test splits were already used by G9 (RE3 also by G4d and G4e), and "
        "the attribution idea came from G9's test-split failure analysis, so those two rows are "
        "re-scored on reused data and are weaker evidence than the Train Ticket row.",
        "",
        "## Dev table (pooled RE2 SS/OB repetition 1 + RE3 repetition 1; chooses the rule)",
        "",
        "| Rule | Pooled top-1 | RE2 top-1 | RE3 top-1 | RE2 joint | Log part used |",
        "|---|---|---|---|---|---|",
    ]
    for t in dev["reference"]:
        tag = {
            "metric": "`metric` (old team)",
            n2.REF_RULE: f"`{n2.REF_RULE}` (G9 rule, reference, not selectable)",
            n2.ALONE_RULE: f"`{n2.ALONE_RULE}` (newcomer v2 alone, not selectable)",
        }[t["rule"]]
        lines.append(
            f"| {tag} | {_p(t['top1'])} | {_p(t['re2_top1'])} | {_p(t['re3_top1'])} | {_p(t['re2_joint'])} | {_p(t['used_logs'])} |"
        )
    for t in dev["table"]:
        star = " (chosen)" if t["rule"] == dev["chosen"] else ""
        lines.append(
            f"| `{t['rule']}`{star} | {_p(t['top1'])} | {_p(t['re2_top1'])} | {_p(t['re3_top1'])} | {_p(t['re2_joint'])} | {_p(t['used_logs'])} |"
        )
    v = report["summary"]["verdict"]
    lines += [
        "",
        f"## Verdict: **{v['verdict']}**"
        + (f" ({'; '.join(v['kill_reasons'])})" if v["kill_reasons"] else ""),
        "",
        "| Criterion | Target | Measured (points, paired 95% CI) | Result |",
        "|---|---|---|---|",
    ]
    for r in v["criteria"]:
        lines.append(
            f"| {r['criterion']} | {r['target']} | {r['measured']} | {r['result']} |"
        )
    for group, g in report["summary"]["groups"].items():
        if not g.get("n"):
            continue
        rule = g["rule"]
        lines += [
            "",
            f"## {GROUP_LABELS[group]}, n = {g['n']}",
            "",
            "| System | Top-1 | Top-3 | Joint | This - old (top-1, paired 95% CI) | Fixed / broken |",
            "|---|---|---|---|---|---|",
            f"| old team | {_p(g['old_top1'])} | {_p(g['old_top3'])} | {_p(g.get('old_joint'))} | - | - |",
        ]
        labels = {
            rule: f"joined v2 (`{rule}`)",
            n2.REF_RULE: f"G9 joined team (`{n2.REF_RULE}`, v1 counts)",
            n2.ALONE_RULE: "newcomer v2 alone (`n_attr`)",
        }
        for name, x in g["rules"].items():
            lines.append(
                f"| {labels[name]} | {_p(x['top1'])} | {_p(x['top3'])} | {_p(x.get('joint'))} | {_pts(x['diff_top1'])} | {x['fixed']} / {x['broke']} |"
            )
        for name, b in g["baselines"].items():
            lines.append(
                f"| {BASELINE_LABELS[name]} | {_p(b['top1'])} | {_p(b['top3'])} | - | joined - this: {_pts(b['joined_minus'])} | - |"
            )
        x = g["rules"][rule]
        a = g["attribution"]
        lines += [
            "",
            f"The log part took part in {_p(x['logs_used'])}% of incidents and changed the top-1 in {_p(x['top1_changed'])}%. "
            f"Attribution moved a count in {_p(a['incidents_moved'])}% of incidents ({a['templates_redirected']} of {a['templates']} new templates were credited to another service).",
        ]
        for key, title in (("by_system", "System"), ("by_fault", "Fault")):
            if key not in g:
                continue
            lines += [
                "",
                f"| {title} | n | Old top-1 | Joined v2 top-1 | G9 rule top-1 | Fixed / broken |",
                "|---|---|---|---|---|---|",
            ]
            for name, y in g[key].items():
                lines.append(
                    f"| {name} | {y['n']} | {_p(y['old_top1'])} | {_p(y['joined_top1'])} | {_p(y['g9_top1'])} | {y['fixed']} / {y['broke']} |"
                )
    lines += [
        "",
        "## Analysis",
        "",
        "<!-- analysis:start -->",
        analysis or "(to be written after the job result is read)",
        "<!-- analysis:end -->",
        "",
    ]
    return "\n".join(lines)
