"""R10 step 2 report: joint accuracy, paired comparisons, ablations and the G4b verdict.

Reads the per-incident records written by ``scripts/r10_step2.py``. The dev split
(repetition 1) picks the rule centre's fault policy per log-specialist model (highest
joint accuracy); every reported number is on the test split (repetitions 2 and 3).

**G4b rule** (``docs/goals.md``): a hybrid team (metric + log specialists) passes if its
joint accuracy (root cause and fault type both right) beats the best single system by at
least 5 points with the 95% CI of the paired difference excluding 0 (Holm-adjusted
Wilcoxon p < 0.05 across the hybrid teams), while its root-cause top-1 is non-inferior to
the best single system (CI lower bound of the difference above -3 points, checked against
the best single by joint accuracy and the best by top-1). Killed if the best single system
is within the CI on joint accuracy for every hybrid team. "Best single system" is chosen
on the same incidents among BARO, and each model's ``single_selected`` and
``oracle_selected``, which favours the singles.
"""

from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Optional

import numpy as np

from ..envs.logtriage import team as tm
from ..evaluation.stats import (
    holm_correction,
    paired_bootstrap_ci,
    wilcoxon_signed_rank,
)
from . import log_triage_report as rp

MARGIN = 0.05
NON_INFERIORITY = -0.03
SINGLES = ("baro_metrics", "single_selected", "oracle_selected")
HYBRIDS = ("team_hybrid_rule", "team_hybrid_llm")
TEAMS = ("team_metric_rule", "team_metric_llm", *HYBRIDS)
LABELS = {
    "baro_metrics": "BARO on all metrics + metric-kind fault (non-LLM)",
    "single_selected": "single model, strong selection",
    "oracle_selected": "single model, long context, same selection",
    "team_metric_rule": "team: metric specialists, rule centre (distributed BARO)",
    "team_metric_llm": "team: metric specialists, LLM centre",
    "team_hybrid_rule": "team: metric + log specialists, rule centre",
    "team_hybrid_llm": "team: metric + log specialists, LLM centre",
}


def short(model: str) -> str:
    return model.replace("ollama:", "")


def is_test(row: dict[str, Any]) -> bool:
    return int(row.get("rep", 0)) != 1


def fault_ok(row: dict[str, Any], policy: Optional[str]) -> int:
    """Fault-type correctness of a row, under ``policy`` for rule-centre rows."""
    fp = (row.get("extra") or {}).get("fault_policies")
    if policy and fp and policy in fp:
        return int(fp[policy] == row["fault"])
    return int(row["fault_ok"])


def joint(row: dict[str, Any], policy: Optional[str]) -> int:
    return int(row["top1"]) * fault_ok(row, policy)


def pick_policies(
    rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Per log-specialist model, the fault policy with the best dev joint accuracy."""
    dev: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["condition"] != "team_hybrid_rule" or is_test(r):
            continue
        for p in tm.FAULT_POLICIES:
            dev[r["model"]][p].append(joint(r, p))
    chosen, table = {}, {}
    for model, by_p in dev.items():
        means = {p: float(np.mean(v)) for p, v in by_p.items()}
        # ties go to the earlier policy in FAULT_POLICIES (the metric-only default first)
        chosen[model] = max(
            tm.FAULT_POLICIES,
            key=lambda p: (means.get(p, -1), -tm.FAULT_POLICIES.index(p)),
        )
        table[model] = {
            "n_dev": len(by_p["metric"]),
            "joint": means,
            "chosen": chosen[model],
        }
    return chosen, table


class Systems:
    """Per-incident vectors of every system on the test split."""

    def __init__(self, rows: Sequence[dict[str, Any]], policies: dict[str, str]):
        self.rows = [r for r in rows if is_test(r)]
        self.policies = policies
        self.by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for r in self.rows:
            self.by_key[(r["model"], r["condition"])].append(r)

    def policy_of(self, model: str, condition: str) -> Optional[str]:
        return self.policies.get(model) if condition.endswith("_rule") else None

    def vec(self, model: str, condition: str, what: str) -> dict[str, float]:
        out: dict[str, float] = {}
        pol = self.policy_of(model, condition)
        for r in self.by_key.get((model, condition), []):
            if what == "joint":
                out[r["incident"]] = float(joint(r, pol))
            elif what == "fault":
                out[r["incident"]] = float(fault_ok(r, pol))
            else:
                out[r["incident"]] = float(r[what])
        return out

    def keys(self) -> list[tuple[str, str]]:
        """(model, condition) of the main systems, singles first."""
        order = {c: i for i, c in enumerate((*SINGLES, *TEAMS))}
        found = [k for k in self.by_key if k[1] in order]
        return sorted(found, key=lambda k: (order[k[1]], k[0]))

    def name(self, key: tuple[str, str]) -> str:
        model, cond = key
        if model == "none":
            return cond
        return f"{cond}[{short(model)}]"


def _rows_for_table(sys_: Systems) -> list[dict[str, Any]]:
    table = []
    for key in sys_.keys():
        model, cond = key
        sel = sys_.by_key[key]
        pol = sys_.policy_of(model, cond)
        table.append(
            {
                "system": sys_.name(key),
                "label": LABELS[cond],
                "model": model,
                "condition": cond,
                "policy": pol,
                "n": len(sel),
                "top1": [r["top1"] for r in sel],
                "top3": [r["top3"] for r in sel],
                "fault": [fault_ok(r, pol) for r in sel],
                "joint": [joint(r, pol) for r in sel],
                "tokens": float(np.mean([r["tokens"] for r in sel])),
                "centre_tokens": float(np.mean([r["centre_tokens"] for r in sel])),
                "bytes": float(
                    np.mean([(r.get("extra") or {}).get("bytes", 0) for r in sel])
                ),
                "latency": float(np.mean([r["latency_s"] for r in sel])),
            }
        )
    return table


def paired(
    a: dict[str, float], b: dict[str, float], seed: int = 0
) -> Optional[dict[str, Any]]:
    ids = sorted(set(a) & set(b))
    if len(ids) < 2:
        return None
    x, y = np.array([a[i] for i in ids]), np.array([b[i] for i in ids])
    diff, lo, hi = paired_bootstrap_ci(x, y, n_boot=10000, seed=seed)
    try:
        _, p = wilcoxon_signed_rank(x, y)
    except ValueError:
        p = 1.0
    return {
        "n": len(ids),
        "diff": float(diff),
        "low": float(lo),
        "high": float(hi),
        "p": float(p),
    }


def best_single(
    sys_: Systems, team_key: tuple[str, str], what: str
) -> Optional[tuple[str, tuple[str, str]]]:
    """The single system with the highest mean ``what`` on the team's incidents."""
    team_ids = set(sys_.vec(*team_key, "top1"))
    best: Optional[tuple[float, tuple[str, str]]] = None
    for key in sys_.keys():
        if key[1] not in SINGLES:
            continue
        v = sys_.vec(*key, what)
        ids = team_ids & set(v)
        if not ids:
            continue
        m = float(np.mean([v[i] for i in ids]))
        if best is None or m > best[0]:
            best = (m, key)
    return (sys_.name(best[1]), best[1]) if best else None


def g4b_verdicts(sys_: Systems) -> dict[str, Any]:
    """The G4b rule for every hybrid team, and the overall verdict."""
    out: dict[str, Any] = {}
    teams = [k for k in sys_.keys() if k[1] in HYBRIDS]
    for key in teams:
        bj = best_single(sys_, key, "joint")
        bt = best_single(sys_, key, "top1")
        if bj is None or bt is None:
            continue
        j = paired(sys_.vec(*key, "joint"), sys_.vec(*bj[1], "joint"), seed=2)
        t_j = paired(sys_.vec(*key, "top1"), sys_.vec(*bj[1], "top1"), seed=3)
        t_t = paired(sys_.vec(*key, "top1"), sys_.vec(*bt[1], "top1"), seed=4)
        if j is None or t_j is None or t_t is None:
            continue
        out[sys_.name(key)] = {
            "best_single_joint": bj[0],
            "best_single_top1": bt[0],
            "joint": j,
            "top1_vs_best_joint_single": t_j,
            "top1_vs_best_top1_single": t_t,
            "team_joint": float(np.mean(list(sys_.vec(*key, "joint").values()))),
            "best_single_joint_value": float(
                np.mean([sys_.vec(*bj[1], "joint")[i] for i in sys_.vec(*key, "joint")])
            ),
        }
    if out:
        adj = holm_correction([v["joint"]["p"] for v in out.values()])
        for v, a in zip(out.values(), adj):
            v["p_holm"] = float(a)
            non_inf = (
                v["top1_vs_best_joint_single"]["low"] > NON_INFERIORITY
                and v["top1_vs_best_top1_single"]["low"] > NON_INFERIORITY
            )
            j = v["joint"]
            if j["low"] > 0 and j["diff"] >= MARGIN and v["p_holm"] < 0.05 and non_inf:
                v["verdict"] = "PASS"
            elif j["low"] > 0 and j["diff"] > 0:
                v["verdict"] = "INCONCLUSIVE"
            else:
                v["verdict"] = "KILL"
            v["non_inferior_top1"] = bool(non_inf)
    kinds = [v["verdict"] for v in out.values()]
    overall = (
        "NOT RUN"
        if not kinds
        else (
            "PASS"
            if "PASS" in kinds
            else "KILL" if all(k == "KILL" for k in kinds) else "INCONCLUSIVE"
        )
    )
    return {"overall": overall, "teams": out}


def hybrid_vs_metric_only(sys_: Systems) -> list[dict[str, Any]]:
    """Dropping the log specialists: each hybrid team against the same centre's metric-only team."""
    rows = []
    for model, cond in sys_.keys():
        if cond not in HYBRIDS:
            continue
        base = ("none", "team_metric_rule") if cond.endswith("_rule") else None
        if base is None:
            cands = [k for k in sys_.by_key if k[1] == "team_metric_llm"]
            if not cands:
                continue
            base = cands[0]
        for what in ("joint", "fault", "top1"):
            r = paired(sys_.vec(model, cond, what), sys_.vec(*base, what), seed=5)
            if r:
                rows.append(
                    {
                        "team": sys_.name((model, cond)),
                        "without_logs": sys_.name(base),
                        "metric": what,
                        **r,
                    }
                )
    return rows


def _score(row: dict[str, Any], what: str, policy: Optional[str]) -> int:
    return joint(row, policy) if what == "joint" else int(row["top1"])


def ablation_rows(sys_: Systems) -> list[dict[str, Any]]:
    """Message ablations: intact vs ablated top-1 and joint accuracy."""
    out = []
    for key in sys_.keys():
        model, cond = key
        if cond not in TEAMS:
            continue
        pol = sys_.policy_of(model, cond)
        intact = {r["incident"]: r for r in sys_.by_key[key]}
        modes = sorted(
            {
                c.split(":", 1)[1]
                for (m, c) in sys_.by_key
                if m == model and c.startswith(cond + ":")
            }
        )
        for mode in modes:
            abl = {r["incident"]: r for r in sys_.by_key[(model, f"{cond}:{mode}")]}
            ids = sorted(set(intact) & set(abl))
            if len(ids) < 2:
                continue
            row = {"team": sys_.name(key), "ablation": mode, "n": len(ids)}
            for what in ("top1", "joint"):
                p_ = pol if what == "joint" else None
                a = np.array([_score(intact[i], what, p_) for i in ids], dtype=float)
                b = np.array([_score(abl[i], what, p_) for i in ids], dtype=float)
                d, lo, hi = paired_bootstrap_ci(b, a, n_boot=5000, seed=6)
                row[what] = {
                    "intact": float(a.mean()),
                    "ablated": float(b.mean()),
                    "change": float(d),
                    "low": float(lo),
                    "high": float(hi),
                }
            out.append(row)
    return out


def comparison_pairs(sys_: Systems) -> list[dict[str, Any]]:
    """Joint-accuracy differences between named systems, Holm over the table."""
    names = {sys_.name(k): k for k in sys_.keys()}
    pairs: list[tuple[str, str]] = []
    singles = [n for n, k in names.items() if k[1] in SINGLES]
    for n, k in names.items():
        if k[1] in HYBRIDS:
            pairs += [(n, s) for s in singles]
            if "team_metric_rule" in names:
                pairs.append((n, "team_metric_rule"))
    if "team_metric_rule" in names and "baro_metrics" in names:
        pairs.append(("team_metric_rule", "baro_metrics"))
    rows = []
    for a, b in pairs:
        r = paired(sys_.vec(*names[a], "joint"), sys_.vec(*names[b], "joint"), seed=7)
        if r:
            rows.append({"a": a, "b": b, **r})
    if rows:
        for row, adj in zip(rows, holm_correction([r["p"] for r in rows])):
            row["p_holm"] = float(adj)
    return rows


def breakdown(sys_: Systems, names: Sequence[str]) -> list[dict[str, Any]]:
    keymap = {sys_.name(k): k for k in sys_.keys()}
    out = []
    groups: dict[str, list[str]] = defaultdict(list)
    for r in sys_.rows:
        groups[f"fault {r['fault']}"].append(r["incident"])
        groups[f"system {r['system']}"].append(r["incident"])
    for g, ids in sorted(groups.items()):
        ids_set = set(ids)
        row: dict[str, Any] = {"group": g, "n": len(ids_set)}
        for n in names:
            if n in keymap:
                v = sys_.vec(*keymap[n], "joint")
                sel = [x for i, x in v.items() if i in ids_set]
                row[n] = float(np.mean(sel)) if sel else None
        out.append(row)
    return out


def distributed_baro_check(sys_: Systems) -> Optional[dict[str, Any]]:
    a = {r["incident"]: r for r in sys_.by_key.get(("none", "baro_metrics"), [])}
    b = {r["incident"]: r for r in sys_.by_key.get(("none", "team_metric_rule"), [])}
    ids = sorted(set(a) & set(b))
    if not ids:
        return None
    same_top = np.mean([a[i]["ranking"][0] == b[i]["ranking"][0] for i in ids])
    same_fault = np.mean([a[i]["pred_fault"] == b[i]["pred_fault"] for i in ids])
    return {
        "n": len(ids),
        "same_top1_service": float(same_top),
        "same_fault": float(same_fault),
    }


def _p(x: float) -> str:
    return f"{100 * x:+.1f}"


def _fmt_p(p: float) -> str:
    return f"{p:.3g}"


def build_report(
    records: Sequence[dict[str, Any]],
    analysis: Optional[str] = None,
    note: Optional[str] = None,
) -> tuple[str, dict[str, Any]]:
    rows = rp.dedupe(records)
    policies, dev_table = pick_policies(rows)
    sys_ = Systems(rows, policies)
    table = _rows_for_table(sys_)
    verdicts = g4b_verdicts(sys_)
    cmp_rows = comparison_pairs(sys_)
    abl = ablation_rows(sys_)
    drop = hybrid_vs_metric_only(sys_)
    names = [t["system"] for t in table]
    split = breakdown(sys_, names)
    dist = distributed_baro_check(sys_)
    n_test = len({r["incident"] for r in sys_.rows})

    md: list[str] = [
        "# R10 step 2: a heterogeneous team with typed messages (goal G4b)",
        "",
    ]
    if note:
        md += [f"*{note}*", ""]
    md += [
        "Step 2 of R10 on RCAEval RE2 (Sock Shop + Online Boutique). Metric specialists (code, one per service, "
        "own metrics only) send typed anomaly reports; log specialists (a model, one per service, own logs around "
        "the reported onset) send typed fault hypotheses; a rule or LLM centre decides. Numbers are on the test "
        f"split ({n_test} incidents); the dev split only chose the rule centre's fault policy.",
        "",
        "## Verdict",
        "",
        f"**G4b: {verdicts['overall']}.** Pass: a hybrid team beats the best single system on joint accuracy by "
        "at least 5 points (95% CI excludes 0, Holm-adjusted p < 0.05) with root-cause top-1 non-inferior "
        "(CI lower bound above -3 points). Kill: the best single system is within the CI on joint accuracy.",
        "",
    ]
    if verdicts["teams"]:
        md += [
            "| Hybrid team | Best single (joint) | Team joint | Single joint | Difference [95% CI] | p (Holm) | Top-1 vs it | Top-1 vs best-top-1 single | Verdict |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for name, v in verdicts["teams"].items():
            j, t1, t2_ = (
                v["joint"],
                v["top1_vs_best_joint_single"],
                v["top1_vs_best_top1_single"],
            )
            md.append(
                f"| {name} | {v['best_single_joint']} | {100 * v['team_joint']:.1f} | {100 * v['best_single_joint_value']:.1f} | "
                f"{_p(j['diff'])} [{_p(j['low'])}, {_p(j['high'])}] | {_fmt_p(v['p_holm'])} | "
                f"{_p(t1['diff'])} [{_p(t1['low'])}, {_p(t1['high'])}] | "
                f"{v['best_single_top1']}: {_p(t2_['diff'])} [{_p(t2_['low'])}, {_p(t2_['high'])}] | **{v['verdict']}** |"
            )
        md.append("")
    md += [
        "Honest reading: BARO is decomposable per service, so a metric-only team with a rule centre is distributed "
        "BARO and cannot beat BARO (check below); the real test is the hybrid's fault-type gain from the log "
        "specialists, at unchanged root-cause accuracy."
        + (
            f" Check: the metric-only rule team names the same top service as BARO on {100 * dist['same_top1_service']:.1f}% "
            f"and the same fault on {100 * dist['same_fault']:.1f}% of {dist['n']} incidents."
            if dist
            else ""
        ),
        "",
        "## Analysis",
        "",
        "<!-- analysis:start -->",
        analysis or "*Analysis pending the full run.*",
        "<!-- analysis:end -->",
        "",
        "## Results (test split)",
        "",
        "Percent per incident with 95% bootstrap CI. Joint = root-cause service and fault type both right. Chance "
        "for top-1 is about 7.5, for fault type 16.7, for joint about 1.3. Tokens are model tokens read and written "
        "per incident over all calls (specialists and centre); bytes are the typed messages' compact JSON; "
        "the rule centre's fault policy per model was chosen on dev.",
        "",
        "| System | n | Top-1 | Top-3 | Fault type | Joint | Tokens / incident | Centre prompt | Bytes / incident | Latency (s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for t in table:
        md.append(
            f"| {t['system']} | {t['n']} | {rp._cell(t['top1'])} | {rp._cell(t['top3'])} | {rp._cell(t['fault'])} | "
            f"{rp._cell(t['joint'])} | {t['tokens']:,.0f} | {t['centre_tokens']:,.0f} | {t['bytes']:,.0f} | {t['latency']:.1f} |"
        )
    md += ["", "### Fault policy of the rule centre (chosen on dev)", ""]
    if dev_table:
        md += [
            "| Log-specialist model | Dev incidents | "
            + " | ".join(tm.FAULT_POLICIES)
            + " | Chosen |",
            "|---|---|" + "---|" * (len(tm.FAULT_POLICIES) + 1),
        ]
        for model, d in dev_table.items():
            md.append(
                f"| {short(model)} | {d['n_dev']} | "
                + " | ".join(
                    f"{100 * d['joint'].get(p, float('nan')):.1f}"
                    for p in tm.FAULT_POLICIES
                )
                + f" | {d['chosen']} |"
            )
    else:
        md.append("No dev records.")
    md += [
        "",
        "### Paired comparisons of joint accuracy",
        "",
        "Difference a - b in points on shared test incidents, paired bootstrap 95% CI, Wilcoxon p, Holm over this table.",
        "",
        "| a | b | n | Difference | 95% CI | p | p (Holm) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in cmp_rows:
        md.append(
            f"| {r['a']} | {r['b']} | {r['n']} | {_p(r['diff'])} | [{_p(r['low'])}, {_p(r['high'])}] | {_fmt_p(r['p'])} | {_fmt_p(r['p_holm'])} |"
        )
    md += [
        "",
        "### Dropping the log specialists",
        "",
        "Each hybrid team against the metric-only team with the same centre.",
        "",
        "| Hybrid team | Without logs | Metric | n | Difference | 95% CI |",
        "|---|---|---|---|---|---|",
    ]
    for r in drop:
        md.append(
            f"| {r['team']} | {r['without_logs']} | {r['metric']} | {r['n']} | {_p(r['diff'])} | [{_p(r['low'])}, {_p(r['high'])}] |"
        )
    md += [
        "",
        "### Message ablations",
        "",
        "One service's messages (metric and log) are emptied (drop) or replaced by another incident's for the same service (shuffle); 'root' is the true root-cause service, 'other' a random other one. LLM centres are ablated on the root service only.",
        "",
        "| Team | Ablation | n | Top-1 intact | Top-1 ablated | Change [95% CI] | Joint intact | Joint ablated | Change [95% CI] |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in abl:
        a, b = r["top1"], r["joint"]
        md.append(
            f"| {r['team']} | {r['ablation']} | {r['n']} | {100 * a['intact']:.1f} | {100 * a['ablated']:.1f} | {_p(a['change'])} [{_p(a['low'])}, {_p(a['high'])}] | "
            f"{100 * b['intact']:.1f} | {100 * b['ablated']:.1f} | {_p(b['change'])} [{_p(b['low'])}, {_p(b['high'])}] |"
        )
    shown = [n for n in names if n in {t["system"] for t in table}][:8]
    md += [
        "",
        "### Joint accuracy by fault type and system",
        "",
        "| Group | n | " + " | ".join(shown) + " |",
        "|---|---|" + "---|" * len(shown),
    ]
    for g in split:
        md.append(
            f"| {g['group']} | {g['n']} | "
            + " | ".join(
                "-" if g.get(n) is None else f"{100 * g[n]:.1f}" for n in shown
            )
            + " |"
        )
    md += [
        "",
        "## Setup",
        "",
        "- **Data and splits:** as step 1 (`docs/results/r10_log_triage.md`): RCAEval RE2, SS and OB, repetition 1 = dev, repetitions 2 and 3 = test. Singles are re-run here (same code and budgets as step 1: 4,096-token centre prompt, 8,192 / 32,768 oracle context) so every comparison is paired per incident; step 1's tables hold no per-incident records.",
        "- **Metric specialist:** per service, score = largest robust deviation (IQR units) of its metrics after the alert, onset = earliest first-deviation time among its top 3 metrics (z >= 3), top metrics with direction. Typed JSON (`METRIC_SCHEMA`), tens of bytes.",
        "- **Log specialist:** per service, its own templated log lines in a window of -120 s to +300 s around the reported onset, ranked by rate change against the earlier baseline, at most 1,500 tokens; typed JSON (`LOG_SCHEMA`): fault, confidence, up to 3 evidence templates. Zero-shot, no training. A reply that is not valid JSON for the schema becomes an empty report (fault unknown).",
        "- **Centres:** rule = argmax score, fault from the top metric kind (`metric`) or the log hypothesis of that service (`log`, or only when confidence is at least 0.5 / 0.8, else the metric kind); LLM = qwen2.5-coder:14b reading all typed messages (JSON) within the 4,096-token prompt.",
        "- **Object layer:** the hybrid team with the rule centre also ran through `CommandCentre`, envelopes and `TraceLog` on a subset (`docs/results/r10_step2.json`, key `runtime`).",
        "- **Not done:** no training of any specialist or centre, no learned compression, no KV handoff.",
        "",
    ]
    payload = {
        "n_test_incidents": n_test,
        "policies": policies,
        "dev_policy_table": dev_table,
        "table": [
            {
                k: (v if not isinstance(v, list) else float(np.mean(v)))
                for k, v in t.items()
            }
            for t in table
        ],
        "verdict": verdicts,
        "comparisons": cmp_rows,
        "drop_logs": drop,
        "ablations": abl,
        "distributed_baro_check": dist,
    }
    return "\n".join(md), payload
