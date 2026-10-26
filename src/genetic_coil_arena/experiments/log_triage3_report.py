"""R10 step 3 report: trained fault-type specialists against BARO, and the G4c verdict.

Reads the per-incident records written by ``scripts/r10_step3.py teams``. Every number is on
the test split (SS/OB repetitions 2 and 3); nothing here selects anything.

**G4c rule** (``docs/goals.md``): a trained team (rule centre picks the service by the metric
anomaly score; the fault type comes from a trained specialist's typed report for that service)
passes if its joint accuracy (service and fault both right) beats BARO by at least 5 points,
the 95% paired-bootstrap CI of the difference excludes 0, the Holm-adjusted Wilcoxon p over
the trained teams is below 0.05, and top-1 is non-inferior (CI lower bound of the difference
above -3 points). Killed if every trained team is within the CI of BARO on joint accuracy.
The three trained teams are metric, log and both (fused); the TT-only metric team is a
diagnostic of cross-system transfer and is not part of the Holm family.
"""

from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Optional

import numpy as np

from ..envs.logtriage.rcaeval import FAULT_TYPES
from ..evaluation.stats import holm_correction, paired_bootstrap_ci
from . import log_triage2_report as rp2
from . import log_triage_report as rp

MARGIN = rp2.MARGIN
NON_INFERIORITY = rp2.NON_INFERIORITY
REFERENCE = "baro_metrics"
MAIN = ("team_trained_metric", "team_trained_log", "team_trained_both")
DIAGNOSTIC = ("team_trained_metric_ttonly",)
ORDER = (
    "baro_metrics",
    "team_metric_rule",
    *MAIN,
    *DIAGNOSTIC,
)
LABELS = {
    "baro_metrics": "BARO on all metrics + metric-kind fault (non-LLM)",
    "team_metric_rule": "step 2: metric specialists, rule centre (distributed BARO)",
    "team_trained_metric": "trained metric fault specialist, rule centre",
    "team_trained_log": "trained log fault specialist (LoRA Qwen3-1.7B), rule centre",
    "team_trained_both": "both specialists fused (rule chosen by cross-validation)",
    "team_trained_metric_ttonly": "diagnostic: metric specialist trained on Train Ticket only",
}


class Systems3(rp2.Systems):
    """Per-incident vectors on the test split, in step 3's own order and names."""

    def keys(self) -> list[tuple[str, str]]:
        order = {c: i for i, c in enumerate(ORDER)}
        return sorted(
            (k for k in self.by_key if k[1] in order), key=lambda k: order[k[1]]
        )

    def name(self, key: tuple[str, str]) -> str:
        return key[1]

    def key_of(self, cond: str) -> Optional[tuple[str, str]]:
        return next((k for k in self.by_key if k[1] == cond), None)


def g4c_verdicts(sys_: Systems3) -> dict[str, Any]:
    """The G4c rule for every main trained team, and the overall verdict."""
    ref = sys_.key_of(REFERENCE)
    out: dict[str, Any] = {}
    if ref is None:
        return {"overall": "NOT RUN", "teams": out}
    for cond in MAIN:
        key = sys_.key_of(cond)
        if key is None:
            continue
        j = rp2.paired(sys_.vec(*key, "joint"), sys_.vec(*ref, "joint"), seed=2)
        t = rp2.paired(sys_.vec(*key, "top1"), sys_.vec(*ref, "top1"), seed=3)
        if j is None or t is None:
            continue
        out[cond] = {
            "joint": j,
            "top1": t,
            "team_joint": float(np.mean(list(sys_.vec(*key, "joint").values()))),
            "baro_joint": float(np.mean(list(sys_.vec(*ref, "joint").values()))),
        }
    if out:
        adj = holm_correction([v["joint"]["p"] for v in out.values()])
        for v, a in zip(out.values(), adj):
            v["p_holm"] = float(a)
            v["non_inferior_top1"] = bool(v["top1"]["low"] > NON_INFERIORITY)
            j = v["joint"]
            if (
                j["low"] > 0
                and j["diff"] >= MARGIN
                and v["p_holm"] < 0.05
                and v["non_inferior_top1"]
            ):
                v["verdict"] = "PASS"
            elif j["low"] > 0 and j["diff"] > 0:
                v["verdict"] = "INCONCLUSIVE"
            else:
                v["verdict"] = "KILL"
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


def comparison_pairs(sys_: Systems3) -> list[dict[str, Any]]:
    """Joint-accuracy differences between named systems, Holm over the table."""
    present = {k[1] for k in sys_.keys()}
    pairs: list[tuple[str, str]] = []
    for cond in (*MAIN, *DIAGNOSTIC):
        if cond in present:
            pairs += [(cond, "baro_metrics")]
            if "team_metric_rule" in present:
                pairs.append((cond, "team_metric_rule"))
    for a, b in (
        ("team_trained_both", "team_trained_metric"),
        ("team_trained_both", "team_trained_log"),
        ("team_trained_metric", "team_trained_metric_ttonly"),
    ):
        if a in present and b in present:
            pairs.append((a, b))
    rows = []
    for a, b in pairs:
        r = rp2.paired(
            sys_.vec(*sys_.key_of(a), "joint"),  # type: ignore[misc]
            sys_.vec(*sys_.key_of(b), "joint"),  # type: ignore[misc]
            seed=7,
        )
        if r:
            rows.append({"a": a, "b": b, **r})
    if rows:
        for row, adj in zip(rows, holm_correction([r["p"] for r in rows])):
            row["p_holm"] = float(adj)
    return rows


def ablation_rows(sys_: Systems3) -> list[dict[str, Any]]:
    out = []
    for key in sys_.keys():
        model, cond = key
        if cond not in (*MAIN, *DIAGNOSTIC):
            continue
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
            row: dict[str, Any] = {"team": cond, "ablation": mode, "n": len(ids)}
            for what in ("top1", "joint"):
                a = np.array(
                    [rp2._score(intact[i], what, None) for i in ids], dtype=float
                )
                b = np.array([rp2._score(abl[i], what, None) for i in ids], dtype=float)
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


def breakdown(sys_: Systems3, names: Sequence[str], what: str) -> list[dict[str, Any]]:
    keymap = {sys_.name(k): k for k in sys_.keys()}
    groups: dict[str, list[str]] = defaultdict(list)
    for r in sys_.rows:
        if r["condition"] != REFERENCE:
            continue
        groups[f"fault {r['fault']}"].append(r["incident"])
        groups[f"system {r['system']}"].append(r["incident"])
    out = []
    for g, ids in sorted(groups.items()):
        ids_set = set(ids)
        row: dict[str, Any] = {"group": g, "n": len(ids_set)}
        for n in names:
            if n in keymap:
                v = sys_.vec(*keymap[n], what)
                sel = [x for i, x in v.items() if i in ids_set]
                row[n] = float(np.mean(sel)) if sel else None
        out.append(row)
    return out


def same_ranking_check(sys_: Systems3) -> Optional[dict[str, Any]]:
    a = {r["incident"]: r for r in sys_.by_key.get(("none", REFERENCE), [])}
    out = {}
    for cond in (*MAIN, *DIAGNOSTIC):
        key = sys_.key_of(cond)
        if key is None:
            continue
        b = {r["incident"]: r for r in sys_.by_key[key]}
        ids = sorted(set(a) & set(b))
        if ids:
            out[cond] = float(
                np.mean([a[i]["ranking"][0] == b[i]["ranking"][0] for i in ids])
            )
    return out or None


def _p(x: float) -> str:
    return f"{100 * x:+.1f}"


def _fmt_p(p: float) -> str:
    return f"{p:.3g}"


def _rows_for_table(sys_: Systems3) -> list[dict[str, Any]]:
    table = []
    for key in sys_.keys():
        sel = sys_.by_key[key]
        table.append(
            {
                "system": sys_.name(key),
                "label": LABELS[key[1]],
                "n": len(sel),
                "top1": [r["top1"] for r in sel],
                "top3": [r["top3"] for r in sel],
                "fault": [r["fault_ok"] for r in sel],
                "joint": [int(r["top1"]) * int(r["fault_ok"]) for r in sel],
                "bytes": float(
                    np.mean([(r.get("extra") or {}).get("bytes", 0) for r in sel])
                ),
            }
        )
    return table


def build_report(
    records: Sequence[dict[str, Any]],
    extras: Optional[dict[str, Any]] = None,
    analysis: Optional[str] = None,
    note: Optional[str] = None,
) -> tuple[str, dict[str, Any]]:
    extras = extras or {}
    rows = rp.dedupe(records)
    sys_ = Systems3(rows, {})
    table = _rows_for_table(sys_)
    verdicts = g4c_verdicts(sys_)
    cmp_rows = comparison_pairs(sys_)
    abl = ablation_rows(sys_)
    names = [t["system"] for t in table]
    joint_split = breakdown(sys_, names, "joint")
    fault_split = breakdown(sys_, names, "fault")
    same = same_ranking_check(sys_)
    n_test = len({r["incident"] for r in sys_.rows if r["condition"] == REFERENCE})
    md: list[str] = ["# R10 step 3: trained fault-type specialists (goal G4c)", ""]
    if note:
        md += [f"*{note}*", ""]
    md += [
        "Step 3 of R10 on RCAEval RE2. The rule centre picks the service by the metric specialists' anomaly score "
        "(as step 2) and takes the fault type from a *trained* specialist's typed report for that service: a numpy "
        "classifier on per-kind metric features (CPU) and a LoRA adapter on Qwen3-1.7B reading the service's own "
        "log view (GPU), alone or fused. Training data: Train Ticket (all repetitions) and Sock Shop / Online "
        "Boutique repetition 1; the test split (SS/OB repetitions 2 and 3, "
        f"{n_test} incidents) is never fitted or selected on.",
        "",
        "## Verdict",
        "",
        f"**G4c: {verdicts['overall']}.** Pass: a trained team beats BARO on joint accuracy by at least 5 points "
        "(paired-bootstrap 95% CI excludes 0, Holm-adjusted p < 0.05 over the trained teams) with top-1 non-inferior "
        "(CI lower bound above -3 points). Kill: every trained team is within the CI of BARO on joint accuracy.",
        "",
    ]
    if verdicts["teams"]:
        md += [
            "| Trained team | Team joint | BARO joint | Difference [95% CI] | p (Holm) | Top-1 difference [95% CI] | Verdict |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, v in verdicts["teams"].items():
            j, t = v["joint"], v["top1"]
            md.append(
                f"| {name} | {100 * v['team_joint']:.1f} | {100 * v['baro_joint']:.1f} | "
                f"{_p(j['diff'])} [{_p(j['low'])}, {_p(j['high'])}] | {_fmt_p(v['p_holm'])} | "
                f"{_p(t['diff'])} [{_p(t['low'])}, {_p(t['high'])}] | **{v['verdict']}** |"
            )
        md.append("")
    md += [
        "All trained teams rank services by the same metric score as BARO"
        + (
            "; check: the top service equals BARO's on "
            + ", ".join(f"{100 * x:.1f}% ({c})" for c, x in same.items())
            + " of the test incidents"
            if same
            else ""
        )
        + ", so top-1 is BARO's by construction and the whole difference is the fault type.",
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
        "for top-1 is about 7.5, for fault type 16.7, for joint about 1.3. Bytes are the typed reports' compact JSON "
        "per incident (the trained reports carry six fault probabilities each).",
        "",
        "| System | n | Top-1 | Top-3 | Fault type | Joint | Bytes / incident |",
        "|---|---|---|---|---|---|---|",
    ]
    for t in table:
        md.append(
            f"| {t['system']} | {t['n']} | {rp._cell(t['top1'])} | {rp._cell(t['top3'])} | "
            f"{rp._cell(t['fault'])} | {rp._cell(t['joint'])} | {t['bytes']:,.0f} |"
        )
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
            f"| {r['a']} | {r['b']} | {r['n']} | {_p(r['diff'])} | [{_p(r['low'])}, {_p(r['high'])}] | "
            f"{_fmt_p(r['p'])} | {_fmt_p(r['p_holm'])} |"
        )
    shown = list(names)[:8]
    for title, split in (
        ("Joint accuracy by fault type and system", joint_split),
        (
            "Fault-type accuracy by fault type and system (service right or wrong)",
            fault_split,
        ),
    ):
        md += [
            "",
            f"### {title}",
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
        "### Message ablations",
        "",
        "`drop_*` / `shuffle_*` empty or swap both reports of one service for another incident's (root = the true "
        "root-cause service, other = a random other one). `*_fault_root` touches only the root service's fault "
        "reports (scores and onset stay), so the centre falls back to the metric-kind heuristic (drop) or reads "
        "another incident's fault report (shuffle).",
        "",
        "| Team | Ablation | n | Top-1 intact | Top-1 ablated | Change [95% CI] | Joint intact | Joint ablated | Change [95% CI] |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in abl:
        a, b = r["top1"], r["joint"]
        md.append(
            f"| {r['team']} | {r['ablation']} | {r['n']} | {100 * a['intact']:.1f} | {100 * a['ablated']:.1f} | "
            f"{_p(a['change'])} [{_p(a['low'])}, {_p(a['high'])}] | {100 * b['intact']:.1f} | "
            f"{100 * b['ablated']:.1f} | {_p(b['change'])} [{_p(b['low'])}, {_p(b['high'])}] |"
        )
    md += _training_sections(extras)
    md += [
        "",
        "## Setup",
        "",
        "- **Splits:** RCAEval RE2. Training = Train Ticket (all repetitions) + SS/OB repetition 1; test = SS/OB "
        "repetitions 2 and 3. A test incident repeats a dev case (same service and fault), so models fitted on "
        "SS/OB-dev may partly memorise cases; leave-one-system-out cross-validation and the TT-only diagnostic "
        "measure cross-system transfer instead. `scripts/r10_step3.py data` checks the split and `tests/test_log_triage3.py` pins it.",
        "- **Metric fault specialist:** per service, features by metric kind from its own metrics (z, direction, "
        "relative change, onset per kind; share of the strongest kind; count of kinds beyond z 5 and 20). A "
        "class-balanced numpy softmax regression or small MLP ensemble, chosen by leave-one-system-out accuracy "
        "over TT / SS-dev / OB-dev. Typed `TRAINED_METRIC_SCHEMA` report: step 2's fields plus `fault` and six "
        "`fault_probs`.",
        "- **Log fault specialist:** Qwen3-1.7B (bfloat16) with one LoRA adapter (rank 8, all linear layers), "
        "trained on the root service's step-2 log view (-120 s to +300 s around the reported onset, 1,500 tokens, "
        "three onset jitters as augmentation) with the fault letter as the target. Scored by closed-set letter "
        "log-probabilities. Typed `TRAINED_LOG_SCHEMA` report.",
        "- **Centre:** rule. Ranking by metric score; fault = argmax of the specialist's distribution for the top "
        "service (metric, log) or of the fused distribution (fusion rule chosen on out-of-fold predictions of the "
        "training data only). Missing or empty reports fall back to step 2's metric-kind heuristic.",
        "- **Not done:** no CommandCentre/envelope run of the trained team, no LLM centre, no trained service ranking.",
        "",
    ]
    payload = {
        "n_test_incidents": n_test,
        "table": [
            {
                k: (v if not isinstance(v, list) else float(np.mean(v)))
                for k, v in t.items()
            }
            for t in table
        ],
        "verdict": verdicts,
        "comparisons": cmp_rows,
        "ablations": abl,
        "same_top1_as_baro": same,
        "joint_by_group": joint_split,
        "fault_by_group": fault_split,
        **{k: v for k, v in extras.items() if v is not None},
    }
    return "\n".join(md), payload


def _training_sections(extras: dict[str, Any]) -> list[str]:
    md: list[str] = []
    metric = extras.get("metric")
    if metric:
        md += [
            "",
            "## Training and cross-validation (training data only)",
            "",
            f"Metric fault specialist: {metric['n_train']} training rows (root-cause services); leave-one-system-out "
            "accuracy of the chosen configuration against step 2's metric-kind heuristic on the same rows.",
            "",
            "| System held out | Trained specialist | Metric-kind heuristic |",
            "|---|---|---|",
        ]
        for s, a in metric["oof_per_system"].items():
            md.append(
                f"| {s} | {100 * a:.1f} | {100 * metric['heuristic_per_system'].get(s, float('nan')):.1f} |"
            )
        md += [
            "",
            f"Chosen configuration: `{metric['chosen']}`. Per fault type (all training systems pooled):",
            "",
            "| Fault | n | Specialist (out-of-fold) | Heuristic |",
            "|---|---|---|---|",
        ]
        for f in FAULT_TYPES:
            d = metric["per_fault"].get(f)
            if d:
                md.append(
                    f"| {f} | {d['n']} | {100 * d['oof_accuracy']:.1f} | {100 * d['heuristic_accuracy']:.1f} |"
                )
        md += [
            "",
            "| Configuration | "
            + " | ".join(next(iter(metric["grid"]))["per_system"])
            + " | Mean |",
            "|---|" + "---|" * (len(next(iter(metric["grid"]))["per_system"]) + 1),
        ]
        for g in metric["grid"]:
            md.append(
                f"| {g['config']} | "
                + " | ".join(f"{100 * v:.1f}" for v in g["per_system"].values())
                + f" | {100 * g['mean']:.1f} |"
            )
    fusion = extras.get("fusion")
    if fusion and fusion.get("table"):
        md += [
            "",
            f"Fusion rule, chosen on {fusion['n_oof']} out-of-fold root-service reports "
            f"(`{fusion.get('n_by_system')}`): **{fusion['chosen']}**.",
            "",
            "| Rule | "
            + " | ".join(k for k in next(iter(fusion["table"].values())))
            + " |",
            "|---|" + "---|" * len(next(iter(fusion["table"].values()))),
        ]
        for rule, per in fusion["table"].items():
            md.append(
                f"| {rule} | "
                + " | ".join(f"{100 * v:.1f}" for v in per.values())
                + " |"
            )
    lora = extras.get("lora")
    if lora:
        md += [
            "",
            "Log fault specialist (LoRA): one adapter per fold plus the final one.",
            "",
            "| Fold | Held out | Fit incidents | Examples | Steps | Final loss | Train (s) | Peak VRAM (MiB) | Held-out fault accuracy |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for fold, d in sorted(lora.items()):
            acc = d.get("heldout_accuracy")
            md.append(
                f"| {fold} | {d.get('held_out') or '-'} | {d['n_fit_incidents']} | {d['n_examples']} | {d['steps']} | "
                f"{d['mean_last_loss']:.3f} | {d['train_s']:.0f} | {d['peak_mib']:.0f} | "
                f"{'-' if acc is None else f'{100 * acc:.1f}'} |"
            )
    return md
