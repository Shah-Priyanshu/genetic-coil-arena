"""R10 step 5 report (goal G4e): an LLM final reader over the fusion's top-3 shortlist.

**G4e rule** (``docs/goals.md``): a reader (``rerank3``) passes if its RE3 test root-cause
top-1 beats ``nonllm_fusion`` by at least 5 points, the 95% paired-bootstrap CI of the
difference excludes 0, and the Holm-adjusted Wilcoxon p over the reader models is below
0.05. Killed if every model's CI contains 0 or lies below it; otherwise inconclusive. The
confidence-threshold variant (``rerank3_thr``, threshold chosen per model on dev) and the
shuffle ablations are reported beside it and do not enter the verdict.
"""

from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Optional

import numpy as np

from ..envs.logtriage import rerank as rr
from ..evaluation.stats import holm_correction
from . import log_triage4_report as rp4
from . import log_triage_report as rp
from .log_triage5 import CONDITIONS5, FUSION

MARGIN = 0.05
THR = "rerank3_thr"
LABELS = {
    FUSION: "non-LLM fusion (dev-fitted reciprocal ranks)",
    "rerank3": "LLM reads the fusion's top-3 (ranker order and score shown)",
    THR: "same, overriding the fusion's top-1 only above a dev-chosen confidence",
    "rerank3:shuffle": "ablation: candidates shuffled, ranker position still shown",
    "rerank3:shuffle_blind": "ablation: candidates shuffled, ranker position hidden",
}


class Systems5(rp4.Systems):
    def main_keys(self) -> list[tuple[str, str]]:
        order = {c: i for i, c in enumerate(LABELS)}
        found = [k for k in self.by_key if k[1] in order]
        return sorted(found, key=lambda k: (order[k[1]], k[0]))


def with_threshold(row: dict[str, Any], thr: float) -> dict[str, Any]:
    """The ``rerank3`` row, overriding the fusion's top-1 only at confidence >= ``thr``."""
    ex = row.get("extra") or {}
    keep = bool(row.get("valid", 1)) and (ex.get("confidence", 0.0) >= thr)
    fusion = list(ex.get("fusion_ranking") or row["ranking"])
    choice = ex.get("choice")
    if keep and choice is not None:
        rk = [choice] + [n for n in fusion if n != choice]
    else:
        rk = fusion
    out = dict(row)
    out.update(
        condition=THR,
        ranking=rk,
        top1=int(bool(rk) and rk[0] == row["root"]),
        top3=int(row["root"] in rk[:3]),
        extra=dict(ex, threshold=thr, overridden=int(bool(rk) and rk[0] != fusion[0])),
    )
    return out


def pick_thresholds(
    rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Per reader model, the threshold with the best dev top-1 (ties: the higher one)."""
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r["condition"] == "rerank3" and not rp4.is_test(r):
            by[r["model"]].append(r)
    chosen, table = {}, {}
    for model, dev in by.items():
        means = {
            t: float(np.mean([with_threshold(r, t)["top1"] for r in dev]))
            for t in rr.CONFIDENCE_GRID
        }
        chosen[model] = max(means, key=lambda t: (means[t], t))
        table[model] = {"n_dev": len(dev), "top1": means, "chosen": chosen[model]}
    return chosen, table


def derive_threshold_rows(
    rows: Sequence[dict[str, Any]], chosen: dict[str, float]
) -> list[dict[str, Any]]:
    return [
        with_threshold(r, chosen[r["model"]])
        for r in rows
        if r["condition"] == "rerank3" and r["model"] in chosen
    ]


def g4e_verdicts(sys_: Systems5) -> dict[str, Any]:
    ref = ("none", FUSION)
    out: dict[str, Any] = {
        "reference": None,
        "models": {},
        "variant": {},
        "overall": "NOT RUN",
    }
    if not sys_.vec(ref):
        return out
    out["reference"] = {
        "system": FUSION,
        "test_top1": sys_.test_mean(ref),
        "test_top3": float(np.mean([r["top3"] for r in sys_.by_key[ref]])),
    }
    for cond, slot in (("rerank3", "models"), (THR, "variant")):
        keys = sorted(k for k in sys_.by_key if k[1] == cond)
        res = {}
        for key in keys:
            cmp_ = rp4.paired(sys_.vec(key), sys_.vec(ref), seed=11)
            if cmp_:
                res[key[0]] = {"top1": sys_.test_mean(key), **cmp_}
        if res:
            adj = holm_correction([v["p"] for v in res.values()])
            for v, a in zip(res.values(), adj):
                v["p_holm"] = float(a)
                if v["low"] > 0 and v["diff"] >= MARGIN and a < 0.05:
                    v["verdict"] = "PASS"
                elif v["low"] > 0:
                    v["verdict"] = "INCONCLUSIVE"
                else:
                    v["verdict"] = "KILL"
        out[slot] = res
    kinds = [v["verdict"] for v in out["models"].values()]
    if kinds:
        out["overall"] = (
            "PASS"
            if "PASS" in kinds
            else "KILL" if all(k == "KILL" for k in kinds) else "INCONCLUSIVE"
        )
    return out


def override_rows(sys_: Systems5) -> list[dict[str, Any]]:
    """How often the reader changes the fusion's top-1, and whether that helps (test split)."""
    rows = []
    for key in sorted(sys_.by_key):
        if not (key[1] in CONDITIONS5 or key[1] == THR):
            continue
        recs = sys_.by_key[key]
        n = len(recs)
        over = [r for r in recs if (r.get("extra") or {}).get("overridden")]
        fix = sum(1 for r in over if r["top1"])
        harm = sum(
            1 for r in over if not r["top1"] and r["root"] == r["extra"]["fusion_top1"]
        )
        rows.append(
            {
                "model": key[0],
                "condition": key[1],
                "n": n,
                "valid": float(np.mean([r.get("valid", 1) for r in recs])),
                "overrides": len(over),
                "right": fix,
                "wrong": harm,
                "both_wrong": len(over) - fix - harm,
                "net": fix - harm,
            }
        )
    return rows


def copy_rows(sys_: Systems5) -> list[dict[str, Any]]:
    """Does the reader just copy the fusion's rank 1 (or the first block shown)?"""
    rows = []
    for key in sorted(sys_.by_key):
        if key[1] not in CONDITIONS5:
            continue
        ex = [r["extra"] for r in sys_.by_key[key] if r.get("extra", {}).get("choice")]
        if not ex:
            continue
        rows.append(
            {
                "model": key[0],
                "condition": key[1],
                "n": len(ex),
                "is_fusion_top1": float(
                    np.mean([e["choice"] == e["fusion_top1"] for e in ex])
                ),
                "is_first_shown": float(
                    np.mean([e["choice"] == e["shown"][0] for e in ex])
                ),
            }
        )
    return rows


def _p(x: float) -> str:
    return f"{100 * x:+.1f}"


def _fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def build_report(
    rows_in: Sequence[dict[str, Any]],
    analysis: Optional[str] = None,
    note: Optional[str] = None,
) -> tuple[str, dict[str, Any]]:
    rows = rp4.dedupe(rows_in)
    chosen, thr_table = pick_thresholds(rows)
    rows = rows + derive_threshold_rows(rows, chosen)
    sys_ = Systems5(rows, {})
    verdict = g4e_verdicts(sys_)
    overrides = override_rows(sys_)
    copies = copy_rows(sys_)
    ablations = rp4.ablation_rows(sys_)
    per_system = rp4.per_system_rows(sys_)
    payload: dict[str, Any] = {
        "verdict": verdict,
        "thresholds": thr_table,
        "overrides": overrides,
        "copy": copies,
        "ablations": ablations,
        "per_system": per_system,
    }
    lines = ["# R10 step 5: an LLM final reader over the fusion's top-3 (goal G4e)", ""]
    if note:
        lines += [note, ""]
    lines += [f"**G4e verdict: {verdict['overall']}**", ""]
    ref = verdict["reference"]
    if ref:
        lines += [
            f"Reference: {FUSION}, test top-1 {100 * ref['test_top1']:.1f}, top-3 "
            f"{100 * ref['test_top3']:.1f} (RE3 test split, repetitions 2-4). Readers see the "
            "fusion's top three only, so their top-3 equals the fusion's by construction.",
            "",
        ]
    lines += ["## Main results (test split, root-cause top-1 and top-3, %)", ""]
    lines += [
        "| system | model | n | top-1 [95% CI] | top-3 | valid replies |",
        "|---|---|---|---|---|---|",
    ]
    for key in sys_.main_keys():
        recs = sys_.by_key[key]
        model = "-" if key[0] == "none" else rp4.short(key[0])
        lines.append(
            f"| {LABELS[key[1]]} | {model} | {len(recs)} | {rp._cell([r['top1'] for r in recs])} | "
            f"{100 * np.mean([r['top3'] for r in recs]):.1f} | "
            f"{100 * np.mean([r.get('valid', 1) for r in recs]):.0f}% |"
        )
    lines.append("")
    for title, slot in (
        ("G4e comparisons (paired, test split)", "models"),
        ("Threshold variant (not part of the verdict)", "variant"),
    ):
        if verdict[slot]:
            lines += [f"## {title}", ""]
            lines += [
                "| model | top-1 | reader - fusion [95% CI] | p (Holm) | verdict |",
                "|---|---|---|---|---|",
            ]
            for m, v in verdict[slot].items():
                lines.append(
                    f"| {rp4.short(m)} | {100 * v['top1']:.1f} | {_p(v['diff'])} "
                    f"[{_p(v['low'])}, {_p(v['high'])}] | {_fmt_p(v['p_holm'])} | {v['verdict']} |"
                )
            lines.append("")
    if thr_table:
        lines += ["## Override threshold chosen on dev", ""]
        lines += [
            "| model | n dev | dev top-1 by threshold | chosen |",
            "|---|---|---|---|",
        ]
        for m, t in thr_table.items():
            lines.append(
                f"| {rp4.short(m)} | {t['n_dev']} | "
                + ", ".join(f"{k:g}: {100 * v:.0f}" for k, v in t["top1"].items())
                + f" | {t['chosen']:g} |"
            )
        lines.append("")
    if overrides:
        lines += ["## Overrides of the fusion's top-1 (test split)", ""]
        lines += [
            "| model | condition | n | valid | overrides | right (fixes a miss) | wrong (breaks a hit) | both wrong | net |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in overrides:
            lines.append(
                f"| {rp4.short(r['model'])} | {r['condition']} | {r['n']} | {100 * r['valid']:.0f}% | "
                f"{r['overrides']} | {r['right']} | {r['wrong']} | {r['both_wrong']} | {r['net']:+d} |"
            )
        lines.append("")
    if copies:
        lines += ["## Does the reader copy the ranker? (test split)", ""]
        lines += [
            "| model | condition | n | picks the fusion's top-1 | picks the first block shown |",
            "|---|---|---|---|---|",
        ]
        for r in copies:
            lines.append(
                f"| {rp4.short(r['model'])} | {r['condition']} | {r['n']} | "
                f"{100 * r['is_fusion_top1']:.0f}% | {100 * r['is_first_shown']:.0f}% |"
            )
        lines.append("")
    if ablations:
        lines += ["## Shuffle ablations (test split, against the intact reader)", ""]
        lines += [
            "| system | ablation | intact top-1 | ablated top-1 | change [95% CI] |",
            "|---|---|---|---|---|",
        ]
        for r in ablations:
            lines.append(
                f"| {r['system']} | {r['ablation']} | {100 * r['intact_top1']:.1f} | {100 * r['top1']:.1f} | "
                f"{_p(r['diff'])} [{_p(r['low'])}, {_p(r['high'])}] |"
            )
        lines.append("")
    if per_system:
        lines += ["## Top-1 per RE3 system (test split, %)", ""]
        cols = sorted({k for r in per_system for k in r if k.startswith("RE3-")})
        lines += [
            "| system | " + " | ".join(c.replace("RE3-", "") for c in cols) + " |",
            "|---|" + "---|" * len(cols),
        ]
        for r in per_system:
            lines.append(
                f"| {r['system']} | "
                + " | ".join(f"{100 * r[c]:.0f}" if c in r else "-" for c in cols)
                + " |"
            )
        lines.append("")
    lines += [
        "## Analysis",
        "",
        rp.ANALYSIS_START,
        analysis or "(to be written after the runs)",
        rp.ANALYSIS_END,
        "",
    ]
    return "\n".join(lines), payload
