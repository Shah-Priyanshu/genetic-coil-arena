"""R10 step 4 report (goal G4d): non-LLM systems, LLM teams, ablations and the verdict.

Reads the records written by ``scripts/r10_step4.py`` (``gate.json`` from ``gate``, JSONL
rows from ``llm``/``single``). The dev split (repetition 1) picks the best non-LLM system and
each LLM team's rule weighting; every reported number is on the test split (repetitions 2-4).

**G4d rule** (``docs/goals.md``): an LLM team (typed log reports from a model reading only
its own service's logs, fused with the metric specialists by a rule or an LLM centre) passes
if its root-cause top-1 beats the best non-LLM system (chosen on dev among BARO on metrics,
BARO on logs, template surprise, the new-template scorers and their dev-fitted fusion) by at
least 5 points, with the 95% CI of the paired difference excluding 0 and a Holm-adjusted
Wilcoxon p < 0.05 over the LLM teams. Killed if every LLM team's CI contains 0 or lies below
it; anything else is inconclusive. A stricter check against the non-LLM system that is best
on the test split itself is reported beside it.
"""

from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Optional

import numpy as np

from ..envs.logtriage import team4 as t4
from ..evaluation.stats import (
    holm_correction,
    paired_bootstrap_ci,
    wilcoxon_signed_rank,
)
from . import log_triage_report as rp

MARGIN = 0.05
NONLLM = "nonllm_"
LABELS = {
    "nonllm_baro_metrics": "BARO on service metrics",
    "nonllm_baro_logs": "BARO on log counts",
    "nonllm_surprise_logs": "template-surprise ranking",
    "nonllm_new_err": "new warning/error templates (count)",
    "nonllm_new_err_mass": "new warning/error templates (severity x log count)",
    "nonllm_new_err_idf": "new warning/error templates (discounted by spread)",
    "nonllm_new_any_mass": "all new templates (severity x log count)",
    "nonllm_new_ext": "new warning/error/lifecycle templates (count)",
    "nonllm_fusion": "reciprocal-rank fusion of metric and log scores (dev-fitted)",
    "team4_rule": "team: metric + LLM log specialists, rule centre",
    "team4_llm": "team: metric + LLM log specialists, LLM centre",
    "single_all": "single LLM, selected view of logs + metrics",
    "single_logs": "single LLM, selected view of logs only",
}


def is_test(row: dict[str, Any]) -> bool:
    return int(row.get("rep", 0)) != 1


def short(model: str) -> str:
    return model.replace("ollama:", "")


def dedupe(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in rows:
        if r.get("kind"):
            continue
        seen[(r["model"], r["condition"], r["incident"])] = r
    return list(seen.values())


def row_for_rule(row: dict[str, Any], rule: Optional[str]) -> dict[str, Any]:
    """The row with its ranking replaced by weighting ``rule`` (rule-centre rows only)."""
    ranks = (row.get("extra") or {}).get("rank_by_rule")
    if not rule or not ranks or rule not in ranks:
        return row
    rk = ranks[rule]
    out = dict(row)
    out["ranking"] = rk
    out["top1"] = int(bool(rk) and rk[0] == row["root"])
    out["top3"] = int(row["root"] in rk[:3])
    return out


def pick_rules(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, str], dict[str, Any]]:
    """Per log-specialist model, the rule weighting with the best dev top-1."""
    dev: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["condition"] != "team4_rule" or is_test(r):
            continue
        for rule in t4.RULE_WEIGHTS:
            dev[r["model"]][rule].append(row_for_rule(r, rule)["top1"])
    chosen, table = {}, {}
    order = list(t4.RULE_WEIGHTS)
    for model, by in dev.items():
        means = {k: float(np.mean(v)) for k, v in by.items()}
        chosen[model] = max(order, key=lambda k: (means[k], -order.index(k)))
        table[model] = {
            "n_dev": len(by[order[0]]),
            "top1": means,
            "chosen": chosen[model],
        }
    return chosen, table


class Systems:
    """Per-incident vectors of every system on the test split (and the dev means)."""

    def __init__(self, rows: Sequence[dict[str, Any]], rules: dict[str, str]):
        self.rules = rules
        self.test = [r for r in rows if is_test(r)]
        self.dev = [r for r in rows if not is_test(r)]
        self.by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        self.dev_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for r in self.test:
            self.by_key[(r["model"], r["condition"])].append(self._fix(r))
        for r in self.dev:
            self.dev_by_key[(r["model"], r["condition"])].append(self._fix(r))

    def _fix(self, r: dict[str, Any]) -> dict[str, Any]:
        base = r["condition"].split(":")[0]
        return (
            row_for_rule(r, self.rules.get(r["model"])) if base == "team4_rule" else r
        )

    def vec(self, key: tuple[str, str], what: str = "top1") -> dict[str, float]:
        return {r["incident"]: float(r[what]) for r in self.by_key.get(key, [])}

    def main_keys(self) -> list[tuple[str, str]]:
        order = {c: i for i, c in enumerate(LABELS)}
        found = [k for k in self.by_key if k[1] in order]
        return sorted(found, key=lambda k: (order[k[1]], k[0]))

    def name(self, key: tuple[str, str]) -> str:
        model, cond = key
        return cond if model == "none" else f"{cond}[{short(model)}]"

    def nonllm_keys(self) -> list[tuple[str, str]]:
        return [k for k in self.by_key if k[1].startswith(NONLLM)]

    def dev_mean(self, key: tuple[str, str]) -> float:
        rows = self.dev_by_key.get(key, [])
        return float(np.mean([r["top1"] for r in rows])) if rows else float("nan")

    def test_mean(self, key: tuple[str, str]) -> float:
        v = list(self.vec(key).values())
        return float(np.mean(v)) if v else float("nan")


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


def best_nonllm(
    sys_: Systems,
) -> tuple[Optional[tuple[str, str]], Optional[tuple[str, str]]]:
    """(best on dev, best on test) among the non-LLM systems."""
    keys = sys_.nonllm_keys()
    if not keys:
        return None, None
    by_dev = max(keys, key=lambda k: (sys_.dev_mean(k), sys_.test_mean(k)))
    by_test = max(keys, key=lambda k: sys_.test_mean(k))
    return by_dev, by_test


def llm_team_keys(sys_: Systems) -> list[tuple[str, str]]:
    return [k for k in sys_.by_key if k[1] in ("team4_rule", "team4_llm")]


def g4d_verdicts(sys_: Systems) -> dict[str, Any]:
    ref_dev, ref_test = best_nonllm(sys_)
    out: dict[str, Any] = {"reference": None, "teams": {}, "overall": "NOT RUN"}
    teams = llm_team_keys(sys_)
    if ref_dev is None or not teams:
        return out
    out["reference"] = {
        "dev_best": sys_.name(ref_dev),
        "dev_best_test_top1": sys_.test_mean(ref_dev),
        "test_best": sys_.name(ref_test),  # type: ignore[arg-type]
        "test_best_test_top1": sys_.test_mean(ref_test),  # type: ignore[arg-type]
    }
    for key in teams:
        v = sys_.vec(key)
        main = paired(v, sys_.vec(ref_dev), seed=2)
        strict = paired(v, sys_.vec(ref_test), seed=3)  # type: ignore[arg-type]
        if main is None or strict is None:
            continue
        out["teams"][sys_.name(key)] = {
            "top1": sys_.test_mean(key),
            "vs_dev_best": main,
            "vs_test_best": strict,
            "rule": sys_.rules.get(key[0]) if key[1] == "team4_rule" else None,
        }
    teams_out = out["teams"]
    if teams_out:
        adj = holm_correction([t["vs_dev_best"]["p"] for t in teams_out.values()])
        for t, a in zip(teams_out.values(), adj):
            t["p_holm"] = float(a)
            d = t["vs_dev_best"]
            if d["low"] > 0 and d["diff"] >= MARGIN and t["p_holm"] < 0.05:
                t["verdict"] = "PASS"
            elif d["low"] > 0:
                t["verdict"] = "INCONCLUSIVE"
            else:
                t["verdict"] = "KILL"
        kinds = [t["verdict"] for t in teams_out.values()]
        out["overall"] = (
            "PASS"
            if "PASS" in kinds
            else "KILL" if all(k == "KILL" for k in kinds) else "INCONCLUSIVE"
        )
    return out


def single_vs_team(sys_: Systems) -> list[dict[str, Any]]:
    """Each LLM team against the best single LLM (chosen on the test incidents, which
    favours the single model)."""
    singles = [k for k in sys_.by_key if k[1] in ("single_all", "single_logs")]
    rows = []
    for team in llm_team_keys(sys_):
        best = None
        for s in singles:
            m = sys_.test_mean(s)
            if best is None or m > best[0]:
                best = (m, s)
        if best is None:
            continue
        cmp_ = paired(sys_.vec(team), sys_.vec(best[1]), seed=5)
        if cmp_:
            rows.append(
                {
                    "team": sys_.name(team),
                    "single": sys_.name(best[1]),
                    "single_top1": best[0],
                    **cmp_,
                }
            )
    return rows


def ablation_rows(sys_: Systems) -> list[dict[str, Any]]:
    rows = []
    for model, cond in sorted(sys_.by_key):
        if ":" not in cond:
            continue
        base, mode = cond.split(":", 1)
        intact = sys_.vec((model, base))
        cmp_ = paired(sys_.vec((model, cond)), intact, seed=6)
        if cmp_ is None:
            continue
        rows.append(
            {
                "system": sys_.name((model, base)),
                "ablation": mode,
                "top1": sys_.test_mean((model, cond)),
                "intact_top1": float(
                    np.mean([intact[i] for i in sys_.vec((model, cond))])
                ),
                **cmp_,
            }
        )
    return rows


def per_system_rows(sys_: Systems) -> list[dict[str, Any]]:
    rows = []
    for key in sys_.main_keys():
        by: dict[str, list[float]] = defaultdict(list)
        for r in sys_.by_key[key]:
            by[r["system"]].append(float(r["top1"]))
        rows.append(
            {
                "system": sys_.name(key),
                **{s: float(np.mean(v)) for s, v in sorted(by.items())},
                "n": {s: len(v) for s, v in sorted(by.items())},
            }
        )
    return rows


# ------------------------------------------------------------------------- text


def _p(x: float) -> str:
    return f"{100 * x:+.1f}"


def _fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def gate_text(payload: dict[str, Any]) -> str:
    """The data description and gate tables of a ``gate.json`` payload."""
    lines = ["## Data (RCAEval RE3)", ""]
    lines += [
        "| system | incidents | cases | root services | candidate services | faults | median log lines |",
        "|---|---|---|---|---|---|---|",
    ]
    for s, d in payload["data"].items():
        lines.append(
            f"| {s} | {d['n']} | {d['cases']} | {', '.join(d['root_services'])} | "
            f"{d['n_candidate_services']} | {', '.join(f'{k}:{v}' for k, v in d['faults'].items())} | "
            f"{d['median_lines']:,} |"
        )
    lines += ["", "## Gate: do the logs carry root-cause signal?", ""]
    lines += [
        "| system | split | root has a new warning/error template | mean such templates, root | mean, other services | root strictly first by that count |",
        "|---|---|---|---|---|---|",
    ]
    for split, key in (("all", "gate"), ("test", "gate_test")):
        for s, g in payload[key].items():
            lines.append(
                f"| {s} | {split} | {100 * g['root_has_new_err']:.0f}% | {g['root_new_err_mean']:.1f} | "
                f"{g['others_new_err_mean']:.2f} | {100 * g['root_ranked_first']:.0f}% |"
            )
    return "\n".join(lines)


def nonllm_table(rows: Sequence[dict[str, Any]]) -> list[str]:
    by: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["model"] != "none":
            continue
        split = "test" if is_test(r) else "dev"
        by[r["condition"]][f"{split}/{r['system']}"].append(float(r["top1"]))
        by[r["condition"]][f"{split}/all"].append(float(r["top1"]))
    cols = [
        f"{sp}/{s}"
        for sp in ("dev", "test")
        for s in ("RE3-SS", "RE3-OB", "RE3-TT", "all")
    ]
    lines = [
        "| system | " + " | ".join(c.replace("RE3-", "") for c in cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for cond in LABELS:
        if cond in by:
            cells = [
                f"{100 * np.mean(by[cond][c]):.0f}" if by[cond].get(c) else "-"
                for c in cols
            ]
            lines.append(f"| {LABELS[cond]} | " + " | ".join(cells) + " |")
    return lines


def build_report(
    rows_in: Sequence[dict[str, Any]],
    gate: Optional[dict[str, Any]] = None,
    analysis: Optional[str] = None,
    note: Optional[str] = None,
) -> tuple[str, dict[str, Any]]:
    rows = dedupe(rows_in)
    rules, rule_table = pick_rules(rows)
    sys_ = Systems(rows, rules)
    verdict = g4d_verdicts(sys_)
    payload: dict[str, Any] = {
        "verdict": verdict,
        "rules": rule_table,
        "single_vs_team": single_vs_team(sys_),
        "ablations": ablation_rows(sys_),
        "per_system": per_system_rows(sys_),
        "gate": gate,
    }
    lines = ["# R10 step 4: LLM log specialists on RCAEval RE3 (goal G4d)", ""]
    if note:
        lines += [note, ""]
    lines += [f"**G4d verdict: {verdict['overall']}**", ""]
    ref = verdict.get("reference")
    if ref:
        lines += [
            f"Best non-LLM system on dev: {ref['dev_best']} (test top-1 {100 * ref['dev_best_test_top1']:.1f}); "
            f"best on the test split itself: {ref['test_best']} ({100 * ref['test_best_test_top1']:.1f}).",
            "",
        ]
    if gate:
        lines += [gate_text(gate), ""]
    lines += ["## Non-LLM systems (root-cause top-1, %)", ""]
    lines += nonllm_table(rows) + [""]
    lines += ["## Main results (test split, root-cause top-1 and top-3, %)", ""]
    lines += [
        "| system | n | top-1 [95% CI] | top-3 | tokens | bytes |",
        "|---|---|---|---|---|---|",
    ]
    for key in sys_.main_keys():
        recs = sys_.by_key[key]
        lines.append(
            f"| {LABELS[key[1]]} ({sys_.name(key)}) | {len(recs)} | {rp._cell([r['top1'] for r in recs])} | "
            f"{100 * np.mean([r['top3'] for r in recs]):.1f} | {np.mean([r.get('tokens', 0) for r in recs]):,.0f} | "
            f"{np.mean([(r.get('extra') or {}).get('bytes', 0) for r in recs]):,.0f} |"
        )
    lines.append("")
    if verdict["teams"]:
        lines += ["## G4d comparisons (paired, test split)", ""]
        lines += [
            "| LLM team | rule | top-1 | vs dev-best non-LLM [95% CI] | p (Holm) | vs test-best non-LLM [95% CI] | verdict |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, t in verdict["teams"].items():
            d, s = t["vs_dev_best"], t["vs_test_best"]
            lines.append(
                f"| {name} | {t['rule'] or '-'} | {100 * t['top1']:.1f} | {_p(d['diff'])} [{_p(d['low'])}, {_p(d['high'])}] | "
                f"{_fmt_p(t['p_holm'])} | {_p(s['diff'])} [{_p(s['low'])}, {_p(s['high'])}] | {t['verdict']} |"
            )
        lines.append("")
    if payload["single_vs_team"]:
        lines += ["## Team against the best single LLM", ""]
        lines += [
            "| team | best single | single top-1 | team - single [95% CI] |",
            "|---|---|---|---|",
        ]
        for r in payload["single_vs_team"]:
            lines.append(
                f"| {r['team']} | {r['single']} | {100 * r['single_top1']:.1f} | "
                f"{_p(r['diff'])} [{_p(r['low'])}, {_p(r['high'])}] |"
            )
        lines.append("")
    if payload["ablations"]:
        lines += ["## Ablations (test split)", ""]
        lines += [
            "| system | ablation | intact top-1 | ablated top-1 | change [95% CI] |",
            "|---|---|---|---|---|",
        ]
        for r in payload["ablations"]:
            lines.append(
                f"| {r['system']} | {r['ablation']} | {100 * r['intact_top1']:.1f} | {100 * r['top1']:.1f} | "
                f"{_p(r['diff'])} [{_p(r['low'])}, {_p(r['high'])}] |"
            )
        lines.append("")
    if payload["per_system"]:
        lines += ["## Top-1 per RE3 system (test split, %)", ""]
        cols = sorted(
            {k for r in payload["per_system"] for k in r if k.startswith("RE3-")}
        )
        lines += ["| system | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for r in payload["per_system"]:
            lines.append(
                f"| {r['system']} | "
                + " | ".join(f"{100 * r[c]:.0f}" if c in r else "-" for c in cols)
                + " |"
            )
        lines.append("")
    if rule_table:
        lines += ["## Rule-centre weighting chosen on dev", ""]
        lines += [
            "| log specialist | n dev | dev top-1 by weighting | chosen |",
            "|---|---|---|---|",
        ]
        for model, t in rule_table.items():
            lines.append(
                f"| {short(model)} | {t['n_dev']} | "
                + ", ".join(f"{k} {100 * v:.0f}" for k, v in t["top1"].items())
                + f" | {t['chosen']} |"
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
