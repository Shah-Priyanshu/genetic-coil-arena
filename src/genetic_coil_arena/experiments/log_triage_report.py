"""R10 report: metrics, paired tests, message ablations and the G4 verdict.

Reads the per-incident records written by ``scripts/r10_log_triage.py`` and builds the
Markdown and JSON of ``docs/results/r10_log_triage``. Everything is per incident:
top-1 and top-3 root-cause service, fault-type accuracy, tokens. Intervals are percentile
bootstraps over incidents (95%); comparisons between conditions are paired on the same
incidents (paired bootstrap of the difference, Wilcoxon signed-rank, Holm correction over
the comparisons of one model).

**G4 rule** (``docs/goals.md``): the team passes if its top-1 beats the best single-model
condition at the same centre context by at least 5 points and the 95% CI of the paired
difference excludes 0. It is killed if the single model is within that CI (the interval
contains 0, or lies below it). Anything else is inconclusive. "Best single model" is the
best of ``single_head``, ``single_tail`` and ``single_selected`` of the same model,
chosen on the same incidents, which favours the single model.
"""

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..evaluation.stats import (
    holm_correction,
    paired_bootstrap_ci,
    stratified_bootstrap_ci,
    wilcoxon_signed_rank,
)

ANALYSIS_START = "<!-- analysis:start -->"
ANALYSIS_END = "<!-- analysis:end -->"
MARGIN = 0.05
LLM_SINGLES = ("single_head", "single_tail", "single_selected")
LABELS = {
    "heur_baro_logs": "(c) BARO on log counts (non-LLM)",
    "heur_surprise_logs": "(c) template-surprise ranking (non-LLM)",
    "heur_baro_metrics": "(c) BARO on service metrics (non-LLM)",
    "single_head": "(a) single model, first lines (truncation)",
    "single_tail": "(a) single model, last lines (truncation)",
    "single_selected": "(b) single model, strong selection",
    "oracle_selected": "(f) single model, long context, same selection",
    "team_typed": "(d) team, typed JSON messages",
    "team_free": "(e) team, free-text messages",
    "team_typed_heur": "(d') team, typed messages computed by code",
}
ORDER = list(LABELS)


def carried_analysis(path: Path) -> Optional[str]:
    """The hand-written Analysis section of an existing report, to carry over."""
    path = Path(path)
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    if ANALYSIS_START in text and ANALYSIS_END in text:
        return text.split(ANALYSIS_START, 1)[1].split(ANALYSIS_END, 1)[0].strip()
    return None


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def _ci(scores: Sequence[float]) -> tuple[float, float, float]:
    arr = np.asarray(scores, dtype=float)
    if arr.size < 2:
        v = float(arr.mean()) if arr.size else float("nan")
        return v, float("nan"), float("nan")
    return stratified_bootstrap_ci(arr, statistic=np.mean, n_boot=4000, seed=0)


def _cell(scores: Sequence[float]) -> str:
    p, lo, hi = _ci(scores)
    if not np.isfinite(lo):
        return _pct(p)
    return f"{_pct(p)} [{_pct(lo)}, {_pct(hi)}]"


def dedupe(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Last record wins per (model, condition, incident)."""
    seen: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in records:
        if r.get("kind") == "overflow":
            continue
        seen[(r["model"], r["condition"], r["incident"])] = r
    return list(seen.values())


def vectors(
    rows: Sequence[dict[str, Any]], model: str, condition: str, field: str = "top1"
) -> dict[str, float]:
    """incident -> value for one condition ('none' rows are shared by every model)."""
    key = "none" if condition.startswith("heur_") else model
    return {
        r["incident"]: float(r[field])
        for r in rows
        if r["model"] == key and r["condition"] == condition
    }


def aligned(*vecs: dict[str, float]) -> tuple[list[str], list[np.ndarray]]:
    ids = sorted(set.intersection(*(set(v) for v in vecs)))
    return ids, [np.array([v[i] for i in ids]) for v in vecs]


def g4_verdict(
    team: dict[str, float], singles: dict[str, dict[str, float]], margin: float = MARGIN
) -> dict[str, Any]:
    """Apply the G4 rule to one model (see the module docstring)."""
    if not team or not singles:
        return {
            "verdict": "NOT RUN",
            "reason": "team or single-model records are missing",
        }
    ids = sorted(set(team).intersection(*(set(v) for v in singles.values())))
    if len(ids) < 2:
        return {"verdict": "NOT RUN", "reason": "fewer than 2 shared incidents"}
    t = np.array([team[i] for i in ids])
    means = {k: float(np.mean([v[i] for i in ids])) for k, v in singles.items()}
    best = max(means, key=lambda k: means[k])
    b = np.array([singles[best][i] for i in ids])
    diff, lo, hi = paired_bootstrap_ci(t, b, n_boot=10000, seed=0)
    if lo > 0 and diff >= margin:
        verdict, why = "PASS", (
            f"team beats {best} by {100 * diff:.1f} points, CI excludes 0 and the gain is "
            f"at least {100 * margin:.0f} points"
        )
    elif lo > 0:
        verdict, why = "INCONCLUSIVE", (
            f"team beats {best} by {100 * diff:.1f} points (CI excludes 0) but by less than "
            f"{100 * margin:.0f} points"
        )
    elif hi < 0:
        verdict, why = (
            "KILL",
            f"team is worse than {best} (CI of the difference below 0)",
        )
    else:
        verdict, why = "KILL", (
            f"{best} is within the CI of the team (difference {100 * diff:+.1f} points, CI "
            f"[{100 * lo:.1f}, {100 * hi:.1f}] contains 0)"
        )
    return {
        "verdict": verdict,
        "reason": why,
        "best_single": best,
        "team_top1": float(t.mean()),
        "best_single_top1": float(b.mean()),
        "diff": diff,
        "ci": [lo, hi],
        "n": len(ids),
        "single_top1": means,
    }


def comparisons(
    rows: Sequence[dict[str, Any]], model: str, pairs: Sequence[tuple[str, str]]
) -> list[dict[str, Any]]:
    out = []
    for a, b in pairs:
        va, vb = vectors(rows, model, a), vectors(rows, model, b)
        if not va or not vb:
            continue
        ids, (x, y) = aligned(va, vb)
        if len(ids) < 2:
            continue
        diff, lo, hi = paired_bootstrap_ci(x, y, n_boot=10000, seed=1)
        _, p = wilcoxon_signed_rank(x, y)
        out.append(
            {"a": a, "b": b, "n": len(ids), "diff": diff, "low": lo, "high": hi, "p": p}
        )
    if out:
        for row, adj in zip(out, holm_correction([r["p"] for r in out])):
            row["p_holm"] = float(adj)
    return out


def _condition_rows(
    rows: Sequence[dict[str, Any]], model: str, services: dict[str, float]
) -> list[dict[str, Any]]:
    table = []
    for cond in ORDER:
        key = "none" if cond.startswith("heur_") else model
        sel = [r for r in rows if r["model"] == key and r["condition"] == cond]
        if not sel:
            continue
        table.append(
            {
                "condition": cond,
                "n": len(sel),
                "top1": [r["top1"] for r in sel],
                "top3": [r["top3"] for r in sel],
                "fault": [r["fault_ok"] for r in sel],
                "tokens": float(np.mean([r["tokens"] for r in sel])),
                "centre_tokens": float(np.mean([r["centre_tokens"] for r in sel])),
                "latency": float(np.mean([r["latency_s"] for r in sel])),
                "valid": float(np.mean([r.get("valid", 1) for r in sel])),
                "valid_messages": (
                    float(
                        np.mean([r["extra"].get("valid_messages", np.nan) for r in sel])
                    )
                    if cond.startswith("team_") and not cond.endswith("_heur")
                    else None
                ),
            }
        )
    return table


def build_report(
    records: Sequence[dict[str, Any]],
    analysis: Optional[str] = None,
    note: Optional[str] = None,
) -> tuple[str, dict[str, Any]]:
    overflow = [r for r in records if r.get("kind") == "overflow"]
    if overflow:  # token counts from one tokenizer only (the most common one)
        names = [o["counter"] for o in overflow]
        main = max(set(names), key=names.count)
        overflow = [o for o in overflow if o["counter"] == main]
    rows = dedupe(records)
    models = sorted({r["model"] for r in rows if r["model"] != "none"})
    chance = {}
    for r in rows:
        chance[r["incident"]] = r["n_services"]
    payload: dict[str, Any] = {
        "models": models,
        "n_records": len(rows),
        "verdicts": {},
        "tables": {},
    }
    lines: list[str] = ["# R10: log triage with forced specialisation (goal G4)", ""]
    if note:
        lines += [f"> **{note}**", ""]
        payload["note"] = note

    verdict_lines = []
    for model in models:
        singles = {c: vectors(rows, model, c) for c in LLM_SINGLES}
        singles = {k: v for k, v in singles.items() if v}
        v = g4_verdict(vectors(rows, model, "team_typed"), singles)
        payload["verdicts"][model] = v
        n_inc = f" ({v['n']} incidents)" if "n" in v else ""
        verdict_lines.append(f"- **{model}: {v['verdict']}.** {v['reason']}{n_inc}.")
    lines += [
        "Step 1 of R10: zero-shot models, no training. A team of one specialist per "
        "service plus a centre is compared with single models at the same centre context.",
        "",
        "## Verdict",
        "",
        "Rule (`docs/goals.md`, G4): pass if the typed-message team (d) beats the best "
        "single model (a, b) at the same centre context by at least 5 points of top-1 "
        "root-cause accuracy with the 95% CI of the paired difference excluding 0; kill "
        "if the single model is within that CI.",
        "",
        *(verdict_lines or ["No model has results yet."]),
        "",
    ]
    if analysis:
        lines += ["## Analysis", "", ANALYSIS_START, analysis, ANALYSIS_END, ""]
    else:
        lines += [
            "## Analysis",
            "",
            ANALYSIS_START,
            "(to be written after the full run)",
            ANALYSIS_END,
            "",
        ]

    # ---- overflow
    if overflow:
        full = np.array([o["ratio_full"] for o in overflow])
        uniq = np.array([o["ratio_unique"] for o in overflow])
        by_sys: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for o in overflow:
            by_sys[o["system"]].append(o)
        payload["overflow"] = {
            "n": len(overflow),
            "ratio_full_median": float(np.median(full)),
            "ratio_full_min": float(full.min()),
            "ratio_unique_median": float(np.median(uniq)),
            "ctx": overflow[0]["ctx"],
            "counter": overflow[0]["counter"],
        }
        lines += [
            "## Overflow: the input does not fit",
            "",
            f"Tokens of every templated log line of an incident, over the centre context of "
            f"{overflow[0]['ctx']} tokens ({overflow[0]['counter']} token counts, "
            f"{len(overflow)} incidents). 'Unique' keeps one line per distinct (service, template) "
            "pair, the most compact text form without counts.",
            "",
            "| System | Incidents | Lines (median) | Services | Full text / context (median, min) | Unique templates / context (median) | Largest single service (lines, median) |",
            "|---|---|---|---|---|---|---|",
        ]
        for system, os_ in sorted(by_sys.items()):
            lines.append(
                f"| {system} | {len(os_)} | {int(np.median([o['n_lines'] for o in os_])):,} | "
                f"{int(np.median([o['n_services'] for o in os_]))} | "
                f"{np.median([o['ratio_full'] for o in os_]):,.0f}x, {min(o['ratio_full'] for o in os_):,.0f}x | "
                f"{np.median([o['ratio_unique'] for o in os_]):,.1f}x | "
                f"{int(np.median([o['largest_service_lines'] for o in os_])):,} |"
            )
        lines += [
            "",
            f"All incidents together: median **{np.median(full):,.0f}x** the context "
            f"(minimum {full.min():,.0f}x); even the unique templates alone are "
            f"{np.median(uniq):,.1f}x (minimum {uniq.min():,.1f}x). One service's own stream "
            "is also far over the budget, so each specialist needs a selection step "
            "too (it reads at most its own `spec_ctx_tokens`).",
            "",
        ]

    # ---- per model tables
    for model in models:
        mrows = [r for r in rows if r["model"] in (model, "none")]
        table = _condition_rows(mrows, model, chance)
        payload["tables"][model] = {
            t["condition"]: {
                "n": t["n"],
                "top1": _ci(t["top1"]),
                "top3": _ci(t["top3"]),
                "fault": _ci(t["fault"]),
                "tokens": t["tokens"],
                "latency_s": t["latency"],
            }
            for t in table
        }
        inc_ids = {r["incident"] for r in mrows if r["model"] == model}
        mean_chance = (
            float(np.mean([1 / chance[i] for i in inc_ids])) if inc_ids else 0.0
        )
        lines += [
            f"## Results: {model}",
            "",
            f"Per incident, percent with 95% bootstrap CI over incidents. Chance for top-1 is "
            f"{_pct(mean_chance)} (1 over the number of services), for fault type 16.7. "
            "Tokens are the model tokens read and written per incident across all calls; "
            "'centre prompt' is the largest single prompt.",
            "",
            "| Condition | n | Top-1 | Top-3 | Fault type | Tokens / incident | Centre prompt | Valid | Latency (s) |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for t in table:
            valid = t["valid"] if t["valid_messages"] is None else t["valid_messages"]
            lines.append(
                f"| {LABELS[t['condition']]} | {t['n']} | {_cell(t['top1'])} | {_cell(t['top3'])} | "
                f"{_cell(t['fault'])} | {t['tokens']:,.0f} | {t['centre_tokens']:,.0f} | "
                f"{_pct(valid)} | {t['latency']:.1f} |"
            )
        lines += [
            "",
            "'Valid' is the share of decisions the model answered in a usable form; for "
            "LLM-specialist teams it is the share of specialist messages that passed the "
            "schema (typed) or were non-empty (free text).",
            "",
        ]

        pairs = [
            ("team_typed", "single_head"),
            ("team_typed", "single_tail"),
            ("team_typed", "single_selected"),
            ("team_typed", "oracle_selected"),
            ("team_typed", "heur_baro_metrics"),
            ("team_free", "single_selected"),
            ("team_free", "team_typed"),
            ("team_typed_heur", "team_typed"),
            ("team_typed_heur", "single_selected"),
            ("single_selected", "heur_baro_metrics"),
        ]
        comps = comparisons(mrows, model, pairs)
        if comps:
            payload.setdefault("comparisons", {})[model] = comps
            lines += [
                f"### Paired comparisons: {model}",
                "",
                "Top-1 difference a - b in points on the incidents both have, paired bootstrap "
                "95% CI, Wilcoxon signed-rank p, Holm-adjusted over this table.",
                "",
                "| a | b | n | Difference | 95% CI | p | p (Holm) |",
                "|---|---|---|---|---|---|---|",
            ]
            for c in comps:
                lines.append(
                    f"| {c['a']} | {c['b']} | {c['n']} | {100 * c['diff']:+.1f} | "
                    f"[{100 * c['low']:+.1f}, {100 * c['high']:+.1f}] | {c['p']:.3g} | "
                    f"{c['p_holm']:.3g} |"
                )
            lines.append("")

        v = payload["verdicts"][model]
        if "best_single" in v:
            lines += [
                f"### G4 verdict: {model}",
                "",
                f"**{v['verdict']}.** {v['reason']}. Team (d) top-1 {_pct(v['team_top1'])}, best "
                f"single model `{v['best_single']}` {_pct(v['best_single_top1'])}, difference "
                f"{100 * v['diff']:+.1f} points (95% CI [{100 * v['ci'][0]:+.1f}, "
                f"{100 * v['ci'][1]:+.1f}], {v['n']} incidents). Single-model top-1 on these "
                "incidents: "
                + ", ".join(f"{k} {_pct(x)}" for k, x in v["single_top1"].items())
                + ".",
                "",
            ]

        # ablations
        abl = []
        for base in ("team_typed", "team_free"):
            vb = vectors(mrows, model, base)
            if not vb:
                continue
            for mode in ("drop_root", "shuffle_root", "drop_other", "shuffle_other"):
                va = vectors(mrows, model, f"{base}:{mode}")
                if not va:
                    continue
                ids, (x, y) = aligned(va, vb)
                if len(ids) < 2:
                    continue
                diff, lo, hi = paired_bootstrap_ci(x, y, n_boot=5000, seed=2)
                abl.append(
                    {
                        "team": base,
                        "mode": mode,
                        "n": len(ids),
                        "intact": float(y.mean()),
                        "ablated": float(x.mean()),
                        "diff": diff,
                        "low": lo,
                        "high": hi,
                    }
                )
        if abl:
            payload.setdefault("ablations", {})[model] = abl
            lines += [
                f"### Message ablations: {model}",
                "",
                "One specialist's message is dropped (replaced by an empty report) or replaced "
                "by the same service's message from another incident (shuffle). 'root' is the "
                "true root-cause service's specialist, 'other' a random other one. A centre that "
                "uses the messages loses accuracy when the root's message is ablated and little "
                "when another's is.",
                "",
                "| Team | Ablation | n | Intact top-1 | Ablated top-1 | Change | 95% CI |",
                "|---|---|---|---|---|---|---|",
            ]
            for a in abl:
                lines.append(
                    f"| {a['team']} | {a['mode']} | {a['n']} | {_pct(a['intact'])} | "
                    f"{_pct(a['ablated'])} | {100 * a['diff']:+.1f} | "
                    f"[{100 * a['low']:+.1f}, {100 * a['high']:+.1f}] |"
                )
            lines.append("")

        # breakdowns
        main = [
            c
            for c in ("single_selected", "team_typed", "team_free", "oracle_selected")
            if vectors(mrows, model, c)
        ]
        if main:
            lines += [f"### Top-1 by system and fault type: {model}", ""]
            meta = {r["incident"]: r for r in mrows}
            groups: dict[str, list[str]] = defaultdict(list)
            for i, r in meta.items():
                groups[f"system {r['system']}"].append(i)
                groups[f"fault {r['fault']}"].append(i)
            lines += [
                "| Group | n | " + " | ".join(main) + " |",
                "|---|---|" + "---|" * len(main),
            ]
            for g in sorted(groups):
                cells = []
                for c in main:
                    vec = vectors(mrows, model, c)
                    ids = [i for i in groups[g] if i in vec]
                    cells.append(
                        _pct(float(np.mean([vec[i] for i in ids]))) if ids else "-"
                    )
                lines.append(f"| {g} | {len(groups[g])} | " + " | ".join(cells) + " |")
            lines.append("")

    lines += [
        "## Setup",
        "",
        "- **Data:** RCAEval RE2 (Pham et al., WWW 2025 Companion, arXiv:2412.17015; Zenodo "
        "10.5281/zenodo.14590730, CC-BY-4.0), Sock Shop (SS) and Online Boutique (OB). Each "
        "incident is one fault injection (a root-cause service and a fault type among cpu, mem, "
        "disk, delay, loss, socket) with per-service logs and service metrics; the alert is "
        "the injection time. Repetition 1 of every case is the dev split (prompt design, "
        "smoke runs); the reported numbers use the test split (repetitions 2 and 3). "
        "Incidents whose root service appears in neither logs nor metrics are dropped.",
        "- **Design change forced by the data:** the task was meant to be log triage, but the "
        "non-LLM log baselines (BARO on log counts, template surprise) are near chance on the "
        "dev split, so logs alone cannot identify the root cause in RE2. Each service is "
        "therefore a source of its logs **and its metrics** (`simple_metrics.csv`: cpu, mem, "
        "socket, diskio, workload, latency, error, summarised as robust deviations from the "
        "pre-alert range). Candidates are all services that appear in either, so infrastructure "
        "services (databases, queue) are candidates too. The metric summary is available to "
        "every condition, so the strong single-model selection (b) sees it as well. Logs-only "
        "results for the non-LLM baselines are in the table (rows marked logs).",
        "- **Conditions:** see `experiments/log_triage.py`. All centre prompts are at most "
        "4,096 tokens, instructions and options included; specialists read at most 3,072 tokens "
        "of their own stream and send at most 128 tokens; the long-context oracle (f) reads "
        "32,768 tokens (8,192 for the Ollama model).",
        "- **Scoring:** the root-cause service is ranked by next-token log-probabilities over "
        "option letters (Qwen3), or by a structured JSON reply (Ollama); fault type is the best "
        "letter given the chosen service.",
        "- **Not done in step 1:** no LoRA training of specialists or centre (step 2), no "
        "learned compression of messages, no KV handoff (G5).",
        "",
    ]
    return "\n".join(lines), payload
