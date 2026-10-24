"""R7a evaluation: run a translator over the evaluation set and compute the G3/G6 metrics.

``evaluate_forward`` translates every item, then runs both the translated spec and the
gold spec through the command centre (``r7a_pipeline.execute``) to get the task scores.
``evaluate_reverse`` summarises each executed trace, checks the summary's facts against
the GoalSpec and outcome, and re-translates the summary for the round trip.
``forward_metrics`` / ``reverse_metrics`` aggregate; ``render_markdown`` writes the report
and ``verdict`` compares it with the G3 and G6 targets in ``docs/goals.md``.
"""

import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any, Optional

import numpy as np

from ..evaluation.stats import mean_ci
from ..protocol.goalspec import SpecRegistry
from ..translator import (
    OK,
    ReverseTranslator,
    TranslationResult,
    exact_match,
    fact_fidelity,
    field_matches,
    percentile,
    refusal_scores,
    wilson,
)
from .r7a_domains import Domain
from .r7a_evalset import EvalItem
from .r7a_pipeline import execute

# G3 / G6 targets from docs/goals.md
G3_EXACT_PASS, G3_EXACT_KILL = 0.95, 0.85
G3_ROUND_TRIP_PASS, G3_ROUND_TRIP_KILL = 0.90, 0.70
G3_WARM_P95_S = 2.0
G3_VRAM_GB = 12.0
G6_FIDELITY_PASS, G6_FIDELITY_KILL = 0.90, 0.75


def _num(x: float) -> Optional[float]:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def select_items(items: Sequence[EvalItem], n: int, seed: int = 0) -> list[EvalItem]:
    """A stratified subsample of about ``n`` items (every group keeps its share)."""
    if n >= len(items):
        return list(items)
    rng = np.random.default_rng(seed)
    groups: dict[tuple[str, str, bool], list[EvalItem]] = defaultdict(list)
    for it in items:
        groups[(it.domain, it.split, it.invalid)].append(it)
    out: list[EvalItem] = []
    for key in sorted(groups):
        members = groups[key]
        take = max(1, round(n * len(members) / len(items)))
        idx = rng.permutation(len(members))[:take]
        out.extend(members[i] for i in sorted(idx))
    return out


# ---------------------------------------------------------------------- forward


def evaluate_forward(
    translate: Callable[[EvalItem], TranslationResult],
    items: Sequence[EvalItem],
    domains: dict[str, Domain],
    registry: SpecRegistry,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> list[dict[str, Any]]:
    """Per-item records: translation outcome, exactness, and NL-driven vs gold-driven score.

    A valid item whose translation failed scores 0 on every seed (the mission was not
    run). Gold-invalid items have no score.
    """
    records = []
    for k, item in enumerate(items):
        domain = domains[item.domain]
        schema = registry.get(domain.spec_name).params_schema
        res = translate(item)
        pred = res.params
        rec: dict[str, Any] = {
            "id": item.id,
            "domain": item.domain,
            "split": item.split,
            "style": item.style,
            "text": item.text,
            "gold": item.gold,
            "status": res.status,
            "refused": res.status != OK,
            "pred": pred,
            "reason": res.reason,
            "attempts": res.attempts,
            "latency_s": res.latency_s,
            "call_latencies": list(res.call_latencies),
            "n_prompt_tokens": res.n_prompt_tokens,
        }
        if item.gold is not None:
            rec["exact"] = exact_match(pred, item.gold, schema)
            rec["fields"] = field_matches(pred, item.gold, schema)
            gold_spec = registry.make(domain.spec_name, item.gold)
            gold_run = execute(domain, registry, gold_spec, item.text)
            rec["gold_state"] = gold_run.state
            rec["gold_per_seed"] = gold_run.per_seed
            if res.spec is not None:
                nl_run = execute(domain, registry, res.spec, item.text)
                rec["nl_state"] = nl_run.state
                rec["nl_per_seed"] = nl_run.per_seed
                rec["nl_view"] = nl_run.view
            else:
                rec["nl_per_seed"] = [0.0] * len(gold_run.per_seed)
        records.append(rec)
        if progress:
            progress(k + 1, len(items), item.id)
    return records


def _exact_block(recs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in recs if r["gold"] is not None]
    hits = sum(r["exact"] for r in valid)
    p, lo, hi = wilson(hits, len(valid))
    fields: dict[str, list[bool]] = defaultdict(list)
    for r in valid:
        for name, ok in r["fields"].items():
            fields[name].append(ok)
    return {
        "n": len(valid),
        "exact": _num(p),
        "exact_ci": [_num(lo), _num(hi)],
        "field_accuracy": {k: float(np.mean(v)) for k, v in sorted(fields.items())},
    }


def _task_block(recs: Sequence[dict[str, Any]]) -> Optional[dict[str, Any]]:
    scored = [r for r in recs if r.get("gold_per_seed")]
    if not scored:
        return None
    nl = np.array([r["nl_per_seed"] for r in scored], dtype=float)
    gold = np.array([r["gold_per_seed"] for r in scored], dtype=float)
    nl_seed, gold_seed = nl.mean(axis=0), gold.mean(axis=0)
    diff = nl_seed - gold_seed
    d, hw = mean_ci(diff)
    nl_m, nl_hw = mean_ci(nl_seed)
    g_m, g_hw = mean_ci(gold_seed)
    return {
        "n_items": len(scored),
        "n_seeds": int(nl.shape[1]),
        "nl_mean": nl_m,
        "nl_ci": _num(nl_hw),
        "gold_mean": g_m,
        "gold_ci": _num(g_hw),
        "diff_mean": d,
        "diff_ci": _num(hw),
        "within_ci": bool(d >= 0 or (not math.isnan(hw) and abs(d) <= hw)),
        "failed_translations": int(sum(1 for r in scored if r["refused"])),
    }


def forward_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate forward records: exact match, per-field accuracy, task score, refusals, latency."""
    out: dict[str, Any] = {"n_items": len(records)}
    groups: dict[str, list[dict[str, Any]]] = {}
    for split in ("indist", "heldout"):
        groups[split] = [r for r in records if r["split"] == split]
        for domain in sorted({r["domain"] for r in records}):
            groups[f"{split}/{domain}"] = [
                r for r in groups[split] if r["domain"] == domain
            ]
    groups["all"] = list(records)
    out["exact"] = {k: _exact_block(v) for k, v in groups.items() if v}
    out["task"] = {}
    for split in ("indist", "heldout", "all"):
        pool = groups[split]
        for domain in ("squad", "plant"):
            block = _task_block([r for r in pool if r["domain"] == domain])
            if block is not None:
                out["task"][f"{split}/{domain}"] = block
    ref = refusal_scores((r["gold"] is None, r["refused"]) for r in records)
    out["refusal"] = {k: _num(v) for k, v in ref.items()}
    out["status_counts"] = dict(sorted(_count(r["status"] for r in records).items()))
    out["retried"] = int(sum(1 for r in records if r["attempts"] > 1))
    lat = [r["latency_s"] for r in records]
    out["latency"] = {
        "p50": _num(percentile(lat, 50)),
        "p95": _num(percentile(lat, 95)),
        "mean": float(np.mean(lat)) if lat else None,
    }
    return out


def _count(values: Any) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for v in values:
        out[v] += 1
    return dict(out)


# ---------------------------------------------------------------------- reverse


def evaluate_reverse(
    reverse: ReverseTranslator,
    records: Sequence[dict[str, Any]],
    domains: dict[str, Domain],
    registry: SpecRegistry,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> list[dict[str, Any]]:
    """Summarise each executed trace and re-translate the summary.

    Works on the forward records that have an NL-driven trace view (valid gold, ok
    translation). The round trip compares the reconstructed spec with the *forward* spec.
    """
    todo = [r for r in records if r.get("nl_view") is not None]
    out = []
    for k, rec in enumerate(todo):
        domain = domains[rec["domain"]]
        schema = registry.get(domain.spec_name).params_schema
        view = rec["nl_view"]
        trip = reverse.round_trip(view, domain.summary_pack, domain.report_pack)
        fid = fact_fidelity(trip.summary.text, domain.facts(view))
        recon = trip.reconstructed
        out.append(
            {
                "id": rec["id"],
                "domain": rec["domain"],
                "split": rec["split"],
                "summary": trip.summary.text,
                "summary_latency_s": trip.summary.latency_s,
                "facts": fid["facts"],
                "fidelity": _num(fid["fidelity"]),
                "all_facts": fid["all"],
                "recon_status": None if recon is None else recon.status,
                "recon_params": None if recon is None else recon.params,
                "recon_latency_s": 0.0 if recon is None else recon.latency_s,
                "round_trip": trip.matches(view.goal, schema),
            }
        )
        if progress:
            progress(k + 1, len(todo), rec["id"])
    return out


def reverse_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate round-trip and summary-fidelity rows."""
    out: dict[str, Any] = {}
    groups: dict[str, list[dict[str, Any]]] = {"all": list(rows)}
    for split in ("indist", "heldout"):
        groups[split] = [r for r in rows if r["split"] == split]
    for domain in sorted({r["domain"] for r in rows}):
        groups[domain] = [r for r in rows if r["domain"] == domain]
    for name, sub in groups.items():
        if not sub:
            continue
        p, lo, hi = wilson(sum(r["round_trip"] for r in sub), len(sub))
        facts: dict[str, list[bool]] = defaultdict(list)
        for r in sub:
            for fname, ok in r["facts"].items():
                facts[fname].append(ok)
        all_p, all_lo, all_hi = wilson(sum(r["all_facts"] for r in sub), len(sub))
        out[name] = {
            "n": len(sub),
            "round_trip": _num(p),
            "round_trip_ci": [_num(lo), _num(hi)],
            "field_fidelity": float(np.mean([r["fidelity"] for r in sub])),
            "all_facts_rate": _num(all_p),
            "all_facts_ci": [_num(all_lo), _num(all_hi)],
            "per_fact": {k: float(np.mean(v)) for k, v in sorted(facts.items())},
        }
    lat = [r["summary_latency_s"] for r in rows]
    out["summary_latency"] = {
        "p50": _num(percentile(lat, 50)),
        "p95": _num(percentile(lat, 95)),
    }
    return out


# ---------------------------------------------------------------------- verdict


def verdict(
    forward: dict[str, Any],
    reverse: Optional[dict[str, Any]],
    warm_p95: Optional[float],
    vram_gb: Optional[float],
    smoke: bool,
) -> dict[str, Any]:
    """Compare a primary translator's metrics with the G3 and G6 targets."""
    rows = []

    def row(
        name: str,
        target: str,
        value: Optional[float],
        passed: Optional[bool],
        killed: Optional[bool],
        shown: str,
    ) -> None:
        rows.append(
            {
                "criterion": name,
                "target": target,
                "value": shown,
                "pass": passed,
                "kill": killed,
            }
        )

    held = forward["exact"].get("heldout", {})
    ex = held.get("exact")
    row(
        "G3 exact match, held-out valid items",
        f">= {G3_EXACT_PASS:.0%} (kill < {G3_EXACT_KILL:.0%})",
        ex,
        None if ex is None else ex >= G3_EXACT_PASS,
        None if ex is None else ex < G3_EXACT_KILL,
        (
            "n/a"
            if ex is None
            else f"{ex:.1%} (n={held['n']}, 95% CI {held['exact_ci'][0]:.1%}-{held['exact_ci'][1]:.1%})"
        ),
    )
    tasks = [t for k, t in forward["task"].items() if k.startswith("all/")]
    if tasks:
        ok = all(t["within_ci"] for t in tasks)
        shown = "; ".join(
            (
                f"{k.split('/')[1]}: NL {t['nl_mean']:.2f} vs gold {t['gold_mean']:.2f}, diff {t['diff_mean']:+.2f} ± {t['diff_ci']:.2f}"
                if t["diff_ci"] is not None
                else f"{k.split('/')[1]}: NL {t['nl_mean']:.2f} vs gold {t['gold_mean']:.2f}"
            )
            for k, t in forward["task"].items()
            if k.startswith("all/")
        )
        row(
            "G3 task score within CI of gold-spec run",
            "paired diff >= 0 or inside its CI, both domains",
            None,
            ok,
            None,
            shown,
        )
    if reverse and "all" in reverse:
        rt = reverse["all"]["round_trip"]
        row(
            "G3 round trip (reverse summary re-translated)",
            f">= {G3_ROUND_TRIP_PASS:.0%} (kill < {G3_ROUND_TRIP_KILL:.0%})",
            rt,
            rt >= G3_ROUND_TRIP_PASS,
            rt < G3_ROUND_TRIP_KILL,
            f"{rt:.1%} (n={reverse['all']['n']}, 95% CI {reverse['all']['round_trip_ci'][0]:.1%}-{reverse['all']['round_trip_ci'][1]:.1%})",
        )
        fid = reverse["all"]["field_fidelity"]
        row(
            "G6 summary field fidelity",
            f">= {G6_FIDELITY_PASS:.0%} (kill < {G6_FIDELITY_KILL:.0%})",
            fid,
            fid >= G6_FIDELITY_PASS,
            fid < G6_FIDELITY_KILL,
            f"{fid:.1%} of facts (all facts in {reverse['all']['all_facts_rate']:.1%} of summaries)",
        )
    if warm_p95 is not None:
        row(
            "G3 warm translator p95",
            f"< {G3_WARM_P95_S:.0f} s",
            warm_p95,
            warm_p95 < G3_WARM_P95_S,
            None,
            f"{warm_p95:.2f} s",
        )
    if vram_gb is not None:
        row(
            "G3 fits in 12 GB",
            f"<= {G3_VRAM_GB:.0f} GB",
            vram_gb,
            vram_gb <= G3_VRAM_GB,
            None,
            f"{vram_gb:.1f} GB peak",
        )
    killed = any(r["kill"] for r in rows)
    passed = all(r["pass"] for r in rows if r["pass"] is not None)
    if smoke:
        label = "SMOKE RUN, no verdict"
    elif killed:
        label = "KILL criterion hit"
    elif passed:
        label = "PASS"
    else:
        label = "NOT PASSED (no kill criterion hit)"
    return {"rows": rows, "label": label}
