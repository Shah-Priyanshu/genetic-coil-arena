"""R11 report: the G8 metrics, verdict and markdown from per-item records.

Everything is recomputed from the JSON-safe item records (``process_incident_item``) and the
routing records of the R7a sealed items (``route_only``), so a partial or resumed run reports
the same way. The G8 targets (``docs/goals.md``, fixed before any model run):

1. exact match on the sealed spec items >= 90% (kill < 85%); domain routing >= 98% on the
   mixed set (R11 sealed spec and clarify items plus R7a sealed spec and clarify items);
2. natural-language task score (joint accuracy of the named incident, a failed translation
   scores 0) non-inferior to the gold-spec run: lower bound of the paired 95% bootstrap CI of
   NL minus gold > -5 points (kill: upper bound < 0);
3. round trip >= 90% of the executed items;
4. summary field fidelity >= 90% of facts (kill < 75%);
5. warm end-to-end p95 <= 10 s (first item dropped) and peak VRAM <= 12 GB.
"""

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np

from ..translator import percentile, wilson

EXACT_PASS, EXACT_KILL = 0.90, 0.85
ROUTING_PASS = 0.98
NONINF_MARGIN = 0.05
ROUND_TRIP_PASS = 0.90
FIDELITY_PASS, FIDELITY_KILL = 0.90, 0.75
P95_S = 10.0
VRAM_GB = 12.0
BOOT = 10_000


def _f(x: Any) -> Optional[float]:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def rate(hits: int, n: int) -> dict[str, Any]:
    """``{n, hits, p, lo, hi}`` with a Wilson 95% interval."""
    p, lo, hi = wilson(hits, n)
    return {"n": n, "hits": hits, "p": _f(p), "lo": _f(lo), "hi": _f(hi)}


def paired_bootstrap(
    diff: Sequence[float], boot: int = BOOT, seed: int = 0
) -> dict[str, Any]:
    """Mean of paired differences with a 95% percentile bootstrap interval over items."""
    d = np.asarray(diff, dtype=float)
    if d.size == 0:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(boot, d.size))].mean(axis=1)
    return {
        "n": int(d.size),
        "mean": float(d.mean()),
        "lo": float(np.percentile(means, 2.5)),
        "hi": float(np.percentile(means, 97.5)),
    }


def _by(
    records: Sequence[Mapping[str, Any]], key: str
) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in records:
        groups[str(r[key])].append(r)
    return dict(sorted(groups.items()))


def system_metrics(
    records: Sequence[Mapping[str, Any]], route_records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """All G8 numbers of one translator from its item and routing records."""
    spec = [r for r in records if r["kind"] == "spec"]
    out: dict[str, Any] = {"n_items": len(records), "n_spec": len(spec)}
    out["exact"] = rate(sum(bool(r.get("exact")) for r in spec), len(spec))
    out["exact_by_category"] = {
        k: rate(sum(bool(r.get("exact")) for r in v), len(v))
        for k, v in _by(spec, "category").items()
    }
    out["exact_by_style"] = {
        k: rate(sum(bool(r.get("exact")) for r in v), len(v))
        for k, v in _by(spec, "style").items()
    }
    fields: dict[str, list[bool]] = defaultdict(list)
    for r in spec:
        for name, ok in (r.get("fields") or {}).items():
            fields[name].append(bool(ok))
    out["field_accuracy"] = {k: float(np.mean(v)) for k, v in sorted(fields.items())}
    # routing: R11 spec and clarify items, then R7a spec and clarify items
    ours = [r for r in records if r["kind"] in ("spec", "clarify")]
    theirs = [r for r in route_records if r["kind"] in ("spec", "clarify")]
    mixed = [bool(r["route_ok"]) for r in ours] + [bool(r["route_ok"]) for r in theirs]
    out["routing"] = rate(sum(mixed), len(mixed))
    out["routing_by_domain"] = {
        "incident": rate(sum(bool(r["route_ok"]) for r in ours), len(ours)),
        **{
            d: rate(sum(bool(r["route_ok"]) for r in v), len(v))
            for d, v in _by(theirs, "domain").items()
        },
    }
    # decisions on every item (spec / clarify / refuse)
    out["decision"] = rate(sum(bool(r["decision_ok"]) for r in records), len(records))
    out["decision_by_kind"] = {
        k: {
            **rate(sum(bool(r["decision_ok"]) for r in v), len(v)),
            "status": _counts(r["status"] for r in v),
        }
        for k, v in _by(records, "kind").items()
    }
    out["decision_by_category"] = {
        k: rate(sum(bool(r["decision_ok"]) for r in v), len(v))
        for k, v in _by(records, "category").items()
    }
    # task score
    scored = [r for r in spec if "gold_score" in r]
    nl = [float(r["nl_score"]) for r in scored]
    gold = [float(r["gold_score"]) for r in scored]
    diff = paired_bootstrap([a - b for a, b in zip(nl, gold)])
    out["task"] = {
        "n": len(scored),
        "nl": float(np.mean(nl)) if nl else None,
        "gold": float(np.mean(gold)) if gold else None,
        "diff": diff,
        "nl_top1": float(np.mean([r["nl_top1"] for r in scored])) if scored else None,
        "gold_top1": (
            float(np.mean([r["gold_top1"] for r in scored])) if scored else None
        ),
        "wrong_incident": sum(
            1
            for r in scored
            if r.get("ticket") is not None and r["ticket"] != r.get("gold_ticket")
        ),
        "no_run": sum(1 for r in scored if r["status"] != "ok"),
    }
    # reverse side (only executed natural-language runs have a summary)
    rev = [r for r in spec if "round_trip" in r]
    out["round_trip"] = rate(sum(bool(r["round_trip"]) for r in rev), len(rev))
    held = [bool(v) for r in rev for v in (r.get("facts") or {}).values()]
    facts_by_name: dict[str, list[bool]] = defaultdict(list)
    for r in rev:
        for name, ok in (r.get("facts") or {}).items():
            facts_by_name[name].append(bool(ok))
    out["fidelity"] = {
        "n_summaries": len(rev),
        "n_facts": len(held),
        "facts": float(np.mean(held)) if held else None,
        "all_facts": (
            float(np.mean([bool(r.get("all_facts")) for r in rev])) if rev else None
        ),
        "by_fact": {k: float(np.mean(v)) for k, v in sorted(facts_by_name.items())},
    }
    # latency (warm: the first executed item is dropped)
    timed = [r for r in spec if "total_s" in r and r.get("summary_s") is not None]
    warm = timed[1:]
    out["latency"] = {
        "n": len(warm),
        "total_p50": _f(percentile([r["total_s"] for r in warm], 50)) if warm else None,
        "total_p95": _f(percentile([r["total_s"] for r in warm], 95)) if warm else None,
        "translate_p50": (
            _f(percentile([r["translate_s"] for r in warm], 50)) if warm else None
        ),
        "team_p50": _f(percentile([r["team_s"] for r in warm], 50)) if warm else None,
        "summary_p50": (
            _f(percentile([r["summary_s"] for r in warm], 50)) if warm else None
        ),
    }
    return out


def _counts(values: Any) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for v in values:
        out[str(v)] += 1
    return dict(sorted(out.items()))


def failures(
    records: Sequence[Mapping[str, Any]], limit: int = 12
) -> dict[str, list[dict[str, Any]]]:
    """Examples of what went wrong: translations, decisions, round trips, facts."""
    spec = [r for r in records if r["kind"] == "spec"]
    keep = (
        "id",
        "category",
        "style",
        "text",
        "gold",
        "pred",
        "status",
        "reason",
        "routed",
    )
    out: dict[str, list[dict[str, Any]]] = {
        "translation": [
            {k: r.get(k) for k in keep} for r in spec if not r.get("exact")
        ][:limit],
        "decision": [
            {k: r.get(k) for k in keep}
            for r in records
            if r["kind"] != "spec" and not r["decision_ok"]
        ][:limit],
        "round_trip": [
            {
                "id": r["id"],
                "summary": r.get("summary"),
                "recon_status": r.get("recon_status"),
                "recon_params": r.get("recon_params"),
                "gold": r["gold"],
            }
            for r in spec
            if "round_trip" in r and not r["round_trip"]
        ][:limit],
        "facts": [
            {
                "id": r["id"],
                "summary": r.get("summary"),
                "missing": [k for k, v in (r.get("facts") or {}).items() if not v],
            }
            for r in spec
            if r.get("facts") and not r.get("all_facts")
        ][:limit],
    }
    return out


# ---------------------------------------------------------------------- verdict


def criteria(m: Mapping[str, Any], vram_gb: Optional[float]) -> list[dict[str, Any]]:
    """The five G8 targets for one system, each with its measured value and result."""
    ex, rt, fid, lat = m["exact"], m["round_trip"], m["fidelity"], m["latency"]
    task = m["task"]["diff"]
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

    row("exact match", f">= {EXACT_PASS:.0%} (kill < {EXACT_KILL:.0%})", ci(ex),
        None if ex["p"] is None else ex["p"] >= EXACT_PASS)  # fmt: skip
    row("domain routing", f">= {ROUTING_PASS:.0%}", ci(m["routing"]),
        None if m["routing"]["p"] is None else m["routing"]["p"] >= ROUTING_PASS)  # fmt: skip
    if task["mean"] is None:
        row(
            "task score, NL vs gold",
            f"lower CI bound > -{NONINF_MARGIN:.0%}",
            "n/a",
            None,
        )
    else:
        row(
            "task score, NL vs gold",
            f"lower CI bound of NL - gold > -{NONINF_MARGIN * 100:.0f} points",
            f"NL {m['task']['nl']:.1%} vs gold {m['task']['gold']:.1%}; diff "
            f"{task['mean'] * 100:+.1f} [{task['lo'] * 100:+.1f}, {task['hi'] * 100:+.1f}] points (n={task['n']})",
            task["lo"] > -NONINF_MARGIN,
        )
    row("round trip", f">= {ROUND_TRIP_PASS:.0%}", ci(rt),
        None if rt["p"] is None else rt["p"] >= ROUND_TRIP_PASS)  # fmt: skip
    fv = fid["facts"]
    row("summary field fidelity", f">= {FIDELITY_PASS:.0%} of facts (kill < {FIDELITY_KILL:.0%})",
        "n/a" if fv is None else f"{fv:.1%} of {fid['n_facts']} facts; all facts in {fid['all_facts']:.1%} of {fid['n_summaries']} summaries",
        None if fv is None else fv >= FIDELITY_PASS)  # fmt: skip
    p95 = lat["total_p95"]
    row("warm end-to-end p95", f"<= {P95_S:.0f} s",
        "n/a" if p95 is None else f"{p95:.2f} s (p50 {lat['total_p50']:.2f} s: translate {lat['translate_p50']:.2f}, team {lat['team_p50']:.2f}, summary {lat['summary_p50']:.2f})",
        None if p95 is None else p95 <= P95_S)  # fmt: skip
    row("peak VRAM", f"<= {VRAM_GB:.0f} GB",
        "n/a" if vram_gb is None else f"{vram_gb:.1f} GB",
        None if vram_gb is None else vram_gb <= VRAM_GB)  # fmt: skip
    return rows


def verdict(m: Mapping[str, Any], vram_gb: Optional[float]) -> dict[str, Any]:
    """``PASS``, ``KILL`` or ``NOT PASSED`` with the reasons."""
    rows = criteria(m, vram_gb)
    kills = []
    ex, fid, diff = m["exact"], m["fidelity"], m["task"]["diff"]
    if ex["p"] is not None and ex["p"] < EXACT_KILL:
        kills.append(f"exact match {ex['p']:.1%} < {EXACT_KILL:.0%}")
    if fid["facts"] is not None and fid["facts"] < FIDELITY_KILL:
        kills.append(f"fidelity {fid['facts']:.1%} < {FIDELITY_KILL:.0%}")
    if diff["hi"] is not None and diff["hi"] < 0:
        kills.append("NL task score significantly below the gold run")
    results = [r["result"] for r in rows]
    if kills:
        label = "KILL"
    elif all(r == "PASS" for r in results):
        label = "PASS"
    else:
        label = "NOT PASSED"
    return {"verdict": label, "kill_reasons": kills, "criteria": rows}


# ---------------------------------------------------------------------- render


def pct(r: Mapping[str, Any]) -> str:
    return "n/a" if r["p"] is None else f"{r['p']:.1%} ({r['hits']}/{r['n']})"


def render_system(name: str, s: Mapping[str, Any]) -> list[str]:
    m, v = s["metrics"], s["verdict"]
    lines = [
        f"### {s['label']}",
        "",
        f"**{v['verdict']}**"
        + (f" ({'; '.join(v['kill_reasons'])})" if v["kill_reasons"] else ""),
        "",
    ]
    lines += ["| Criterion | Target | Measured | Result |", "|---|---|---|---|"]
    for r in v["criteria"]:
        lines.append(
            f"| {r['criterion']} | {r['target']} | {r['measured']} | {r['result']} |"
        )
    lines += [
        "",
        "Routing by desk: "
        + "; ".join(f"{k} {pct(x)}" for k, x in m["routing_by_domain"].items())
        + ".",
        "",
    ]
    lines += [
        f"Final decision on all {m['n_items']} items (spec answered with a spec, clarify with a question, refuse with a refusal): {pct(m['decision'])}. "
        + " ".join(
            f"{k}: {pct(x)} {x['status']}." for k, x in m["decision_by_kind"].items()
        ),
        "",
    ]
    task = m["task"]
    if task["n"]:
        lines += [
            f"Task score: NL joint accuracy {task['nl']:.1%} (top-1 {task['nl_top1']:.1%}), gold {task['gold']:.1%} (top-1 {task['gold_top1']:.1%}) over {task['n']} requests; {task['no_run']} had no run (refused, clarified or invalid) and {task['wrong_incident']} named a different incident than the gold.",
            "",
        ]
    lines += ["| Category | Exact match | Decision (all kinds) |", "|---|---|---|"]
    cats = sorted(set(m["exact_by_category"]) | set(m["decision_by_category"]))
    for c in cats:
        e = m["exact_by_category"].get(c)
        lines.append(
            f"| {c} | {pct(e) if e else '-'} | {pct(m['decision_by_category'][c])} |"
        )
    lines += ["", "| Style | Exact match |", "|---|---|"]
    for c, e in m["exact_by_style"].items():
        lines.append(f"| {c} | {pct(e)} |")
    fa = m["field_accuracy"]
    if fa:
        lines += [
            "",
            "Field accuracy on spec items: "
            + ", ".join(f"{k} {x:.1%}" for k, x in fa.items())
            + ".",
        ]
    bf = m["fidelity"]["by_fact"]
    if bf:
        lines += [
            "",
            "Fact accuracy: " + ", ".join(f"{k} {x:.1%}" for k, x in bf.items()) + ".",
        ]
    return lines + [""]


def render_failures(
    label: str, fails: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[str]:
    lines = [f"#### {label}: failure examples", ""]
    for kind, rows in fails.items():
        if not rows:
            continue
        lines += [f"{kind} ({len(rows)} shown):", ""]
        for r in rows:
            body = "; ".join(
                f"{k}={_short(v)}"
                for k, v in r.items()
                if k != "id" and v not in (None, "", [])
            )
            lines.append(f"- `{r['id']}` {body}")
        lines.append("")
    return lines


def _short(value: Any, n: int = 220) -> str:
    text = value if isinstance(value, str) else str(value)
    return text if len(text) <= n else text[: n - 1] + "..."


def render(report: Mapping[str, Any], analysis: Optional[str] = None) -> str:
    """The markdown of ``docs/results/r11_ecosystem.md`` (``analysis`` is carried over)."""
    goal = report.get("goal", "G8")
    title = report.get("title", "R11: the ecosystem on real incidents")
    sealed_file = report.get("sealed_file", "r11_sealed_test.jsonl")
    script = report.get("command", "python scripts/r11_ecosystem.py report")
    lines = [
        f"# {title} (goal {goal})",
        "",
        f"Generated {report['generated']}. Command: `{script}`.",
        "",
        "One shared translator picks the domain (squad, plant or incident), extracts the GoalSpec, the command centre runs `rca.investigate` through the G4c trained team on a real RCAEval RE2 incident, the reverse translator summarises the trace, and the summary is routed and translated again.",
        "",
        f"- **Sealed set:** {report['n_sealed']} hand-written incident requests (`experiments/data/{sealed_file}`, SHA-256 `{report['sealed_sha256']}`), plus the R7a sealed spec and clarify items for the routing check (SHA-256 `{report['r7a_sealed_sha256']}`). Each translator ran once.",
        f"- **Catalogue:** {report['catalogue']['n']} test incidents (SS/OB repetitions 2 and 3), fingerprint `{report['catalogue']['fingerprint']}`.",
        f"- **Team:** {report['team']}",
        "",
        "## Verdict",
        "",
        f"| Translator | {goal} verdict | Exact match | Routing | NL - gold task score | Round trip | Fidelity | Warm p95 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in report["systems"].values():
        m = s["metrics"]
        d = m["task"]["diff"]
        task = (
            "n/a"
            if d["mean"] is None
            else f"{d['mean'] * 100:+.1f} [{d['lo'] * 100:+.1f}, {d['hi'] * 100:+.1f}]"
        )
        fid = m["fidelity"]["facts"]
        p95 = m["latency"]["total_p95"]
        lines.append(
            f"| {s['label']} | **{s['verdict']['verdict']}** | {pct(m['exact'])} | {pct(m['routing'])} | {task} | {pct(m['round_trip'])} | "
            f"{'n/a' if fid is None else f'{fid:.1%}'} | {'n/a' if p95 is None else f'{p95:.2f} s'} |"
        )
    lines += ["", "## Per translator", ""]
    for name, s in report["systems"].items():
        lines += render_system(name, s)
        lines += render_failures(s["label"], s["failures"])
    lines += [
        "## Analysis",
        "",
        "<!-- analysis:start -->",
        analysis or "*Analysis pending the full run.*",
        "<!-- analysis:end -->",
        "",
    ]
    return "\n".join(lines)
