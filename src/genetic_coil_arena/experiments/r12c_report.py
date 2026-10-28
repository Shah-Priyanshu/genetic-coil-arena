"""R12c report: the G10b numbers, verdict and markdown from the paired per-item rows.

**G10b** (the desk checks the translator's literals, ``docs/goals.md``), on the R12c sealed set,
check on = desk v4, check off = the G10 desk, both scored from the same raw translations:

1. check-on exact match >= 95% on the spec items;
2. silent wrong answers <= 1 (a spec item run on another incident than its ``resolves_to``, or
   a clarify or refuse item that was answered);
3. paired: exact-match difference (on - off) >= +3 points and an exact binomial (McNemar) test
   on the discordant pairs with p < 0.05;
4. NL task score non-inferior to the gold run, the lower bound of the paired 95% bootstrap CI
   of NL minus gold > -5 points.

Kill: check-on exact match < 90%, or the check breaks more items than it fixes.
"""

from collections.abc import Mapping, Sequence
from math import comb
from typing import Any, Optional

from ..envs.logtriage.literal_check import check_params
from ..translator import OK, exact_match
from . import r11_report as rep
from .r12c_desk import side, silent_wrong

EXACT_PASS = 0.95
EXACT_KILL = 0.90
SILENT_MAX = 1
GAIN = 0.03
ALPHA = 0.05


def mcnemar_exact(fixed: int, broke: int) -> float:
    """Two-sided exact binomial p of ``fixed`` vs ``broke`` discordant pairs (1.0 for none)."""
    n = fixed + broke
    if n == 0:
        return 1.0
    k = min(fixed, broke)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return float(min(1.0, 2 * tail))


def _exact(rec: Mapping[str, Any]) -> bool:
    return bool(rec.get("exact"))


def side_metrics(
    rows: Sequence[Mapping[str, Any]],
    which: str,
    resolves_to: Mapping[str, Optional[str]],
    traps: set[str],
) -> dict[str, Any]:
    recs = [side(r, which) for r in rows]
    m = rep.system_metrics(recs, [])
    wrong = [r["id"] for r in recs if silent_wrong(r, resolves_to.get(r["id"]))]
    spec = [r for r in recs if r["kind"] == "spec"]
    right_incident_other = [
        r["id"]
        for r in spec
        if r["status"] == OK
        and r.get("ticket") == resolves_to.get(r["id"])
        and not _exact(r)
    ]
    return {
        "metrics": m,
        "exact": m["exact"],
        "silent_wrong": wrong,
        "silent_wrong_traps": [i for i in wrong if i in traps],
        "right_incident_other_mismatch": right_incident_other,
        "task": m["task"],
        "decision": m["decision"],
    }


def paired(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    spec = [r for r in rows if r["kind"] == "spec"]
    on = [_exact(r) for r in spec]
    off = [_exact(r["off"]) for r in spec]
    fixed = [r["id"] for r, a, b in zip(spec, on, off) if a and not b]
    broke = [r["id"] for r, a, b in zip(spec, on, off) if b and not a]
    diff = rep.paired_bootstrap([float(a) - float(b) for a, b in zip(on, off)])
    return {
        "n": len(spec),
        "fixed": fixed,
        "broke": broke,
        "p": mcnemar_exact(len(fixed), len(broke)),
        "diff": diff,
    }


def rules_fired(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        for rule in r.get("rules", []):
            out[rule] = out.get(rule, 0) + 1
    return out


def g10b_metrics(
    rows: Sequence[Mapping[str, Any]],
    resolves_to: Mapping[str, Optional[str]],
    traps: set[str],
) -> dict[str, Any]:
    return {
        "n_items": len(rows),
        "off": side_metrics(rows, "off", resolves_to, traps),
        "on": side_metrics(rows, "on", resolves_to, traps),
        "paired": paired(rows),
        "rules": rules_fired(rows),
        "n_changed": sum(1 for r in rows if r.get("rules")),
    }


def criteria(m: Mapping[str, Any]) -> list[dict[str, Any]]:
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

    ex = m["on"]["exact"]
    row(
        "exact match, check on",
        f">= {EXACT_PASS:.0%} (kill < {EXACT_KILL:.0%})",
        _ci(ex),
        None if ex["p"] is None else ex["p"] >= EXACT_PASS,
    )
    sw = m["on"]["silent_wrong"]
    row(
        "silent wrong answers, check on",
        f"<= {SILENT_MAX}",
        f"{len(sw)}" + (f" ({', '.join(sw)})" if sw else ""),
        len(sw) <= SILENT_MAX,
    )
    p = m["paired"]
    d = p["diff"]
    row(
        "exact match, check on - check off",
        f">= +{GAIN * 100:.0f} points, exact binomial p < {ALPHA}",
        (
            f"{d['mean'] * 100:+.1f} points ({len(p['fixed'])} fixed, {len(p['broke'])} broken), p = {p['p']:.4f}"
            if d["mean"] is not None
            else "n/a"
        ),
        None if d["mean"] is None else (d["mean"] >= GAIN - 1e-9 and p["p"] < ALPHA),
    )
    t = m["on"]["task"]
    td = t["diff"]
    row(
        "task score, NL vs gold, check on",
        f"lower CI bound of NL - gold > -{rep.NONINF_MARGIN * 100:.0f} points",
        (
            f"NL {t['nl']:.1%} vs gold {t['gold']:.1%}; diff {td['mean'] * 100:+.1f} [{td['lo'] * 100:+.1f}, {td['hi'] * 100:+.1f}] points (n={td['n']})"
            if td["mean"] is not None
            else "n/a"
        ),
        None if td["mean"] is None else td["lo"] > -rep.NONINF_MARGIN,
    )
    return rows


def _ci(r: Mapping[str, Any]) -> str:
    if r["p"] is None:
        return "n/a"
    return f"{r['p']:.1%} (n={r['n']}, 95% CI {r['lo']:.1%}-{r['hi']:.1%})"


def verdict(m: Mapping[str, Any]) -> dict[str, Any]:
    rows = criteria(m)
    kills = []
    ex = m["on"]["exact"]
    if ex["p"] is not None and ex["p"] < EXACT_KILL:
        kills.append(f"check-on exact match {ex['p']:.1%} < {EXACT_KILL:.0%}")
    p = m["paired"]
    if len(p["broke"]) > len(p["fixed"]):
        kills.append(
            f"the check breaks more items ({len(p['broke'])}) than it fixes ({len(p['fixed'])})"
        )
    label = (
        "KILL"
        if kills
        else "PASS" if all(r["result"] == "PASS" for r in rows) else "NOT PASSED"
    )
    return {"verdict": label, "kill_reasons": kills, "criteria": rows}


# ---------------------------------------------------------------------- post hoc


def posthoc(
    records: Sequence[Mapping[str, Any]],
    registry: Any,
    spec_name: str,
    now: Any,
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """The check applied to the stored raw predictions of an earlier sealed run.

    Only the stored ``pred`` (the translator's validated params) is used, so this re-scores
    exact match and the answered-but-should-refuse items; it does not re-run resolution or the
    team. Post-hoc: not part of any verdict.
    """
    from ..protocol.goalspec import SpecValidationError

    spec = [r for r in records if r["kind"] == "spec"]
    fixed, broke, would_refuse = [], [], []
    n_exact_off = n_exact_on = 0
    for r in records:
        pred = r.get("pred")
        out = pred
        valid = pred is not None
        if pred is not None and r["status"] == "ok":
            res = check_params(pred, r["text"], now)
            if res.rules:
                try:
                    registry.make(spec_name, res.params)
                    out = res.params
                except SpecValidationError:
                    out, valid = None, False
        if r["kind"] == "spec":
            before = bool(r.get("exact"))
            after = (
                valid and r["status"] == "ok" and exact_match(out, r["gold"], schema)
                if out is not None
                else False
            )
            if out is pred:  # nothing changed: keep the stored result
                after = before
            n_exact_off += before
            n_exact_on += after
            if after and not before:
                fixed.append(r["id"])
            if before and not after:
                broke.append(r["id"])
        elif r["status"] == "ok" and out is None and pred is not None:
            would_refuse.append(r["id"])
    n = len(spec)
    return {
        "n_spec": n,
        "exact_off": rep.rate(n_exact_off, n),
        "exact_on": rep.rate(n_exact_on, n),
        "fixed": fixed,
        "broke": broke,
        "p": mcnemar_exact(len(fixed), len(broke)),
        "answered_but_would_refuse": would_refuse,
    }


# ---------------------------------------------------------------------- markdown


def _short(value: Any, n: int = 90) -> str:
    text = "-" if value is None else str(value)
    return text if len(text) <= n else text[: n - 1] + "..."


def _fields(params: Optional[Mapping[str, Any]]) -> str:
    if params is None:
        return "-"
    keep = {k: v for k, v in params.items() if k in ("ticket", "at", "top_k")}
    extra = []
    if params.get("exclude_services"):
        extra.append(f"excl={params['exclude_services']}")
    if params.get("want_fault_type") is False:
        extra.append("no fault type")
    return ", ".join([f"{k}={v}" for k, v in keep.items()] + extra) or "-"


def render_system(s: Mapping[str, Any]) -> list[str]:
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
    off, on = m["off"], m["on"]
    lines += [
        "",
        "| | Check off (G10 desk) | Check on (desk v4) |",
        "|---|---|---|",
        f"| exact match, spec items | {_ci(off['exact'])} | {_ci(on['exact'])} |",
        f"| silent wrong answers | {len(off['silent_wrong'])} | {len(on['silent_wrong'])} |",
        f"| silent wrong on the guess traps | {len(off['silent_wrong_traps'])} | {len(on['silent_wrong_traps'])} |",
        f"| right incident, other field wrong | {len(off['right_incident_other_mismatch'])} | {len(on['right_incident_other_mismatch'])} |",
        f"| decision (spec / clarify / refuse) | {rep.pct(off['decision'])} | {rep.pct(on['decision'])} |",
        f"| NL task score vs gold (mean diff) | {_td(off['task'])} | {_td(on['task'])} |",
        "",
        f"The check changed {m['n_changed']} of {m['n_items']} translations; rules fired: "
        + (", ".join(f"{k} {v}" for k, v in sorted(m["rules"].items())) or "none")
        + ".",
        "",
    ]
    by_cat = sorted(
        set(off["metrics"]["exact_by_category"])
        | set(on["metrics"]["exact_by_category"])
    )
    lines += ["| Category | Exact off | Exact on |", "|---|---|---|"]
    for c in by_cat:
        a = off["metrics"]["exact_by_category"].get(c)
        b = on["metrics"]["exact_by_category"].get(c)
        lines.append(
            f"| {c} | {rep.pct(a) if a else '-'} | {rep.pct(b) if b else '-'} |"
        )
    lines += ["", "Items the check fixed, broke or left wrong:", ""]
    rows = {r["id"]: r for r in s["records"]}
    p = m["paired"]
    for label, ids in (("fixed", p["fixed"]), ("broke", p["broke"])):
        for i in ids:
            lines.append(_item_line(label, rows[i]))
    for r in s["records"]:
        if r["kind"] == "spec" and not r.get("exact"):
            lines.append(_item_line("still wrong", r))
        if silent_wrong(r, s["resolves_to"].get(r["id"])):
            lines.append(_item_line("silent wrong", r))
    return lines + [""]


def _item_line(label: str, r: Mapping[str, Any]) -> str:
    return (
        f"- *{label}* `{r['id']}` \"{_short(r['text'], 110)}\": gold {_fields(r['gold'])}; "
        f"raw {_fields(r['raw_pred'])} ({r['off']['status']}); checked {_fields(r['checked_pred'])} "
        f"({r['status']}); rules {r['rules'] or '-'}"
    )


def _td(t: Mapping[str, Any]) -> str:
    d = t["diff"]
    return (
        "n/a"
        if d["mean"] is None
        else f"{d['mean'] * 100:+.1f} [{d['lo'] * 100:+.1f}, {d['hi'] * 100:+.1f}]"
    )


def render(report: Mapping[str, Any], analysis: Optional[str] = None) -> str:
    """The markdown of ``docs/results/r12c_literal_check.md``."""
    g = report["g10b"]
    lines = [
        "# R12c: the desk checks the translator's literals (goal G10b)",
        "",
        f"Generated {report['generated']}. Command: `python scripts/r12c_desk.py report`.",
        "",
        f"- **Sealed set:** {g['n_sealed']} hand-written requests (`experiments/data/r12c_sealed_test.jsonl`, SHA-256 `{g['sealed_sha256']}`), {g['n_traps']} of them guess traps. Each translator ran once; the report scores the same raw translations with the literal check off (the G10 desk) and on (desk v4).",
        f"- **Catalogue:** {g['catalogue']['n']} test incidents, fingerprint `{g['catalogue']['fingerprint']}`; desk clock Wednesday 2024-12-11 09:00 UTC; prompts and spec v3 as in G10.",
        "- **Rules** (`envs/logtriage/literal_check.py`): " + g["rules_text"],
        "",
        "### Verdict",
        "",
        "| Translator | G10b verdict | Exact off | Exact on | Fixed / broken | McNemar p | Silent wrong (off / on) | NL - gold, on |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in g["systems"].values():
        m = s["metrics"]
        lines.append(
            f"| {s['label']} | **{s['verdict']['verdict']}** | {rep.pct(m['off']['exact'])} | {rep.pct(m['on']['exact'])} | {len(m['paired']['fixed'])} / {len(m['paired']['broke'])} | {m['paired']['p']:.4f} | {len(m['off']['silent_wrong'])} / {len(m['on']['silent_wrong'])} | {_td(m['on']['task'])} |"
        )
    lines += ["", "### Per translator", ""]
    for s in g["systems"].values():
        lines += render_system(s)
    if report.get("posthoc"):
        lines += [
            "### Post-hoc (not part of any verdict)",
            "",
            "The check applied to the stored raw predictions of the earlier sealed 14B runs (exact match re-scored from the stored params; resolution and the team are not re-run).",
            "",
            "| Earlier run | Spec items | Exact as run | Exact with the check | Fixed / broken | Answered items the check would refuse |",
            "|---|---|---|---|---|---|",
        ]
        for name, h in report["posthoc"].items():
            lines.append(
                f"| {name} | {h['n_spec']} | {rep.pct(h['exact_off'])} | {rep.pct(h['exact_on'])} | {len(h['fixed'])} / {len(h['broke'])} | {len(h['answered_but_would_refuse'])} |"
            )
        lines.append("")
    lines += [
        "## Analysis",
        "",
        "<!-- analysis:start -->",
        analysis or "*Analysis pending the full run.*",
        "<!-- analysis:end -->",
        "",
    ]
    return "\n".join(lines)
