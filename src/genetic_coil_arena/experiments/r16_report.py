"""R16 report and the G14 verdict.

Bars (fixed before the sealed run), for qwen2.5-coder:14b through ``GroundTranslator``
on the sealed set:

1. **Right command:** exact on at least 90% of the requests that should become a command.
2. **No wrong commands:** at most 3% of all requests end as a command other than the one
   asked (including any command on a request that should be asked about or refused),
   with the Wilson upper bound at most 6%.
3. **Better than keywords:** exact beats the keyword baseline by at least 20 points.

The state check runs after translation. The report also counts the wrong commands that
would still reach the drones in a nominal swarm (six drones, one with a vision module,
outdoors, batteries at 90%), i.e. those the check does not block.
"""

from __future__ import annotations

from typing import Any

from ..envs.swarm import ground as gs
from ..translator.metrics import wilson

MAIN = "ollama:qwen2.5-coder:14b"
BASELINE = "keyword"
EXACT_BAR, WRONG_BAR, WRONG_UPPER, GAIN_BAR = 0.90, 0.03, 0.06, 0.20
NOMINAL = gs.SwarmState(
    tuple(gs.Drone(i, 90.0, has_vision=(i == 0)) for i in range(6)), indoor=False
)


def reaches_drones(row: dict[str, Any], state: gs.SwarmState = NOMINAL) -> bool:
    """A sent command the state check lets through."""
    return row["status"] == "ok" and gs.check(row["params"], state).ok


def g14(runs: dict[tuple[str, str], dict], split: str = "sealed") -> dict[str, Any]:
    main, base = runs.get((split, MAIN)), runs.get((split, BASELINE))
    if main is None:
        return {"available": False}
    s = main["summary"]
    exact, wrong = s["exact"], s["wrong_commands"]
    out = {
        "available": True,
        "exact": exact,
        "wrong_commands": wrong,
        "bar_exact": exact[0] >= EXACT_BAR,
        "bar_wrong": wrong[0] <= WRONG_BAR and wrong[2] <= WRONG_UPPER,
    }
    if base is not None:
        gain = exact[0] - base["summary"]["exact"][0]
        out.update(gain=gain, bar_gain=gain >= GAIN_BAR)
    out["passed"] = bool(out["bar_exact"] and out["bar_wrong"] and out.get("bar_gain"))
    return out


def _pct(w: Any) -> str:
    p, lo, hi = w
    return "–" if p != p else f"{100 * p:.1f}% [{100 * lo:.0f}, {100 * hi:.0f}]"


def render(runs_list: list[dict]) -> tuple[str, dict]:
    runs = {(r["summary"]["split"], r["summary"]["translator"]): r for r in runs_list}
    # G14b (v5) is scored on its own fresh sealed set, "sealed-b"
    split = "sealed-b" if any(k[0] == "sealed-b" for k in runs) else "sealed"
    goal = "G14b" if split == "sealed-b" else "G14"
    verdict = g14(runs, split)
    verdict_split = split
    lines = ["# R16 (G14): a plain-language ground station for OpenSwarm", ""]
    if verdict["available"]:
        yes = lambda ok: "PASS" if ok else "FAIL"  # noqa: E731
        lines += [
            f"**{goal} {'PASSED' if verdict['passed'] else 'NOT PASSED'}** "
            f"(`{split}` set, `{MAIN}`).",
            "",
            f"- Right command: exact {_pct(verdict['exact'])} "
            f"(bar ≥ {100 * EXACT_BAR:.0f}%): {yes(verdict['bar_exact'])}.",
            f"- Wrong commands: {_pct(verdict['wrong_commands'])} of all requests "
            f"(bar ≤ {100 * WRONG_BAR:.0f}%, upper bound ≤ {100 * WRONG_UPPER:.0f}%): "
            f"{yes(verdict['bar_wrong'])}.",
        ]
        if "gain" in verdict:
            lines.append(
                f"- Better than keywords: {100 * verdict['gain']:+.1f} points exact "
                f"(bar ≥ +{100 * GAIN_BAR:.0f}): {yes(verdict['bar_gain'])}."
            )
        lines.append("")
    lines += [
        "Brackets: 95% Wilson intervals. *Exact*: the command sent is the one asked, every "
        "parameter equal after the station fills its defaults. *Asked* and *refused* are "
        "scored on the requests that should be asked about or refused. *Wrong commands*: "
        "any request that ends as a different command than asked, over all requests. "
        "*Reach the drones*: those wrong commands the state check lets through in a "
        "nominal swarm.",
        "",
        "| Split | Translator | Requests | Exact | Asked when unclear | Refused when should "
        "| Wrong commands | Reach the drones | Median latency |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    data: dict[str, Any] = {"g14": verdict, "runs": {}}
    for (split, tr), r in sorted(runs.items()):
        s = r["summary"]
        wrong_rows = [x for x in r["rows"] if x["outcome"] == "wrong command"]
        reach = sum(reaches_drones(x) for x in wrong_rows)
        lines.append(
            f"| {split} | `{tr}` | {s['n']} | {_pct(s['exact'])} "
            f"| {_pct(s['asked_when_unclear'])} | {_pct(s['refused_when_should'])} "
            f"| {_pct(s['wrong_commands'])} | {reach} | {s['median_latency_s']:.2f} s |"
        )
        data["runs"][f"{split}|{tr}"] = {
            **s, "reach_drones": reach,
            "reach_drones_rate": wilson(reach, s["n"]),
        }  # fmt: skip
    for (split, tr), r in sorted(runs.items()):
        if split != verdict_split:
            continue
        bad = [x for x in r["rows"] if not x["correct"]]
        if not bad:
            continue
        lines += [
            "",
            f"Misses on the `{split}` set, `{tr}` ({len(bad)}):",
            "",
            "| Request | Expected | Got |",
            "|---|---|---|",
        ]
        for x in bad:
            exp = x["gold"] if x["expect"] == "ok" else x["expect"]
            got = (
                x["params"] if x["status"] == "ok" else f"{x['status']}: {x['reason']}"
            )
            text = str(x["text"]).replace("|", "/")
            lines.append(f"| {text} | `{exp}` | {str(got).replace('|', '/')} |")
    lines += [
        "",
        "Reproduce: `python scripts/r16_ground_station.py run --translator <t> --split "
        "sealed`, then `report --from runs:r16-*`.",
        "",
    ]
    return "\n".join(lines), data
