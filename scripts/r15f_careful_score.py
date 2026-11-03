"""R15f: score the careful-mode answers on the crops of ``r15f_careful_crops.py``.

Three truths, as in R15e: clear cases (R15b-matched people and spots the visual check,
committed before the session in ``docs/results/r15f_adjudication.json``, calls a person,
against spots it calls clearly not a person), R15b's matching rule, and any labelled
person within the ring. "Can't tell" counts as not confirmed. The clear-cases row is
compared with R15e's quick look (Fisher's exact test; different spots drawn the same way).

Read-out, fixed before any answer was read: ENOUGH if the clear-case false accept is at
most 5% and the clear-case hit at least 85% (R17 step 6b: accepting 5% of empty views
already pushes wrong finds to 7%); NOT ENOUGH if the false accept is between 5% and 15%
or the hit falls below 85%; NO GAIN if the false accept is 15% or more.

python scripts/r15f_careful_score.py --answers <folder of answer JSON files> [--out docs/results/r15f_careful.md]
"""

import argparse
import json
import math
from pathlib import Path

from genetic_coil_arena.experiments import r15_scout_brain as r15

OUT = r15.DATA_DIR / "r15f_careful"
FA_ENOUGH, FA_NO_GAIN, HIT_MIN = 0.05, 0.15, 0.85


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if not n:
        return (float("nan"),) * 3
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


def fisher(a: int, n1: int, b: int, n2: int) -> float:
    """Two-sided Fisher's exact p for a/n1 against b/n2."""
    m, n = a + b, n1 + n2

    def prob(x: int) -> float:
        return math.comb(n1, x) * math.comb(n2, m - x) / math.comb(n, m)

    obs = prob(a)
    lo, hi = max(0, m - n2), min(n1, m)
    return min(
        1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= obs * (1 + 1e-9))
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--answers", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("docs/results/r15f_careful.md"))
    args = p.parse_args()
    answers = {}
    for f in sorted(args.answers.glob("k*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        answers[f.stem] = d.get("data", d)
    truth = {
        t["id"]: t for t in json.loads((OUT / "truth.json").read_text(encoding="utf-8"))
    }
    assert set(answers) <= set(truth), "answers for unknown spots"
    adj = json.loads(Path("docs/results/r15f_adjudication.json").read_text("utf-8"))
    said = {i: a["label"] for i, a in answers.items()}
    truths = [
        ("clear cases (visual check made before the session)",
         {i for i in said if truth[i]["person"]} | (set(adj["person"]) & set(said)),
         set(adj["not_person"]) & set(said)),
        ("matched by R15b's scoring rule",
         {i for i in said if truth[i]["person"]}, {i for i in said if not truth[i]["person"]}),
        ("a labelled person within the ring",
         {i for i in said if truth[i]["near_person"]},
         {i for i in said if not truth[i]["near_person"]}),
    ]  # fmt: skip
    rows = []
    for name, pos, neg in truths:
        hit = sum(said[i] == "person" for i in pos)
        fa = sum(said[i] == "person" for i in neg)
        rows.append({"truth": name, "persons": len(pos), "hit_k": hit,
                     "hit": wilson(hit, len(pos)), "unsure_pos": sum(said[i] == "can't tell" for i in pos),
                     "empties": len(neg), "fa_k": fa, "false_accept": wilson(fa, len(neg)),
                     "unsure_neg": sum(said[i] == "can't tell" for i in neg)})  # fmt: skip
    quick = json.loads(Path("docs/results/r15e_operator.json").read_text("utf-8"))[
        "rows"
    ][0]
    c = rows[0]
    compare = {
        "quick_hit": [quick["hit_k"], quick["persons"]],
        "quick_fa": [quick["fa_k"], quick["empties"]],
        "p_hit": fisher(c["hit_k"], c["persons"], quick["hit_k"], quick["persons"]),
        "p_fa": fisher(c["fa_k"], c["empties"], quick["fa_k"], quick["empties"]),
    }
    fa, hit = c["false_accept"][0], c["hit"][0]
    if fa >= FA_NO_GAIN:
        verdict = "NO GAIN"
    elif fa <= FA_ENOUGH and hit >= HIT_MIN:
        verdict = "ENOUGH"
    else:
        verdict = "NOT ENOUGH"
    ms = sorted(a["ms"] for a in answers.values())
    rep = {"rows": rows, "compare": compare, "verdict": verdict, "answered": len(answers),
           "median_ms": ms[len(ms) // 2] if ms else None,
           "unsure": sum(v == "can't tell" for v in said.values())}  # fmt: skip
    lines = ["# R15f: does a careful look cut the false alarms an operator confirms?", "",
             f"Verdict (read-out fixed before the session): **{verdict}**. Clear-case false "
             f"accept {100 * fa:.0f}% (ENOUGH needs ≤ {100 * FA_ENOUGH:.0f}% with a hit rate "
             f"≥ {100 * HIT_MIN:.0f}%; NO GAIN is ≥ {100 * FA_NO_GAIN:.0f}%).", "",
             f"{len(answers)} of 200 new spots answered (median {rep['median_ms'] / 1000:.1f} s; "
             f"buttons locked for the first 2 s), {rep['unsure']} \"can't tell\" (counted as not "
             "confirmed). Close-up and a wider view of each spot.", "",
             "| Truth | Spots on a person | Says person | Can't tell | Spots without | Says "
             "person (false accept) | Can't tell |", "|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in rows:
        h, f = r["hit"], r["false_accept"]
        lines.append(f"| {r['truth']} | {r['persons']} | {r['hit_k']} = {100 * h[0]:.0f}% "
                     f"[{100 * h[1]:.0f}, {100 * h[2]:.0f}] | {r['unsure_pos']} | {r['empties']} | "
                     f"{r['fa_k']} = {100 * f[0]:.0f}% [{100 * f[1]:.0f}, {100 * f[2]:.0f}] | "
                     f"{r['unsure_neg']} |")  # fmt: skip
    qh, qf = compare["quick_hit"], compare["quick_fa"]
    lines += ["", f"Quick look (R15e, clear cases, other spots): hit {qh[0]}/{qh[1]}, false "
              f"accept {qf[0]}/{qf[1]}. Careful against quick, Fisher's exact: hit p = "
              f"{compare['p_hit']:.3f}, false accept p = {compare['p_fa']:.3f}.", "",
              "Brackets: 95% Wilson intervals. One operator, a second session (some learning "
              "from R15e is possible). Unlike R15e, the visual check of the false alarms was "
              "made from both views and committed before any answer was read.", ""]  # fmt: skip
    md = "\n".join(lines)
    args.out.write_text(md, encoding="utf-8")
    args.out.with_suffix(".json").write_text(
        json.dumps(rep, indent=1), encoding="utf-8"
    )
    print(md)


if __name__ == "__main__":
    main()
