"""R21: the ground station's LLM with its weights edited (experiments/r21_ground_lora.py).

python scripts/r21_ground_lora.py data                     # check labels and leaks, write train.jsonl
python scripts/r21_ground_lora.py audit                    # the 14B on every training label
python scripts/r21_ground_lora.py data                     # again, with label_drops.json
python scripts/r21_ground_lora.py fit --seed 0             # QLoRA, dev after each epoch
python scripts/r21_ground_lora.py run --system asis|lora --split dev|sealed-b
python scripts/r21_ground_lora.py report
python scripts/r21_ground_lora.py fit --seed 1          # R21b: the replicate seed
python scripts/r21_ground_lora.py run --system lora --seed 1 --split sealed-b
python scripts/r21_ground_lora.py report-b
# R21c: G14c's fresh sealed set, once per system (the 14B through scripts/r16_ground_station.py)
python scripts/r21_ground_lora.py run --system lora --seed 0 --split sealed-c   # also --seed 1, asis
python scripts/r21_ground_lora.py report-c

Everything is written under ``D:/Projects/genetic-coil-arena-data/r21_ground_lora``;
``report`` writes ``docs/results/r21_ground_lora.{md,json}``. The sealed set is run once
per system, after the read-out was committed.
"""

import argparse
import collections
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", r"D:\hf_cache")

from genetic_coil_arena.experiments import r16_sealed  # noqa: E402
from genetic_coil_arena.experiments import r21_ground_lora as rg  # noqa: E402

G14B = Path("docs/results/r16b_ground_station.json")
Q14_ROWS = Path(
    "D:/Projects/genetic-coil-arena-gpu-results/runs/r16b-sealed-q14/artifacts/"
    "r16_sealed-b_ollama_qwen2.5-coder_14b.jsonl"
)  # G14b's per-item rows
BARS = {"exact": 0.90, "wrong": 0.03, "wrong_hi": 0.06}
EDIT_POINTS = 0.10  # editing matters: exact at least 10 points above the 7B as is


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def cmd_data(_args) -> None:
    items = rg.load_train()
    bad = dict(rg.check_labels(items))
    leaks = dict(rg.check_leaks(items))
    drops_path = rg.DATA_DIR / "label_drops.json"  # wrong labels found in the 14B audit
    drops = json.loads(drops_path.read_text("utf-8")) if drops_path.exists() else {}
    bad.update(drops)
    keep = [it for it in items if it["id"] not in bad and it["id"] not in leaks]
    for k, v in sorted(bad.items()):
        log(f"label dropped {k}: {v}")
    log(f"{len(items)} written, {len(bad)} label problems, {len(leaks)} too close to a "
        f"held-out text, {len(keep)} kept: "
        f"{dict(collections.Counter(it['expect'] for it in keep))}")  # fmt: skip
    with open(rg.DATA_DIR / "train.jsonl", "w", encoding="utf-8") as f:
        for it in keep:
            f.write(json.dumps(it) + "\n")
    (rg.DATA_DIR / "data.json").write_text(json.dumps(
        {"written": len(items), "label_problems": bad, "leaks": leaks, "kept": len(keep)},
        indent=1), "utf-8")  # fmt: skip


def cmd_audit(_args) -> None:
    """The deployed 14B (Ollama, as in G14b) on every kept training request; the
    disagreements are reviewed by hand and wrong labels listed in ``label_drops.json``."""
    import r16_ground_station as r16s
    from r7a_vertical_slice import OllamaServer

    from genetic_coil_arena.backends.ollama import DEFAULT_HOST, OllamaBackend

    items = rg.r16.load_items(rg.DATA_DIR / "train.jsonl")
    model = "qwen2.5-coder:14b"
    with OllamaServer(DEFAULT_HOST) as server:
        server.require(model)
        server.unload_all()
        backend = OllamaBackend(model, keep_alive=r16s.KEEP, options=r16s.CTX)
        try:
            tr = rg.r16.GroundTranslator(backend, rg.r16.build_registry(), keep_alive=r16s.KEEP)
            rows = rg.run_items(tr, items, "audit", log)
        finally:
            backend.unload()
    with open(rg.DATA_DIR / "audit.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    off = [r for r in rows if not r["correct"]]
    log(f"audit: the 14B disagrees with {len(off)} of {len(rows)} labels")


def cmd_fit(args) -> None:
    start = time.perf_counter()
    items = rg.r16.load_items(rg.DATA_DIR / "train.jsonl")
    out = rg.DATA_DIR / "models" / f"lora_s{args.seed}"
    hist = rg.train_lora(items, r16_sealed.load_dev(), out, seed=args.seed,
                         epochs=args.epochs, log=log)  # fmt: skip
    best = rg.pick_epoch(hist)
    (out / "fit.json").write_text(json.dumps(
        {"seed": args.seed, "history": hist, "chosen_epoch": best,
         "seconds": time.perf_counter() - start}, indent=1), "utf-8")  # fmt: skip
    log(f"chosen epoch {best}")


def adapter_path(seed: int = 0) -> Path:
    out = rg.DATA_DIR / "models" / f"lora_s{seed}"
    return out / f"epoch{json.loads((out / 'fit.json').read_text('utf-8'))['chosen_epoch']}"


def system_name(system: str, seed: int) -> str:
    """File name of a system's runs: R21's seed-0 adapter is plain "lora"."""
    return f"lora_s{seed}" if system == "lora" and seed else system


def cmd_run(args) -> None:
    name = system_name(args.system, args.seed)
    out = rg.DATA_DIR / "runs" / f"{name}_{args.split}.jsonl"
    if args.split != "dev" and out.exists():
        raise SystemExit(f"{out} exists: the sealed set is run once per system")
    items = (r16_sealed.load_sealed(b=True) if args.split == "sealed-b"
             else r16_sealed.load_sealed(c=True) if args.split == "sealed-c"
             else r16_sealed.load_dev())  # fmt: skip
    net, tok = rg.load_model(adapter_path(args.seed) if args.system == "lora" else None)
    net.eval()
    tr = rg.translator(net, tok, f"{rg.BASE}+{name}")
    rows = rg.run_items(tr, items, f"{name} {args.split}", log)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    log(f"{name} {args.split}: {json.dumps(summarise(rows))}")


def summarise(rows: list[dict]) -> dict:
    from genetic_coil_arena.translator.metrics import wilson

    by = collections.defaultdict(collections.Counter)
    for r in rows:
        by[r["expect"]][r["outcome"]] += 1
    n_ok = sum(by["ok"].values())
    wrong = sum(c["wrong command"] for c in by.values())
    return {"n": len(rows), "by_expect": {k: dict(v) for k, v in by.items()},
            "exact": list(wilson(by["ok"]["exact"], n_ok)),
            "asked_when_unclear": list(wilson(by["clarify"]["asked"],
                                              sum(by["clarify"].values()))),
            "refused_when_should": list(wilson(by["refuse"]["refused"],
                                               sum(by["refuse"].values()))),
            "wrong_commands": list(wilson(wrong, len(rows))),
            "correct": list(wilson(sum(r["correct"] for r in rows), len(rows))),
            "median_latency_s": sorted(r["latency_s"] for r in rows)[len(rows) // 2]}  # fmt: skip


def rows_of(system: str, split: str) -> list[dict] | None:
    p = rg.DATA_DIR / "runs" / f"{system}_{split}.jsonl"
    if not p.exists():
        return None
    return [json.loads(x) for x in p.read_text("utf-8").splitlines() if x.strip()]


def mcnemar(a: list[dict], b: list[dict]) -> dict:
    """Exact two-sided McNemar on the ok items' exact matches (a vs b, paired by id)."""
    from math import comb

    eb = {r["id"]: r["outcome"] == "exact" for r in b if r["expect"] == "ok"}
    pairs = [(r["outcome"] == "exact", eb[r["id"]]) for r in a if r["expect"] == "ok"]
    n10 = sum(x and not y for x, y in pairs)
    n01 = sum(y and not x for x, y in pairs)
    n = n10 + n01
    p = min(1.0, 2 * sum(comb(n, k) for k in range(min(n10, n01) + 1)) / 2**n) if n else 1.0
    return {"a_only": n10, "b_only": n01, "p": p}


def cmd_report(_args) -> None:
    rep: dict = {"bars": BARS, "systems": {}}
    fit = rg.DATA_DIR / "models" / "lora_s0" / "fit.json"
    if fit.exists():
        rep["fit"] = json.loads(fit.read_text("utf-8"))
    data = rg.DATA_DIR / "data.json"
    if data.exists():
        d = json.loads(data.read_text("utf-8"))
        rep["data"] = {"written": d["written"], "label_problems": len(d["label_problems"]),
                       "leaks": len(d["leaks"]), "kept": d["kept"]}  # fmt: skip
    for system in ("asis", "lora"):
        for split in ("dev", "sealed-b"):
            rows = rows_of(system, split)
            if rows is not None:
                rep["systems"][f"{system}_{split}"] = summarise(rows)
    if G14B.exists():
        g = json.loads(G14B.read_text("utf-8"))
        rep["g14b"] = g
    lora, asis = rows_of("lora", "sealed-b"), rows_of("asis", "sealed-b")
    if lora is not None:
        s = rep["systems"]["lora_sealed-b"]
        ok = (s["exact"][0] >= BARS["exact"] and s["wrong_commands"][0] <= BARS["wrong"]
              and s["wrong_commands"][2] <= BARS["wrong_hi"])  # fmt: skip
        kill = s["exact"][0] < 0.80 or s["wrong_commands"][0] > 0.10
        rep["verdict"] = "PASS" if ok else ("KILL" if kill else "NOT PASSED")
        if asis is not None:
            gain = s["exact"][0] - rep["systems"]["asis_sealed-b"]["exact"][0]
            rep["editing"] = {"gain": gain, "mcnemar": mcnemar(lora, asis),
                              "matters": bool(round(gain, 9) >= EDIT_POINTS)}  # fmt: skip
            rep["editing"]["matters"] &= rep["editing"]["mcnemar"]["p"] < 0.05
        if Q14_ROWS.exists():  # measured with no bar: the deployed 14B on the same items
            q14 = [json.loads(x) for x in Q14_ROWS.read_text("utf-8").splitlines() if x.strip()]
            qc = {r["id"]: r["correct"] for r in q14}
            a = sum(r["correct"] and not qc[r["id"]] for r in lora)
            b = sum(qc[r["id"]] and not r["correct"] for r in lora)
            p = min(1.0, 2 * sum(math.comb(a + b, k) for k in range(min(a, b) + 1)) / 2 ** (a + b))
            rep["vs_14b"] = {"exact": mcnemar(lora, q14),
                             "correct": {"a_only": a, "b_only": b, "p": p}}  # fmt: skip
        rep["misses"] = [{k: r[k] for k in ("id", "text", "expect", "gold", "params",
                                            "outcome", "reason")}
                         for r in lora if not r["correct"]]  # fmt: skip
    Path("docs/results/r21_ground_lora.json").write_text(json.dumps(rep, indent=1), "utf-8")
    Path("docs/results/r21_ground_lora.md").write_text(render(rep), "utf-8")
    print(render(rep))


def pct(w) -> str:
    return f"{100 * w[0]:.1f}% [{100 * w[1]:.0f}, {100 * w[2]:.0f}]"


def render(rep: dict) -> str:
    lines = ["# R21: the ground station's language model with its weights edited", ""]
    if "verdict" in rep:
        s = rep["systems"]["lora_sealed-b"]
        lines += [f"**{rep['verdict']}.** Qwen2.5-Coder-7B with QLoRA, on G14b's sealed set "
                  f"(150 requests): exact {pct(s['exact'])} (bar >= 90%), wrong commands "
                  f"{pct(s['wrong_commands'])} (bar <= 3%, upper <= 6%)."]  # fmt: skip
        if "editing" in rep:
            e = rep["editing"]
            lines += ["", f"Editing: {100 * e['gain']:+.1f} points of exact match over the "
                      f"same 7B as is (McNemar p = {e['mcnemar']['p']:.3g}; "
                      f"{'meets' if e['matters'] else 'misses'} the >= 10-point bar)."]  # fmt: skip
        if "vs_14b" in rep:
            v = rep["vs_14b"]
            lines += ["", f"Against the deployed 14B on the same items (no bar): exact match "
                      f"{v['exact']['a_only']} items only the tuned 7B gets, "
                      f"{v['exact']['b_only']} only the 14B (p = {v['exact']['p']:.2f}); "
                      f"right outcome overall (exact, asked, refused as expected) "
                      f"{v['correct']['a_only']} against {v['correct']['b_only']} "
                      f"(p = {v['correct']['p']:.3f})."]  # fmt: skip
        lines.append("")
    if "data" in rep:
        d = rep["data"]
        lines += [f"Training requests: {d['written']} written by five writers from the "
                  f"label guide alone; {d['label_problems']} dropped by the label check, "
                  f"{d['leaks']} as too close to a dev or sealed text; {d['kept']} kept.", ""]  # fmt: skip
    if "fit" in rep:
        lines += ["| Epoch | Train loss | Dev exact | Dev wrong commands | Dev correct |",
                  "|---|---|---|---|---|"]  # fmt: skip
        for h in rep["fit"]["history"]:
            lines.append(f"| {h['epoch']} | {h['loss']:.4f} | {h['exact']}/{h['n_ok']} | "
                         f"{h['wrong']} | {h['correct']}/{h['n']} |")  # fmt: skip
        lines += ["", f"Chosen on dev: epoch {rep['fit']['chosen_epoch']}.", ""]
    lines += ["| System | Split | Exact (ok items) | Asked when unclear | Refused when "
              "should | Wrong commands | Median latency |", "|---|---|---|---|---|---|---|"]  # fmt: skip
    names = {"asis": "Qwen2.5-Coder-7B as is", "lora": "Qwen2.5-Coder-7B + QLoRA"}
    for key, s in rep["systems"].items():
        system, split = key.split("_", 1)
        lines.append(f"| {names[system]} | {split} | {pct(s['exact'])} | "
                     f"{pct(s['asked_when_unclear'])} | {pct(s['refused_when_should'])} | "
                     f"{pct(s['wrong_commands'])} | {s['median_latency_s']:.2f} s |")  # fmt: skip
    lines += ["", "G14b reference (qwen2.5-coder:14b as is, Ollama structured output, same "
              "sealed set): exact 88.9%, wrong commands 2.7%.", ""]  # fmt: skip
    if rep.get("misses"):
        lines += ["Sealed misses of the tuned model:", "", "| Id | Request | Expected | Got |",
                  "|---|---|---|---|"]  # fmt: skip
        for m in rep["misses"]:
            got = m["params"] if m["outcome"] in ("exact", "wrong command") else m["outcome"]
            lines.append(f"| {m['id']} | {m['text']} | {m['gold'] or m['expect']} | {got} |")
        lines.append("")
    lines += ["Reproduce: `python scripts/r21_ground_lora.py report`.", ""]
    return "\n".join(lines)


def passes(s: dict) -> bool:
    return bool(s["exact"][0] >= BARS["exact"] and s["wrong_commands"][0] <= BARS["wrong"]
                and s["wrong_commands"][2] <= BARS["wrong_hi"])  # fmt: skip


def cmd_report_b(_args) -> None:
    """R21b: the replicate seed's adapter on G14b's sealed set against seed 0's, G17's bars."""
    a, b = rows_of("lora", "sealed-b"), rows_of("lora_s1", "sealed-b")
    if a is None or b is None:
        raise SystemExit("both adapters' sealed runs are needed")
    fit = json.loads((rg.DATA_DIR / "models" / "lora_s1" / "fit.json").read_text("utf-8"))
    sa, sb = summarise(a), summarise(b)
    rep = {"bars": BARS, "seed0": sa, "seed1": sb, "seed1_fit": fit,
           "seed1_vs_seed0": mcnemar(b, a), "verdict": "PASS" if passes(sb) else "NOT PASSED",
           "seed1_misses": [{k: r[k] for k in ("id", "text", "expect", "gold", "params",
                                               "outcome", "reason")}
                            for r in b if not r["correct"]]}  # fmt: skip
    pct = lambda w: f"{100 * w[0]:.1f}% [{100 * w[1]:.0f}, {100 * w[2]:.0f}]"  # noqa: E731
    m = rep["seed1_vs_seed0"]
    lines = ["# R21b: the ground station's QLoRA edit with a second training seed", "",
             f"**{rep['verdict']}** (G17's bars: exact >= 90% of the ok items, wrong commands "
             "<= 3% with the Wilson upper bound <= 6%). G14b's sealed set, run once per adapter; "
             "same training data, recipe and dev epoch rule, only the seed differs.", "",
             "| Adapter | Exact (ok items) | Wrong commands | Asked when unclear | "
             "Refused when it should | Correct overall |", "|---|---|---|---|---|---|"]  # fmt: skip
    for name, s in (("seed 0 (R21)", sa), ("seed 1", sb)):
        lines.append(f"| {name} | {pct(s['exact'])} | {pct(s['wrong_commands'])} | "
                     f"{pct(s['asked_when_unclear'])} | {pct(s['refused_when_should'])} | "
                     f"{pct(s['correct'])} |")  # fmt: skip
    lines += ["", f"Seed 1 chose epoch {fit['chosen_epoch']} on dev. Paired exact matches: seed 1 "
              f"only {m['a_only']}, seed 0 only {m['b_only']} (exact McNemar p = {m['p']:.3g}).",
              "", "Reproduce: `python scripts/r21_ground_lora.py report-b`.", ""]  # fmt: skip
    out = Path("docs/results/r21b_replicate.md")
    out.write_text("\n".join(lines), "utf-8")
    out.with_suffix(".json").write_text(json.dumps(rep, indent=1), "utf-8")
    log(f"R21b: {rep['verdict']}")


Q14_C_ROWS = rg.DATA_DIR / "runs" / "q14" / "r16_sealed-c_ollama_qwen2.5-coder_14b.jsonl"
C_SYSTEMS = (("lora", "seed 0 (R21)"), ("lora_s1", "seed 1 (R21b)"), ("asis", "7B as is"))


def cmd_report_c(_args) -> None:
    """R21c: both QLoRA adapters on G14c's fresh sealed set, G17's bars for each; the 7B and
    the 14B as is measured with no bar. No request text or param is stored or printed: the
    set stays sealed for the next change."""
    rows = {name: rows_of(name, "sealed-c") for name, _ in C_SYSTEMS}
    if Q14_C_ROWS.exists():
        rows["q14"] = [json.loads(x) for x in Q14_C_ROWS.read_text("utf-8").splitlines()
                       if x.strip()]  # fmt: skip
    if rows["lora"] is None or rows["lora_s1"] is None:
        raise SystemExit("both adapters' G14c runs are needed")
    sums = {name: summarise(r) for name, r in rows.items() if r is not None}
    kill = any(sums[a]["exact"][0] < 0.80 or sums[a]["wrong_commands"][0] > 0.10
               for a in ("lora", "lora_s1"))  # fmt: skip
    verdict = ("PASS" if passes(sums["lora"]) and passes(sums["lora_s1"])
               else "KILL" if kill else "NOT PASSED")  # fmt: skip
    paired = {f"{a}_vs_{b}": mcnemar(rows[a], rows[b])
              for a, b in (("lora_s1", "lora"), ("lora_s1", "asis"), ("lora_s1", "q14"))
              if rows.get(b) is not None}  # fmt: skip
    rep = {"bars": BARS, "verdict": verdict, "systems": sums, "paired_exact": paired,
           "outcomes": {name: dict(collections.Counter(
               f"{x['expect']} -> {x['outcome']}" for x in r if not x["correct"]))
               for name, r in rows.items() if r is not None},
           "miss_ids": {name: [x["id"] for x in r if not x["correct"]]
                        for name, r in rows.items() if r is not None}}  # fmt: skip
    names = dict(C_SYSTEMS) | {"q14": "14B as is (deployed before R21)"}
    lines = ["# R21c: the ground station's QLoRA edit on a fresh sealed set (G14c)", "",
             f"**{verdict}** (G17's bars for each adapter: exact >= 90% of the ok items, wrong "
             "commands <= 3% with the Wilson upper bound <= 6%; KILL if exact < 80% or wrong "
             "commands > 10%). G14c: 150 requests (90 / 30 / 30) written by a separate agent "
             "from G14b's label guide alone, after both adapters were trained; run once per "
             "system.", "",
             "| System | Exact (ok items) | Wrong commands | Asked when unclear | "
             "Refused when it should | Correct overall | Passes |", "|---|---|---|---|---|---|---|"]  # fmt: skip
    for name, sm in sums.items():
        mark = ("yes" if passes(sm) else "no") if name.startswith("lora") else "(no bar)"
        lines.append(f"| {names[name]} | {pct(sm['exact'])} | {pct(sm['wrong_commands'])} | "
                     f"{pct(sm['asked_when_unclear'])} | {pct(sm['refused_when_should'])} | "
                     f"{pct(sm['correct'])} | {mark} |")  # fmt: skip
    lines += ["", "Paired exact matches on the ok items (exact McNemar):", ""]
    for key, m in paired.items():
        a, b = key.split("_vs_")
        lines.append(f"- {names[a]} only {m['a_only']}, {names[b]} only {m['b_only']} "
                     f"(p = {m['p']:.3g}).")  # fmt: skip
    lines += ["", "Misses by kind (expected -> got):", ""]
    for name, kinds in rep["outcomes"].items():
        lines.append(f"- {names[name]}: " + (", ".join(f"{k} {v}" for k, v in sorted(kinds.items()))
                                             or "none"))  # fmt: skip
    lines += ["", "Request texts are not shown, so the set can test the next change. "
              "Reproduce: `python scripts/r21_ground_lora.py report-c`.", ""]  # fmt: skip
    out = Path("docs/results/r21c_fresh_sealed.md")
    out.write_text("\n".join(lines), "utf-8")
    out.with_suffix(".json").write_text(json.dumps(rep, indent=1), "utf-8")
    log(f"R21c: {verdict}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data")
    sub.add_parser("audit")
    f = sub.add_parser("fit")
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--epochs", type=int, default=3)
    r = sub.add_parser("run")
    r.add_argument("--system", choices=("asis", "lora"), required=True)
    r.add_argument("--split", choices=("dev", "sealed-b", "sealed-c"), required=True)
    r.add_argument("--seed", type=int, default=0, help="the LoRA's training seed")
    sub.add_parser("report")
    sub.add_parser("report-b")
    sub.add_parser("report-c")
    args = p.parse_args()
    {"data": cmd_data, "audit": cmd_audit, "fit": cmd_fit, "run": cmd_run, "report": cmd_report,
     "report-b": cmd_report_b, "report-c": cmd_report_c}[args.cmd](args)  # fmt: skip


if __name__ == "__main__":
    main()
