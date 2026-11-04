"""R21: the ground station's LLM with its weights edited (experiments/r21_ground_lora.py).

python scripts/r21_ground_lora.py data                     # check labels and leaks, write train.jsonl
python scripts/r21_ground_lora.py audit                    # the 14B on every training label
python scripts/r21_ground_lora.py data                     # again, with label_drops.json
python scripts/r21_ground_lora.py fit --seed 0             # QLoRA, dev after each epoch
python scripts/r21_ground_lora.py run --system asis|lora --split dev|sealed-b
python scripts/r21_ground_lora.py report

Everything is written under ``D:/Projects/genetic-coil-arena-data/r21_ground_lora``;
``report`` writes ``docs/results/r21_ground_lora.{md,json}``. The sealed set is run once
per system, after the read-out was committed.
"""

import argparse
import collections
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", r"D:\hf_cache")

from genetic_coil_arena.experiments import r16_sealed  # noqa: E402
from genetic_coil_arena.experiments import r21_ground_lora as rg  # noqa: E402

G14B = Path("docs/results/r16b_ground_station.json")
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


def cmd_run(args) -> None:
    out = rg.DATA_DIR / "runs" / f"{args.system}_{args.split}.jsonl"
    if args.split != "dev" and out.exists():
        raise SystemExit(f"{out} exists: the sealed set is run once per system")
    items = r16_sealed.load_sealed(b=True) if args.split == "sealed-b" else r16_sealed.load_dev()
    net, tok = rg.load_model(adapter_path() if args.system == "lora" else None)
    net.eval()
    tr = rg.translator(net, tok, f"{rg.BASE}+{args.system}")
    rows = rg.run_items(tr, items, f"{args.system} {args.split}", log)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    log(f"{args.system} {args.split}: {json.dumps(summarise(rows))}")


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
                              "matters": bool(gain >= EDIT_POINTS)}  # fmt: skip
            rep["editing"]["matters"] &= rep["editing"]["mcnemar"]["p"] < 0.05
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
        lines.append("")
    if "data" in rep:
        d = rep["data"]
        lines += [f"Training requests: {d['written']} written by three writers from the "
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
    r.add_argument("--split", choices=("dev", "sealed-b"), required=True)
    sub.add_parser("report")
    args = p.parse_args()
    {"data": cmd_data, "audit": cmd_audit, "fit": cmd_fit, "run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
