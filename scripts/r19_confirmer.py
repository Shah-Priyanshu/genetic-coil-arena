"""R19: an open-weight vision model as the confirmer (experiments/r19_confirmer.py).

python scripts/r19_confirmer.py spots                  # the 400 R15e/R15f operator spots
python scripts/r19_confirmer.py crops --split val      # then train, test
python scripts/r19_confirmer.py fit --model dino-full --seed 0
python scripts/r19_confirmer.py score --model dino-full --seed 0 --split val|test|spots
python scripts/r19_confirmer.py report

Models: dino-probe, dino-full (DINOv2-small), qwen-zero, qwen-lora (Qwen3-VL-2B).
Everything is written under ``D:/Projects/genetic-coil-arena-data/r19_confirmer``;
``report`` writes ``docs/results/r19_confirmer.{md,json}``. Test-dev and the operator
spots are scored once per model, after the read-out was committed.
"""

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HOME", r"D:\hf_cache")

from genetic_coil_arena.experiments import r15_scout_brain as r15  # noqa: E402
from genetic_coil_arena.experiments import r15b_aerial_scout as r15b  # noqa: E402
from genetic_coil_arena.experiments import r19_confirmer as rc  # noqa: E402

MODELS = ("dino-probe", "dino-full", "qwen-zero", "qwen-lora")
SPLIT_OF = {"val": "val", "test": "test", "spots": "spots"}


def log(m: str) -> None:
    print(m, flush=True)


def r15f_module():
    spec = importlib.util.spec_from_file_location(
        "r15f_careful_crops", Path(__file__).with_name("r15f_careful_crops.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def answers(folder: Path) -> dict[str, str]:
    return {
        p.stem: json.loads(p.read_text("utf-8"))["label"] for p in folder.glob("*.json")
    }


def cmd_spots(_args) -> None:
    """R15e's spots re-drawn exactly as r15f_careful_crops.py re-draws them, R15f's from
    its truth file; each with R15b's label, the visual check and the operator's answer.
    """
    import torch

    f15 = r15f_module()
    val = r15b.load_tiles(r15.DATA_DIR / "visdrone_tiles224", "val")
    native = np.flatnonzero(np.asarray(val["scale"]) == 1.0)
    origins, members, _ = f15.tile_origins()
    model = r15.build_model(0.5)
    model.load_state_dict(torch.load(rc.WEIGHTS, map_location="cpu"))
    rng = np.random.default_rng(1519)
    pick, flat, _, table, t1, f1 = f15.draw(rng, val, native, model, 4000, 0.5)
    old = np.concatenate([rng.choice(t1, 100, replace=False),
                          rng.choice(f1, 100, replace=False)])  # fmt: skip
    rng.shuffle(old)  # r15e_operator_crops.py numbers the spots after this shuffle
    e_truth = json.loads(
        (r15.DATA_DIR / "r15e_operator" / "truth.json").read_text("utf-8")
    )
    spots = []
    for n, d in enumerate(old):
        x, y, w, h, score = flat[d]
        f, s, x0, y0 = origins[int(pick[int(table["det_tile"][d])])]
        t = e_truth[n]
        assert t["id"] == f"c{n:03d}" and abs(t["score"] - float(score)) < 1e-5
        spots.append({"id": t["id"], "session": "r15e", "frame": members[f].rsplit("/", 1)[-1],
                      "fx": float(x0 + x + w / 2), "fy": float(y0 + y + h / 2),
                      "score": float(score), "matched": t["person"]})  # fmt: skip
    for t in json.loads(
        (r15.DATA_DIR / "r15f_careful" / "truth.json").read_text("utf-8")
    ):
        spots.append({"id": t["id"], "session": "r15f", "frame": t["frame"], "fx": t["fx"],
                      "fy": t["fy"], "score": t["score"], "matched": t["person"]})  # fmt: skip
    adj = {}
    for name in ("r15e", "r15f"):
        a = json.loads(
            Path(f"docs/results/{name}_adjudication.json").read_text("utf-8")
        )
        for k in ("person", "not_person", "ambiguous"):
            adj.update(dict.fromkeys(a[k], k))
    ops = {**answers(r15.DATA_DIR / "r15e_operator" / "answers_raw" / "answers"),
           **answers(r15.DATA_DIR / "r15f_careful" / "answers_raw" / "answers")}  # fmt: skip
    for s in spots:
        s["check"] = "matched" if s["matched"] else adj[s["id"]]
        s["operator"] = ops[s["id"]]
    want = {s["frame"] for s in spots}
    rc.OUT_DIR.mkdir(parents=True, exist_ok=True)
    imgs = {}
    for _, name, img, _ in rc.frames("val"):
        if name in want:
            imgs[name] = img
    with open(rc.OUT_DIR / "spots_crops.u8", "wb") as fh:
        for s in spots:
            fh.write(rc.cut(imgs[s["frame"]], s["fx"], s["fy"]).tobytes())
    rc.save_json(rc.OUT_DIR / "spots.json", spots)
    counts = {}
    for s in spots:
        counts[s["check"]] = counts.get(s["check"], 0) + 1
    log(f"{len(spots)} spots: {counts}; operator labels "
        f"{sorted({s['operator'] for s in spots})}")  # fmt: skip


def load_spots() -> tuple[np.ndarray, list[dict]]:
    spots = json.loads((rc.OUT_DIR / "spots.json").read_text("utf-8"))
    crops = np.memmap(rc.OUT_DIR / "spots_crops.u8", np.uint8, "r",
                      shape=(len(spots), 2 * rc.HALF, 2 * rc.HALF, 3))  # fmt: skip
    return crops, spots


def cmd_crops(args) -> None:
    reserved = None
    if args.split == "val":
        reserved = {}
        for s in json.loads((rc.OUT_DIR / "spots.json").read_text("utf-8")):
            reserved.setdefault(s["frame"], []).append((s["fx"], s["fy"]))
    train = args.split == "train"
    rc.build(args.split, rc.TRAIN_THRESHOLD if train else rc.EVAL_THRESHOLD,
             cap=rc.TRAIN_CAP if train else 0, reserved_spots=reserved, log=log)  # fmt: skip


def selection():
    vc, vm = rc.load("val")
    sel = ~vm["reserved"]
    return np.asarray(vc[sel]), vm["person"][sel]


def model_path(model: str, seed: int, tag: str = "") -> Path:
    return rc.OUT_DIR / "models" / f"{model}{tag}_s{seed}"


def cmd_fit(args) -> None:
    start = time.perf_counter()
    crops, meta = rc.load("train")
    val = selection()
    log(f"train {len(crops)} candidates ({meta['person'].mean():.1%} person); "
        f"selection {len(val[0])} ({val[1].mean():.1%} person)")  # fmt: skip
    out = model_path(args.model, args.seed, args.tag)
    out.mkdir(parents=True, exist_ok=True)
    if args.model.startswith("dino"):
        import torch

        m, hist = rc.train_dino(args.model.split("-")[1], crops, meta["person"], val,
                                args.seed, epochs=args.epochs, size=args.size or 224,
                                log=log)  # fmt: skip
        torch.save({"net": m.net.state_dict(), "head": m.head.state_dict(), "size": m.size},
                   out / "model.pt")  # fmt: skip
    elif args.model == "qwen-lora":
        q = rc.QwenConfirmer(rc._gpu(), size=args.size or 448)
        hist = rc.train_lora(q, crops, meta["person"], val, args.seed, steps=args.steps,
                             log=log)  # fmt: skip
        q.net.save_pretrained(out / "adapter")
    else:
        raise SystemExit("qwen-zero has nothing to fit; use score")
    rc.save_json(out / "fit.json", {"model": args.model, "seed": args.seed, "history": hist,
                                    "seconds": time.perf_counter() - start,
                                    "size": args.size})  # fmt: skip


def confirmer(model: str, seed: int, size: int | None, tag: str = ""):
    import torch

    dev = rc._gpu()
    if model.startswith("dino"):
        state = torch.load(
            model_path(model, seed, tag) / "model.pt", map_location="cpu"
        )
        m = rc.DinoConfirmer(dev, state.get("size", 224))
        m.net.load_state_dict(state["net"])
        m.head.load_state_dict(state["head"])
        return m
    q = rc.QwenConfirmer(dev, size=size or 448)
    if model == "qwen-lora":
        from peft import PeftModel

        q.net = PeftModel.from_pretrained(
            q.net, model_path(model, seed, tag) / "adapter"
        )
    return q


def cmd_score(args) -> None:
    start = time.perf_counter()
    out = (
        rc.OUT_DIR / "scores" / f"{args.model}{args.tag}_s{args.seed}_{args.split}.npy"
    )
    if args.split in ("test", "spots") and out.exists():
        raise SystemExit(f"{out} exists: {args.split} is scored once per model")
    if args.split == "spots":
        crops, _ = load_spots()
    else:
        crops, _ = rc.load(args.split)
    if args.limit:
        crops = crops[: args.limit]
    m = confirmer(args.model, args.seed, args.size, args.tag)
    s = m.score(np.asarray(crops) if len(crops) < 50_000 else crops)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not args.limit:
        np.save(out, s)
    log(
        f"{args.model} seed {args.seed} {args.split}: {len(s)} scores in "
        f"{time.perf_counter() - start:.0f} s -> {out if not args.limit else '(limit, not saved)'}"
    )
    if args.split == "val":
        _, vm = rc.load("val")
        sel = ~vm["reserved"][: len(s)]
        p = vm["person"][: len(s)][sel]
        thr = rc.threshold_at(s[sel][p])
        log(f"  selection AUC {rc.auc(s[sel][p], s[sel][~p]):.3f}, false alarms passed at "
            f"{rc.KEEP:.0%} keep {(s[sel][~p] >= thr).mean():.1%}")  # fmt: skip


EDITED = {"dino-full": "dino-probe", "qwen-lora": "qwen-zero"}
NAMES = {"dino-probe": "DINOv2-small, weights kept (linear probe)",
         "dino-full": "DINOv2-small, fine-tuned",
         "qwen-zero": "Qwen3-VL-2B, as is (asked)",
         "qwen-lora": "Qwen3-VL-2B, LoRA-tuned"}  # fmt: skip


def scores(model: str, seed: int, split: str) -> np.ndarray | None:
    p = rc.OUT_DIR / "scores" / f"{model}_s{seed}_{split}.npy"
    return np.load(p) if p.exists() else None


def cmd_report(_args) -> None:
    _, vm = rc.load("val")
    sel = ~vm["reserved"]
    vp = vm["person"][sel]
    _, tm = rc.load("test")
    tp = tm["person"]
    boot = rc.frame_bootstrap(tm["frame"])
    _, spots = load_spots()
    check = np.array([s["check"] for s in spots])
    op_yes = np.array([s["operator"] == "person" for s in spots])
    systems = []
    for p in sorted((rc.OUT_DIR / "scores").glob("*_val.npy")):
        model, seed = p.name[: -len("_val.npy")].rsplit("_s", 1)
        base = max((m for m in MODELS if model.startswith(m)), key=len)
        tag = model[len(base) :]
        sv = np.load(p)[sel]
        thr = rc.threshold_at(sv[vp])
        rec = {"model": model, "base": base, "tag": tag, "seed": int(seed), "threshold": thr,
               "name": NAMES[base] + (f" ({tag.strip('-')})" if tag else ""),
               "val": {"auc": rc.auc(sv[vp], sv[~vp]),
                       "pass": float((sv[~vp] >= thr).mean())}}  # fmt: skip
        st = scores(model, int(seed), "test")
        if st is not None:
            keep = st >= thr
            rec["test"] = {**rc.rates(st, tp, thr), "auc": rc.auc(st[tp], st[~tp]),
                           "pass_boot": ((boot * (keep & ~tp)).sum(1)
                                         / (boot * ~tp).sum(1)).tolist(),
                           "bins": {name: rc.rates(st[m], tp[m], thr) for name, lo, hi
                                    in r15b.BIN_NAMES[1:]
                                    for m in [(tm["size"] >= lo) & (tm["size"] < hi)]}}  # fmt: skip
        ss = scores(model, int(seed), "spots")
        if ss is not None:
            keep = ss >= thr
            rec["spots"] = {
                c: rc.wilson(int(keep[check == c].sum()), int((check == c).sum()))
                for c in ("matched", "person", "not_person", "ambiguous")
            }
            both = keep & op_yes  # the model passes it and the operator says "person"
            rec["chain"] = {
                c: rc.wilson(int(both[check == c].sum()), int((check == c).sum()))
                for c in ("matched", "person", "not_person", "ambiguous")
            }
        systems.append(rec)
    operator = {c: rc.wilson(int(op_yes[check == c].sum()), int((check == c).sum()))
                for c in ("matched", "person", "not_person", "ambiguous")}  # fmt: skip
    edited = [s for s in systems if s["base"] in EDITED]
    chosen = min(edited, key=lambda s: s["val"]["pass"]) if edited else None
    verdict, ratios = "INCOMPLETE", {}
    for s in edited:
        # the unedited model at the same setting (input size, epochs)
        base = next((b for b in systems if b["base"] == EDITED[s["base"]]
                     and b["tag"] == s["tag"] and "test" in b), None)  # fmt: skip
        if base is None or "test" not in s:
            continue
        r = np.array(s["test"]["pass_boot"]) / np.array(base["test"]["pass_boot"])
        ratios[f"{s['model']}_s{s['seed']}"] = {
            "vs": f"{base['model']}_s{base['seed']}",
            "ratio": s["test"]["pass"][0] / base["test"]["pass"][0],
            "lo": float(np.percentile(r, 2.5)), "hi": float(np.percentile(r, 97.5))}  # fmt: skip
    if chosen is not None and "spots" in chosen:
        fa = chosen["spots"]["not_person"][0]
        hit = chosen["spots"]["matched"][0]
        if fa >= 0.30:
            verdict = "KILL"
        elif fa <= 0.10 and hit >= 0.90:
            verdict = "PASS"
        else:
            verdict = "NOT PASSED"
        r = ratios.get(f"{chosen['model']}_s{chosen['seed']}")
        editing = None if r is None else bool(r["ratio"] <= 0.5 and r["hi"] < 1)
    else:
        editing = None
    rep = {"systems": systems, "operator": operator, "chosen": chosen and
           f"{chosen['model']}_s{chosen['seed']}", "verdict": verdict,
           "editing_matters": editing, "ratios": ratios, "keep": rc.KEEP,
           "counts": {"val_selection": int(sel.sum()), "val_person": float(vp.mean()),
                      "test": len(tp), "test_person": float(tp.mean()),
                      "test_frames": int(len(np.unique(tm["frame"]))),
                      "spots": {c: int((check == c).sum()) for c in set(check)}}}  # fmt: skip
    for s in systems:
        s.get("test", {}).pop("pass_boot", None)
    out = Path("docs/results/r19_confirmer.json")
    rc.save_json(out, rep)
    out.with_suffix(".md").write_text(render(rep), encoding="utf-8")
    print(render(rep))


def pct(w) -> str:
    return f"{100 * w[0]:.1f}% [{100 * w[1]:.0f}, {100 * w[2]:.0f}]"


def render(rep: dict) -> str:
    c = rep["counts"]
    editing = ""
    if rep["editing_matters"] is not None:
        r = rep["ratios"][rep["chosen"]]
        editing = (f"; editing the weights {'matters' if rep['editing_matters'] else 'misses its bar'}: "
                   f"the chosen model passes {r['ratio']:.2f}× [{r['lo']:.2f}, {r['hi']:.2f}] of the "
                   "test-dev false alarms the same model passes unedited (bar ≤ 0.5×, upper bound < 1)")  # fmt: skip
    lines = ["# R19: an open-weight confirmer for the scout's detections (goal G15)", "",
             f"**Verdict: {rep['verdict']}**{editing}",
             "",
             "Candidates are R15b's width-0.5 brain's detections at confidence ≥ 0.5 on VisDrone-DET "
             "frames at native scale; each model sees the 144-px view around the detection. Every "
             f"threshold keeps {100 * rep['keep']:.0f}% of the people in the val selection set "
             f"({c['val_selection']} candidates, {100 * c['val_person']:.0f}% people). Test-dev: "
             f"{c['test']} candidates from {c['test_frames']} frames, {100 * c['test_person']:.0f}% "
             f"people (VisDrone labels). Chosen model (lowest val pass among the edited ones): "
             f"`{rep['chosen']}`.", "",
             "## Val (selection) and test-dev, VisDrone labels", "",
             "| Model | Seed | Val AUC | Val false alarms passed | Test AUC | Test people kept | Test false alarms passed |",
             "|---|---|---|---|---|---|---|"]  # fmt: skip
    for s in rep["systems"]:
        t = s.get("test")
        lines.append(f"| {s['name']} | {s['seed']} | {s['val']['auc']:.3f} | "
                     f"{100 * s['val']['pass']:.1f}% | "
                     + (f"{t['auc']:.3f} | {pct(t['keep'])} | {pct(t['pass'])} |" if t
                        else "– | – | – |"))  # fmt: skip
    if rep["ratios"]:
        lines += ["", "Edited against the same model unedited (test-dev false alarms passed, "
                  "frame-bootstrap 95% interval):", ""]  # fmt: skip
        for k, r in rep["ratios"].items():
            lines.append(
                f"- `{k}` vs `{r['vs']}`: {r['ratio']:.2f}× [{r['lo']:.2f}, {r['hi']:.2f}]"
            )
    sp = c["spots"]
    lines += ["", "## The 400 operator spots (R15e and R15f, val)", "",
              f"Claude's visual checks: {sp.get('matched', 0)} matched people, "
              f"{sp.get('person', 0)} unlabelled people, {sp.get('not_person', 0)} clear "
              f"non-persons, {sp.get('ambiguous', 0)} ambiguous. \"Chained\" = the model passes "
              "the spot and the operator said person.", "",
              "| Who | Matched people kept | Clear non-persons passed | Ambiguous passed | Unlabelled people kept |",
              "|---|---|---|---|---|"]  # fmt: skip
    o = rep["operator"]
    lines.append(f"| Operator alone | {pct(o['matched'])} | {pct(o['not_person'])} | "
                 f"{pct(o['ambiguous'])} | {pct(o['person'])} |")  # fmt: skip
    for s in rep["systems"]:
        for key, who in (("spots", s["name"]), ("chain", f"{s['name']} + operator")):
            if key in s:
                v = s[key]
                lines.append(f"| {who} (seed {s['seed']}) | {pct(v['matched'])} | "
                             f"{pct(v['not_person'])} | {pct(v['ambiguous'])} | "
                             f"{pct(v['person'])} |")  # fmt: skip
    for s in rep["systems"]:
        if "test" in s:
            lines += ["", f"### {s['name']} (seed {s['seed']}) on test-dev by detection size", "",
                      "| Size (px) | People kept | False alarms passed |", "|---|---|---|"]  # fmt: skip
            for b, v in s["test"]["bins"].items():
                lines.append(f"| {b} | {pct(v['keep'])} | {pct(v['pass'])} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("spots")
    c = sub.add_parser("crops")
    c.add_argument("--split", choices=("train", "val", "test"), required=True)
    f = sub.add_parser("fit")
    f.add_argument("--model", choices=MODELS, required=True)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--epochs", type=int, default=3)
    f.add_argument("--steps", type=int, default=1500)
    f.add_argument(
        "--size", type=int, default=None, help="input pixels (Qwen 448, DINOv2 224)"
    )
    f.add_argument("--tag", default="", help="variant suffix, e.g. -e10")
    s = sub.add_parser("score")
    s.add_argument("--model", choices=MODELS, required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--split", choices=sorted(SPLIT_OF), required=True)
    s.add_argument("--size", type=int, default=None, help="Qwen input pixels (448)")
    s.add_argument("--tag", default="", help="variant suffix, e.g. -e10")
    s.add_argument("--limit", type=int, default=0, help="first N only, not saved")
    sub.add_parser("report")
    args = p.parse_args()
    {"spots": cmd_spots, "crops": cmd_crops, "fit": cmd_fit, "score": cmd_score,
     "report": cmd_report}[args.cmd](args)  # fmt: skip


if __name__ == "__main__":
    main()
