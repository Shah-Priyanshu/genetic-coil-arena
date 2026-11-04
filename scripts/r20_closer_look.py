"""R20: a closer look against a second model on the same view (experiments/r20_closer_look.py).

python scripts/r20_closer_look.py crops --split train     # then val, test
python scripts/r20_closer_look.py fit --view far --seed 0  # far|close, seeds 0 and 1
python scripts/r20_closer_look.py score --view far --seed 0 --split val|test
python scripts/r20_closer_look.py report
python scripts/r20_closer_look.py sample                   # blind visual-check sheets
python scripts/r20_closer_look.py report                   # again, with the visual check

Everything is written under ``D:/Projects/genetic-coil-arena-data/r20_closer_look``;
``report`` writes ``docs/results/r20_closer_look.{md,json}``. Test-dev is scored once
per model, after the read-out was committed.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HOME", r"D:\hf_cache")

from genetic_coil_arena.experiments import r19_confirmer as r19  # noqa: E402
from genetic_coil_arena.experiments import r20_closer_look as rl  # noqa: E402

EPOCHS, SIZE = 10, 336  # R19's chosen recipe (dino-full-e10s336)
CHAINS = {
    "control": ("far", 0, "far", 1),
    "closer": ("far", 0, "close", 0),
    "closer_replicate": ("far", 1, "close", 1),
}
CHAIN_NAMES = {
    "control": "scout's view, checked by two models",
    "closer": "scout's view, then the closer look",
    "closer_replicate": "the same, other seeds",
}
VISUAL = Path("docs/results/r20_visual_check.json")
VISUAL_N = 150


def log(m: str) -> None:
    print(m, flush=True)


def model_dir(view: str, seed: int) -> Path:
    return rl.OUT_DIR / "models" / f"{view}_s{seed}"


def score_path(view: str, seed: int, split: str) -> Path:
    return rl.OUT_DIR / "scores" / f"{view}_s{seed}_{split}.npy"


def cmd_crops(args) -> None:
    train = args.split == "train"
    rl.build(args.split, r19.TRAIN_THRESHOLD if train else r19.EVAL_THRESHOLD,
             cap=r19.TRAIN_CAP if train else 0, log=log)  # fmt: skip


def cmd_fit(args) -> None:
    import torch

    start = time.perf_counter()
    crops, meta = rl.load("train", args.view)
    vc, vm = rl.load("val", args.view)
    log(f"{args.view}: train {len(crops)} candidates ({meta['person'].mean():.1%} "
        f"person); val {len(vc)} ({vm['person'].mean():.1%} person)")  # fmt: skip
    m, hist = r19.train_dino("full", crops, meta["person"], (np.asarray(vc), vm["person"]),
                             args.seed, epochs=EPOCHS, size=SIZE, log=log)  # fmt: skip
    out = model_dir(args.view, args.seed)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"net": m.net.state_dict(), "head": m.head.state_dict(), "size": m.size},
               out / "model.pt")  # fmt: skip
    r19.save_json(out / "fit.json", {"view": args.view, "seed": args.seed, "history": hist,
                                     "seconds": time.perf_counter() - start})  # fmt: skip


def cmd_score(args) -> None:
    import torch

    start = time.perf_counter()
    out = score_path(args.view, args.seed, args.split)
    if args.split == "test" and out.exists():
        raise SystemExit(f"{out} exists: test-dev is scored once per model")
    crops, meta = rl.load(args.split, args.view)
    state = torch.load(model_dir(args.view, args.seed) / "model.pt", map_location="cpu")
    m = r19.DinoConfirmer(r19._gpu(), state["size"])
    m.net.load_state_dict(state["net"])
    m.head.load_state_dict(state["head"])
    s = m.score(np.asarray(crops))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, s)
    log(f"{args.view} seed {args.seed} {args.split}: {len(s)} scores in "
        f"{time.perf_counter() - start:.0f} s")  # fmt: skip
    if args.split == "val":
        p = meta["person"]
        thr = r19.threshold_at(s[p])
        log(f"  val AUC {r19.auc(s[p], s[~p]):.3f}, false alarms passed at "
            f"{r19.KEEP:.0%} keep {(s[~p] >= thr).mean():.1%}")  # fmt: skip


def scores(view: str, seed: int, split: str) -> np.ndarray | None:
    p = score_path(view, seed, split)
    return np.load(p) if p.exists() else None


def wil(mask: np.ndarray, of: np.ndarray) -> list[float]:
    return list(r19.wilson(int((mask & of).sum()), int(of.sum())))


def cmd_report(_args) -> None:
    _, vm = rl.load("val", "far")
    _, tm = rl.load("test", "far")
    vp, tp = vm["person"], tm["person"]
    boot = r19.frame_bootstrap(tm["frame"])
    rep: dict = {"factor": rl.FACTOR, "keep": r19.KEEP,
                 "candidates": {"val": int(len(vp)), "test": int(len(tp)),
                                "test_people": int(tp.sum()),
                                "test_false": int((~tp).sum())},
                 "models": {}, "chains": {}}  # fmt: skip
    single: dict = {}
    for view in rl.VIEWS:
        for seed in (0, 1):
            sv, st = scores(view, seed, "val"), scores(view, seed, "test")
            if sv is None:
                continue
            thr = r19.threshold_at(sv[vp])
            rec = {"val_auc": r19.auc(sv[vp], sv[~vp]),
                   "val_pass": float((sv[~vp] >= thr).mean()), "threshold": thr}  # fmt: skip
            if st is not None:
                single[(view, seed)] = st >= thr
                rec |= {"test_auc": r19.auc(st[tp], st[~tp]),
                        "test_keep": wil(st >= thr, tp), "test_pass": wil(st >= thr, ~tp)}  # fmt: skip
            rep["models"][f"{view}_s{seed}"] = rec
    passed: dict = {}
    for name, (va, sa, vb, sb) in CHAINS.items():
        a, b = scores(va, sa, "val"), scores(vb, sb, "val")
        if a is None or b is None:
            continue
        ta, tb, q = rl.chain_thresholds(a, b, vp)
        rec = {"q": q, "val_keep": float(((a >= ta) & (b >= tb))[vp].mean()),
               "val_pass": float(((a >= ta) & (b >= tb))[~vp].mean())}  # fmt: skip
        at, bt = scores(va, sa, "test"), scores(vb, sb, "test")
        if at is not None and bt is not None:
            ok = (at >= ta) & (bt >= tb)
            passed[name] = ok
            rec |= {"test_keep": wil(ok, tp), "test_pass": wil(ok, ~tp)}
        rep["chains"][name] = rec
    if {"control", "closer"} <= passed.keys():
        neg = ~tp
        num, den = passed["closer"] & neg, passed["control"] & neg
        r = boot_ci(num, den, boot)
        rep["ratio"] = {"closer": r}
        if "closer_replicate" in passed:
            rep["ratio"]["closer_replicate"] = boot_ci(passed["closer_replicate"] & neg, den,
                                                       boot)  # fmt: skip
        keep_ok = rep["chains"]["closer"]["test_keep"][0] >= rl.KEEP_TEST
        rep_r = rep["ratio"].get("closer_replicate", [np.inf])[0]
        if r[0] <= rl.RATIO_PASS and r[2] < 1 and rep_r <= rl.RATIO_PASS and keep_ok:
            rep["verdict"] = "PASS"
        elif r[0] >= rl.RATIO_KILL:
            rep["verdict"] = "KILL"
        else:
            rep["verdict"] = "NOT PASSED"
        rep["overlap"] = overlap(tm, single)
    if VISUAL.exists() and passed:
        rep["visual"] = visual(passed, tp)
    r19.save_json(Path("docs/results/r20_closer_look.json"), rep)
    Path("docs/results/r20_closer_look.md").write_text(render(rep), "utf-8")
    log(render(rep))


def boot_ci(num: np.ndarray, den: np.ndarray, boot: np.ndarray) -> list[float]:
    b = rl.boot_ratio(num, den, boot)
    return [float(num.sum() / max(den.sum(), 1)), float(np.quantile(b, 0.025)),
            float(np.quantile(b, 0.975))]  # fmt: skip


def overlap(tm: dict, single: dict) -> dict:
    """How the second model's errors depend on the first's, on test-dev false alarms."""
    neg = ~tm["person"]
    out = {}
    if ("far", 0) not in single:
        return out
    first = single[("far", 0)] & neg
    for key, name in ((("far", 1), "far_s1"), (("close", 0), "close_s0")):
        if key not in single:
            continue
        second = single[key] & neg
        both = (first & second).sum() / neg.sum()
        indep = first.sum() / neg.sum() * second.sum() / neg.sum()
        out[name] = {"pass_alone": float(second.sum() / neg.sum()),
                     "pass_given_first_passed": float((first & second).sum() / first.sum()),
                     "lift_over_independent": float(both / indep)}  # fmt: skip
    for key, name in ((("far", 1), "far_s1"), (("close", 0), "close_s0")):
        sa, sb = scores("far", 0, "test"), scores(*key, "test")
        if sa is not None and sb is not None and name in out:
            out[name]["spearman_on_false_alarms"] = rl.spearman(sa[neg], sb[neg])
    return out


def visual_pool(passed: dict, tp: np.ndarray) -> np.ndarray:
    """Test-dev false alarms passed by the control or the closer-look chain, a fixed
    random sample of at most ``VISUAL_N``, in an order that hides the chain."""
    pool = np.flatnonzero((passed["control"] | passed["closer"]) & ~tp)
    rng = np.random.default_rng(2020)
    if len(pool) > VISUAL_N:
        pool = np.sort(rng.choice(pool, VISUAL_N, replace=False))
    return rng.permutation(pool)


def visual(passed: dict, tp: np.ndarray) -> dict:
    """Each chain's share of the visually clear non-persons it passes (Claude's blind
    check of the sample), against the sample's clear non-persons."""
    check = json.loads(VISUAL.read_text("utf-8"))  # sheet numbers per class, and the pool
    idx = np.array(check["index"])
    label = {n: k for k in ("person", "not_person", "ambiguous") for n in check[k]}
    lab = np.array([label[n] for n in range(len(idx))])
    out = {"sample": int(len(idx)), "counts": {k: int((lab == k).sum())
                                                for k in ("person", "not_person", "ambiguous")}}  # fmt: skip
    clear = lab == "not_person"
    for name in ("control", "closer"):
        out[name] = {"clear_non_persons_passed": int((passed[name][idx] & clear).sum()),
                     "persons_passed": int((passed[name][idx] & (lab == "person")).sum())}  # fmt: skip
    c, p = out["control"]["clear_non_persons_passed"], out["closer"]["clear_non_persons_passed"]
    out["clear_ratio"] = p / c if c else None
    return out


def cmd_sample(_args) -> None:
    """Contact sheets of the visual-check sample (native view, 2x), numbered in a random
    order, plus the index list; the chain each spot came from is not shown."""
    from PIL import Image, ImageDraw

    _, tm = rl.load("test", "far")
    tp = tm["person"]
    passed = {}
    _, vm = rl.load("val", "far")
    for name in ("control", "closer"):
        va, sa, vb, sb = CHAINS[name]
        ta, tb, _ = rl.chain_thresholds(scores(va, sa, "val"), scores(vb, sb, "val"),
                                        vm["person"])  # fmt: skip
        passed[name] = (scores(va, sa, "test") >= ta) & (scores(vb, sb, "test") >= tb)
    pool = visual_pool(passed, tp)
    close, _ = rl.load("test", "close")
    far, _ = rl.load("test", "far")
    out = rl.OUT_DIR / "visual"
    out.mkdir(parents=True, exist_ok=True)
    per = 20
    for sheet in range(0, len(pool), per):
        img = Image.new("RGB", (5 * 2 * 300, 4 * 310), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        for k, i in enumerate(pool[sheet : sheet + per]):
            x0, y0 = (k % 5) * 600, (k // 5) * 310
            img.paste(Image.fromarray(np.asarray(close[i])).resize((288, 288)), (x0, y0 + 20))
            img.paste(Image.fromarray(np.asarray(far[i])).resize((288, 288)),
                      (x0 + 296, y0 + 20))  # fmt: skip
            draw.text((x0 + 4, y0 + 4), f"#{sheet + k:03d}  close | scout", fill=(0, 0, 0))
        img.save(out / f"sheet_{sheet // per:02d}.png")
    r19.save_json(out / "pool.json", {"index": [int(i) for i in pool]})
    log(f"{len(pool)} spots on {(len(pool) + per - 1) // per} sheets in {out}")


def pct(w) -> str:
    return f"{100 * w[0]:.1f}% [{100 * w[1]:.0f}, {100 * w[2]:.0f}]"


def render(rep: dict) -> str:
    lines = ["# R20: a closer look from the confirming drone against a second model on the "
             "scout's view", ""]  # fmt: skip
    if "verdict" in rep:
        r = rep["ratio"]["closer"]
        lines += [f"**{rep['verdict']}.** On test-dev false alarms, the closer-look chain "
                  f"passes {r[0]:.2f}x [{r[1]:.2f}, {r[2]:.2f}] of what the control chain "
                  f"passes (bar: <= {rl.RATIO_PASS}x with the bootstrap upper bound below "
                  f"1, replicated, and >= {rl.KEEP_TEST:.0%} of test-dev people kept; kill "
                  f"at >= {rl.RATIO_KILL}x).", ""]  # fmt: skip
    c = rep["candidates"]
    lines += [f"Scout's view: VisDrone2019-DET reduced {rep['factor']}x (the scout twice as "
              f"high); closer look: native resolution. {c['val']} val and {c['test']} "
              f"test-dev candidates ({c['test_people']} people, {c['test_false']} false "
              f"alarms by VisDrone's labels). DINOv2-small fine-tuned per view (R19's "
              f"recipe), thresholds keep {rep['keep']:.0%} of val people.", "",
              "| Model | Val AUC | Val false alarms passed | Test-dev people kept | "
              "Test-dev false alarms passed |", "|---|---|---|---|---|"]  # fmt: skip
    for name, m in rep["models"].items():
        lines.append(f"| {name} | {m['val_auc']:.3f} | {100 * m['val_pass']:.1f}% | "
                     + (f"{pct(m['test_keep'])} | {pct(m['test_pass'])} |"
                        if "test_keep" in m else "- | - |"))  # fmt: skip
    lines += ["", "| Chain (both must pass) | Val people kept | Val false alarms passed | "
              "Test-dev people kept | Test-dev false alarms passed |",
              "|---|---|---|---|---|"]  # fmt: skip
    for name, ch in rep["chains"].items():
        lines.append(f"| {CHAIN_NAMES[name]} | {100 * ch['val_keep']:.1f}% | "
                     f"{100 * ch['val_pass']:.1f}% | "
                     + (f"{pct(ch['test_keep'])} | {pct(ch['test_pass'])} |"
                        if "test_keep" in ch else "- | - |"))  # fmt: skip
    if rep.get("overlap"):
        lines += ["", "Second model's errors on test-dev false alarms, against far seed 0's "
                  "(each model at its own 95% threshold):", "",
                  "| Second model | Passes alone | Passes when far s0 passed | Lift over "
                  "independent | Score rank correlation |", "|---|---|---|---|---|"]  # fmt: skip
        for name, o in rep["overlap"].items():
            lines.append(f"| {name} | {100 * o['pass_alone']:.1f}% | "
                         f"{100 * o['pass_given_first_passed']:.1f}% | "
                         f"{o['lift_over_independent']:.2f}x | "
                         f"{o.get('spearman_on_false_alarms', float('nan')):.2f} |")  # fmt: skip
    if "visual" in rep:
        v = rep["visual"]
        lines += ["", f"Blind visual check of {v['sample']} test-dev false alarms passed by "
                  f"either chain: {v['counts']['not_person']} clear non-persons, "
                  f"{v['counts']['person']} unlabelled people, {v['counts']['ambiguous']} "
                  f"ambiguous. Clear non-persons passed: control "
                  f"{v['control']['clear_non_persons_passed']}, closer look "
                  f"{v['closer']['clear_non_persons_passed']}"
                  + (f" ({v['clear_ratio']:.2f}x)." if v["clear_ratio"] is not None else ".")]  # fmt: skip
    lines += ["", "Reproduce: `python scripts/r20_closer_look.py report`.", ""]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("crops")
    c.add_argument("--split", choices=("train", "val", "test"), required=True)
    f = sub.add_parser("fit")
    f.add_argument("--view", choices=rl.VIEWS, required=True)
    f.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("score")
    s.add_argument("--view", choices=rl.VIEWS, required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--split", choices=("val", "test"), required=True)
    sub.add_parser("report")
    sub.add_parser("sample")
    args = p.parse_args()
    {"crops": cmd_crops, "fit": cmd_fit, "score": cmd_score, "report": cmd_report,
     "sample": cmd_sample}[args.cmd](args)  # fmt: skip


if __name__ == "__main__":
    main()
