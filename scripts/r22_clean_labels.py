"""R22: the scout brain trained on confirmer-cleaned labels (experiments/r22_clean_labels.py).

python scripts/r22_clean_labels.py fit --fold 0          # then --fold 1 (GPU)
python scripts/r22_clean_labels.py score --fold 0        # then --fold 1 (GPU)
python scripts/r22_clean_labels.py pseudo                # pseudo boxes + blind check sheets
python scripts/r22_clean_labels.py tiles                 # visdrone_tiles224_clean (CPU)
python scripts/r15b_aerial_scout.py --tiles-suffix _clean train --init coco --alphas 0.5 --seeds 0 1
python scripts/r22_clean_labels.py detect --model control_s0 --split val   # every model, val and test
python scripts/r22_clean_labels.py sample                # blind sheets of test-dev false alarms
python scripts/r22_clean_labels.py report

Everything is written under ``D:/Projects/genetic-coil-arena-data/r22_clean_labels``;
``report`` writes ``docs/results/r22_clean_labels.{md,json}``. Test-dev is detected once
per model.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HOME", r"D:\hf_cache")

from genetic_coil_arena.experiments import r19_confirmer as r19  # noqa: E402
from genetic_coil_arena.experiments import r22_clean_labels as rc  # noqa: E402

EPOCHS, SIZE = 10, 336  # R19's chosen recipe
VISUAL = Path("docs/results/r22_visual_check.json")  # labels per test-dev sheet spot
PSEUDO_VISUAL = Path("docs/results/r22_pseudo_check.json")  # labels per pseudo sheet spot
PSEUDO_N = 100


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def train_candidates() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    meta = dict(np.load(r19.OUT_DIR / "train_meta.npz"))
    crops = np.memmap(r19.OUT_DIR / "train_crops.u8", np.uint8, "r",
                      shape=(len(meta["frame"]), 2 * r19.HALF, 2 * r19.HALF, 3))  # fmt: skip
    return crops, meta


def val_candidates() -> tuple[np.ndarray, np.ndarray]:
    meta = dict(np.load(r19.OUT_DIR / "val_meta.npz"))
    crops = np.memmap(r19.OUT_DIR / "val_crops.u8", np.uint8, "r",
                      shape=(len(meta["frame"]), 2 * r19.HALF, 2 * r19.HALF, 3))  # fmt: skip
    keep = ~meta["reserved"].astype(bool)  # R19's val selection set
    return np.asarray(crops[keep]), meta["person"][keep].astype(bool)


def cmd_fit(args) -> None:
    import torch

    start = time.perf_counter()
    crops, meta = train_candidates()
    idx = np.flatnonzero(rc.candidate_folds(meta) != args.fold)  # train on the other fold
    log(f"fold {args.fold}: fitting on {len(idx)} candidates of the other fold")
    m, hist = r19.train_dino("full", np.asarray(crops[idx]), meta["person"][idx].astype(bool),
                             val_candidates(), 0, epochs=EPOCHS, size=SIZE, log=log)  # fmt: skip
    out = rc.OUT_DIR / "models" / f"fold{args.fold}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"net": m.net.state_dict(), "head": m.head.state_dict(), "size": m.size},
               out / "model.pt")  # fmt: skip
    r19.save_json(out / "fit.json", {"fold": args.fold, "history": hist,
                                     "seconds": time.perf_counter() - start})  # fmt: skip


def cmd_score(args) -> None:
    import torch

    crops, meta = train_candidates()
    idx = np.flatnonzero(rc.candidate_folds(meta) == args.fold)
    state = torch.load(rc.OUT_DIR / "models" / f"fold{args.fold}" / "model.pt",
                       map_location="cpu")  # fmt: skip
    m = r19.DinoConfirmer(r19._gpu(), state["size"])
    m.net.load_state_dict(state["net"])
    m.head.load_state_dict(state["head"])
    s = m.score(np.asarray(crops[idx]))
    np.save(rc.OUT_DIR / f"oof_fold{args.fold}.npy", np.stack([idx, s]))
    p = meta["person"][idx].astype(bool)
    log(f"fold {args.fold}: {len(idx)} scores, AUC {r19.auc(s[p], s[~p]):.3f}")


def oof_scores(meta: dict[str, np.ndarray]) -> np.ndarray:
    s = np.full(len(meta["frame"]), np.nan)
    for f in (0, 1):
        idx, v = np.load(rc.OUT_DIR / f"oof_fold{f}.npy")
        s[idx.astype(int)] = v
    assert not np.isnan(s).any(), "score both folds first"
    return s


def cmd_pseudo(_args) -> None:
    crops, meta = train_candidates()
    s = oof_scores(meta)
    fold = rc.candidate_folds(meta)
    mask, thr = rc.pseudo_mask(meta, s, fold)
    boxes = rc.pseudo_boxes(meta, mask)
    np.savez(rc.OUT_DIR / "pseudo.npz", mask=mask, score=s)
    rejected = ~meta["person"].astype(bool) & (meta["dist"] > 0) & ~mask
    rng = np.random.default_rng(rc.FOLD_SEED)
    pool = np.concatenate([rng.choice(np.flatnonzero(mask), PSEUDO_N, replace=False),
                           rng.choice(np.flatnonzero(rejected), PSEUDO_N, replace=False)])  # fmt: skip
    pool = rng.permutation(pool)
    rc.contact_sheets(np.asarray(crops[pool]), rc.OUT_DIR / "pseudo_visual")
    r19.save_json(rc.OUT_DIR / "pseudo_visual" / "pool.json", {"index": [int(i) for i in pool]})
    rec = {"thresholds": thr, "pseudo": int(mask.sum()), "frames": len(boxes),
           "unlabelled_off_box": int((~meta["person"].astype(bool) & (meta["dist"] > 0)).sum()),
           "candidates": int(len(mask))}  # fmt: skip
    r19.save_json(rc.OUT_DIR / "pseudo.json", rec)
    log(f"pseudo: {rec}")


def cmd_tiles(_args) -> None:
    _, meta = train_candidates()
    mask = np.load(rc.OUT_DIR / "pseudo.npz")["mask"]
    rc.build_clean_tiles(rc.pseudo_boxes(meta, mask), log=log)


def det_path(model: str, split: str) -> Path:
    return rc.OUT_DIR / "dets" / f"{model}_{split}.npz"


def cmd_detect(args) -> None:
    out = det_path(args.model, args.split)
    if args.split == "test" and out.exists():
        raise SystemExit(f"{out} exists: test-dev is detected once per model")
    d = rc.detect(rc.MODELS[args.model], args.split, log)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **d)
    log(f"{args.model} {args.split}: {len(d['dets'])} detections")


def load(model: str, split: str) -> dict[str, np.ndarray]:
    return dict(np.load(det_path(model, split)))


def thresholds() -> dict[str, float]:
    base = rc.recall(load("control_s0", "val"), rc.BASE_THRESHOLD)
    out = {m: rc.matched_threshold(load(m, "val"), base) for m in rc.MODELS}
    out["control_s0"] = rc.BASE_THRESHOLD
    return out


def test_outcomes() -> dict[str, dict[str, np.ndarray]]:
    thr = thresholds()
    return {m: rc.outcomes(load(m, "test"), thr[m]) for m in rc.MODELS}


def cmd_sample(_args) -> None:
    outs = test_outcomes()
    spots = rc.sample_spots({m: np.flatnonzero(o["fa"]) for m, o in outs.items()})
    want: dict[int, list[int]] = {}
    for k, (m, i) in enumerate(spots):
        want.setdefault(int(outs[m]["frame"][i]), []).append(k)
    crops = np.zeros((len(spots), 2 * r19.HALF, 2 * r19.HALF, 3), np.uint8)
    for f, _, img, _ in r19.frames("test"):
        for k in want.get(f, []):
            m, i = spots[k]
            crops[k] = r19.cut(img, float(outs[m]["cx"][i]), float(outs[m]["cy"][i]))
    n = rc.contact_sheets(crops, rc.OUT_DIR / "visual")
    r19.save_json(rc.OUT_DIR / "visual" / "pool.json", {"spots": spots})
    log(f"{len(spots)} spots on {n} sheets")


def wil(k: int, n: int) -> list[float]:
    from genetic_coil_arena.translator.metrics import wilson

    return list(wilson(int(k), int(n)))


def cmd_report(_args) -> None:
    thr = thresholds()
    rep: dict = {"thresholds": thr, "models": {}}
    if (rc.OUT_DIR / "pseudo.json").exists():
        rep["pseudo"] = json.loads((rc.OUT_DIR / "pseudo.json").read_text("utf-8"))
    if PSEUDO_VISUAL.exists():
        chk = json.loads(PSEUDO_VISUAL.read_text("utf-8"))
        mask = np.load(rc.OUT_DIR / "pseudo.npz")["mask"]
        pool = json.loads((rc.OUT_DIR / "pseudo_visual" / "pool.json").read_text("utf-8"))["index"]
        lab = [chk["labels"][str(n)] for n in range(len(pool))]
        rep["pseudo_check"] = {
            grp: {c: sum(1 for i, x in zip(pool, lab) if bool(mask[i]) == want and x == c)
                  for c in rc.CLASSES}
            for grp, want in (("pseudo_positives", True), ("rejected", False))}  # fmt: skip
    have_test = all(det_path(m, "test").exists() for m in rc.MODELS)
    outs = test_outcomes() if have_test else {}
    for m in rc.MODELS:
        v = load(m, "val")
        rec = {"threshold": thr[m], "val_recall": rc.recall(v, thr[m])}
        if m in outs:
            o = outs[m]
            rec |= {"test_recall": wil(o["found"].sum(), o["people"].sum()),
                    "test_fa_per_frame": float(o["fa"].sum() / len(o["people"]))}  # fmt: skip
        rep["models"][m] = rec
    if outs and VISUAL.exists():
        rep |= visual(outs)
    r19.save_json(Path("docs/results/r22_clean_labels.json"), rep)
    Path("docs/results/r22_clean_labels.md").write_text(render(rep), "utf-8")
    log(render(rep))


def visual(outs: dict[str, dict[str, np.ndarray]]) -> dict:
    """Clear non-person false alarms per frame, by model, with frame-bootstrap ratios."""
    spots = json.loads((rc.OUT_DIR / "visual" / "pool.json").read_text("utf-8"))["spots"]
    labels = json.loads(VISUAL.read_text("utf-8"))["labels"]
    n_frames = len(next(iter(outs.values()))["people"])
    boot = r19.frame_bootstrap(np.arange(n_frames)).astype(float)
    rates, counts = {}, {}
    for m, o in outs.items():
        n_fa = np.bincount(o["frame"][o["fa"]], minlength=n_frames).astype(float)
        sampled, clear = np.zeros(n_frames), np.zeros(n_frames)
        counts[m] = dict.fromkeys(rc.CLASSES, 0)
        for k, (mm, i) in enumerate(spots):
            if mm == m:
                f, c = int(o["frame"][i]), labels[str(k)]
                sampled[f] += 1
                clear[f] += c == "not_person"
                counts[m][c] += 1
        point = rc.clear_rate(n_fa, sampled, clear, np.ones((1, n_frames)))[0]
        rates[m] = (point, rc.clear_rate(n_fa, sampled, clear, boot))
    def ratio(num, den):
        p = sum(rates[m][0] for m in num) / sum(rates[m][0] for m in den)
        b = sum(rates[m][1] for m in num) / np.maximum(sum(rates[m][1] for m in den), 1e-9)
        return [float(p), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]
    out = {"sample_counts": counts,
           "clear_per_frame": {m: float(r[0]) for m, r in rates.items()},
           "ratio": {"pooled": ratio(("clean_s0", "clean_s1"), ("control_s0", "control_s1")),
                     "seed0": ratio(("clean_s0",), ("control_s0",)),
                     "seed1": ratio(("clean_s1",), ("control_s1",))}}  # fmt: skip
    r = out["ratio"]
    found = {m: outs[m]["found"].sum() / outs[m]["people"].sum() for m in rc.MODELS}
    recall_ok = (found["clean_s0"] + found["clean_s1"]) / 2 >= (
        found["control_s0"] + found["control_s1"]) / 2 - rc.RECALL_DROP
    if (r["pooled"][0] <= rc.RATIO_PASS and r["pooled"][2] < 1 and recall_ok
            and max(r["seed0"][0], r["seed1"][0]) <= rc.RATIO_SEED):  # fmt: skip
        out["verdict"] = "PASS"
    elif r["pooled"][0] >= rc.RATIO_KILL:
        out["verdict"] = "KILL"
    else:
        out["verdict"] = "NOT PASSED"
    return out


def render(rep: dict) -> str:
    lines = ["# R22: the scout brain trained on confirmer-cleaned labels", ""]
    if "verdict" in rep:
        r = rep["ratio"]["pooled"]
        lines += [f"**{rep['verdict']}.** Clear non-person false alarms per test-dev frame, "
                  f"cleaned scouts over control scouts (both seeds pooled): {r[0]:.2f}x "
                  f"[{r[1]:.2f}, {r[2]:.2f}]; seed 0 {rep['ratio']['seed0'][0]:.2f}x, seed 1 "
                  f"{rep['ratio']['seed1'][0]:.2f}x (bar: <= {rc.RATIO_PASS}x with the upper "
                  f"bound below 1, each seed <= {rc.RATIO_SEED}x, people found within "
                  f"{100 * rc.RECALL_DROP:.0f} points; kill at >= {rc.RATIO_KILL}x).", ""]  # fmt: skip
    if "pseudo" in rep:
        p = rep["pseudo"]
        lines += [f"Pseudo-labels: {p['pseudo']} of {p['unlabelled_off_box']} unlabelled "
                  f"off-box train candidates became person boxes, in {p['frames']} frames.", ""]  # fmt: skip
    if "pseudo_check" in rep:
        for grp, c in rep["pseudo_check"].items():
            lines.append(f"- blind check, {grp.replace('_', ' ')}: {c['person']} people, "
                         f"{c['ambiguous']} unclear, {c['not_person']} clearly not a person")  # fmt: skip
        lines.append("")
    lines += ["| Scout | Threshold | Val people found | Test-dev people found | "
              "Test-dev false alarms per frame | Clear non-persons per frame |",
              "|---|---|---|---|---|---|"]  # fmt: skip
    for m, rec in rep["models"].items():
        t = rec.get("test_recall")
        lines.append(f"| {m} | {rec['threshold']:.3f} | {100 * rec['val_recall']:.1f}% | "
                     + (f"{100 * t[0]:.1f}% [{100 * t[1]:.0f}, {100 * t[2]:.0f}] | "
                        f"{rec['test_fa_per_frame']:.2f} | " if t else "- | - | ")
                     + (f"{rep['clear_per_frame'][m]:.2f} |" if "clear_per_frame" in rep
                        else "- |"))  # fmt: skip
    if "sample_counts" in rep:
        lines += ["", "Blind check of each scout's test-dev false alarms (VisDrone labels):", "",
                  "| Scout | People | Unclear | Clearly not a person |", "|---|---|---|---|"]  # fmt: skip
        for m, c in rep["sample_counts"].items():
            lines.append(f"| {m} | {c['person']} | {c['ambiguous']} | {c['not_person']} |")
    lines += ["", "Reproduce: `python scripts/r22_clean_labels.py report`.", ""]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("fit", "score"):
        s = sub.add_parser(name)
        s.add_argument("--fold", type=int, choices=(0, 1), required=True)
    for name in ("pseudo", "tiles", "sample", "report"):
        sub.add_parser(name)
    d = sub.add_parser("detect")
    d.add_argument("--model", choices=list(rc.MODELS), required=True)
    d.add_argument("--split", choices=("val", "test"), required=True)
    args = p.parse_args()
    {"fit": cmd_fit, "score": cmd_score, "pseudo": cmd_pseudo, "tiles": cmd_tiles,
     "detect": cmd_detect, "sample": cmd_sample, "report": cmd_report}[args.cmd](args)  # fmt: skip


if __name__ == "__main__":
    main()
