"""R25 (goal G21): does the 448-px camera cut the false alarms that pass the confirmer?

R17 step 6h: to keep wrong finds under 5% with margin, the chain must pass about a third
fewer empty candidates than it does now. R15c found that the 448-px brain, at the same
pixels per person, raises about a third fewer false detections per area than the 224 brain
(0.065 against 0.10 per 224-px area at 0.5) and finds people of a given pixel size as
often. This step asks whether that survives R19's confirmer, which was trained on the 224
scout's candidates.

Scouts: R15b's 224 brain and R15c's 448 brain, seeds 0 and 1 each, float, at threshold 0.5,
over every VisDrone2019-DET val and test-dev frame at native scale (R19's ``build``: tiles
of the brain's size, ignore regions greyed, tiles merged). The confirmer is R19's chosen
model (DINOv2-small fine-tuned, ``dino-full-e10s336`` seed 2), unchanged, on the 144-px view
around each candidate. Per scout its threshold keeps 95% of that scout's val people (R19's
rule, with the candidates near R15e/R15f's operator spots kept out). The 224 seed-0
candidates and their confirmer scores are R19's own.

Bars, fixed before any 448 or 224 seed-1 candidate is built:

- Read-out per test-dev frame, VisDrone labels: the share of counted persons that scout
  and confirmer both pass ("people found"), and the labelled false alarms the confirmer
  passes per frame.
- **PASS** if, pooled over seeds, the 448 chain passes at most 0.67x the 224 chain's false
  alarms per frame (frame-bootstrap 95% upper bound below 1), each seed at most 0.8x, and
  each seed's 448 chain finds at least its 224 chain's share of people minus 1 point.
- **KILL** if the pooled ratio is 0.9x or more: the 448 camera's lower false-alarm rate does
  not survive the confirmer.
- Measured with no bar: candidates per frame, the confirmer's test-dev AUC per scout, the
  share of false alarms it passes.

The scouts are float; R24b tests whether the compiled 448 brain matches float.

R25b (goal G21b): R25 was a KILL because R19's confirmer, trained on the 224 scout's
candidates, separates the 448 scout's candidates worse (test-dev AUC 0.85 against 0.91).
R25b gives the 448 camera its own confirmer: R19's recipe (DINOv2-small fine-tuned, 10
epochs, 336 px, the epoch with the lowest val pass kept) on the 448 seed-0 scout's train
candidates (R19's rule: threshold 0.4, at most 12 per frame), seeds 0-2, the seed with
the lowest val pass chosen on that scout's val selection set (R19's rule). The 448 chains
(both seeds) use it; the 224 chains keep R19's confirmer; R25's bars unchanged.
Pre-registered after R25's test-dev numbers were seen; the new confirmer is a new system
and scores test-dev once.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from . import r15_scout_brain as r15
from . import r19_confirmer as rc

OUT_DIR = r15.DATA_DIR / "r25_camera"
CONFIRMER = ("dino-full", "-e10s336", 2)  # R19's chosen model: name, tag, seed
SCOUTS = {  # tag -> (brain weights, tile size)
    "s224_seed0": ("r15b_coco_a0.5_seed0", 224),
    "s224_seed1": ("r15b_coco_a0.5_seed1", 224),
    "s448_seed0": ("r15b_coco_a0.5_seed0_s448", 448),
    "s448_seed1": ("r15b_coco_a0.5_seed1_s448", 448),
}
PAIRS = (("s224_seed0", "s448_seed0"), ("s224_seed1", "s448_seed1"))
POOLED_MAX, SEED_MAX, PEOPLE_DROP, KILL = 0.67, 0.80, 0.01, 0.90
OWN, OWN_SCOUT, OWN_SEEDS = "own", "s448_seed0", (0, 1, 2)  # R25b's confirmer
TITLES = {
    "r19": "# R25: does the 448-px camera cut the false alarms that pass the confirmer?",
    OWN: "# R25b: the 448-px camera with a confirmer trained on its own candidates",
}
N_BOOT, BOOT_SEED = 2000, 25


def scout_dir(tag: str) -> Path:
    """R19's folder for its own scout (224 seed 0), else this step's."""
    return rc.OUT_DIR if tag == "s224_seed0" else OUT_DIR / tag


def score_path(tag: str, split: str, variant: str = "r19") -> Path:
    if variant == OWN:
        return OUT_DIR / tag / f"confirmer_own_{split}.npy"
    if tag == "s224_seed0":
        name, ctag, seed = CONFIRMER
        return rc.OUT_DIR / "scores" / f"{name}{ctag}_s{seed}_{split}.npy"
    return OUT_DIR / tag / f"confirmer_{split}.npy"


def build(tag: str, split: str, log: Any = print) -> dict[str, Any]:
    """The scout's candidates at R19's evaluation threshold (val keeps R19's reserved spots);
    train (R25b, the 448 seed-0 scout only) at R19's training threshold and cap."""
    if tag == "s224_seed0":
        raise SystemExit("s224_seed0 is R19's own candidate set")
    weights, size = SCOUTS[tag]
    kw = {"log": log, "weights": r15.DATA_DIR / "r15_models" / f"{weights}.pt", "size": size,
          "out_dir": scout_dir(tag)}  # fmt: skip
    if split == "train":
        if tag != OWN_SCOUT:
            raise SystemExit(f"R25b trains on {OWN_SCOUT}'s candidates only")
        return rc.build(split, rc.TRAIN_THRESHOLD, cap=rc.TRAIN_CAP, **kw)
    reserved = None
    if split == "val":
        reserved = {}
        for s in json.loads((rc.OUT_DIR / "spots.json").read_text("utf-8")):
            reserved.setdefault(s["frame"], []).append((s["fx"], s["fy"]))
    return rc.build(split, rc.EVAL_THRESHOLD, reserved_spots=reserved, **kw)


def own_dir(seed: int) -> Path:
    return OUT_DIR / "models" / f"dino-full-e10s336_s{seed}"


def fit_own(seed: int, log: Any = print) -> None:
    """R25b: R19's chosen recipe on the 448 seed-0 scout's train candidates."""
    import time

    import torch

    start = time.perf_counter()
    crops, meta = rc.load("train", scout_dir(OWN_SCOUT))
    vc, vm = rc.load("val", scout_dir(OWN_SCOUT))
    sel = ~vm["reserved"]
    log(f"train {len(crops)} candidates ({meta['person'].mean():.1%} person); selection "
        f"{int(sel.sum())} ({vm['person'][sel].mean():.1%} person)")  # fmt: skip
    val = (np.asarray(vc[sel]), vm["person"][sel])
    m, hist = rc.train_dino("full", crops, meta["person"], val, seed, epochs=10, size=336,
                            log=log)  # fmt: skip
    own_dir(seed).mkdir(parents=True, exist_ok=True)
    torch.save({"net": m.net.state_dict(), "head": m.head.state_dict(), "size": m.size},
               own_dir(seed) / "model.pt")  # fmt: skip
    rc.save_json(own_dir(seed) / "fit.json", {"seed": seed, "history": hist,
                                              "seconds": time.perf_counter() - start})  # fmt: skip


def own_choice() -> dict[str, Any]:
    """R19's rule: the seed whose kept epoch passes the fewest val false alarms."""
    fits = {s: json.loads((own_dir(s) / "fit.json").read_text("utf-8")) for s in OWN_SEEDS}
    best = {s: min(h["pass"] for h in f["history"]) for s, f in fits.items()}
    seed = min(best, key=best.get)
    return {"seed": seed, "val_pass": best}


def confirmer(variant: str = "r19") -> Any:
    import torch

    if variant == OWN:
        path = own_dir(own_choice()["seed"]) / "model.pt"
    else:
        name, ctag, seed = CONFIRMER
        path = rc.OUT_DIR / "models" / f"{name}{ctag}_s{seed}" / "model.pt"
    state = torch.load(path, map_location="cpu")
    m = rc.DinoConfirmer(rc._gpu(), state.get("size", 224))
    m.net.load_state_dict(state["net"])
    m.head.load_state_dict(state["head"])
    return m


def score(tag: str, split: str, log: Any = print, variant: str = "r19") -> np.ndarray:
    out = score_path(tag, split, variant)
    if out.exists() and (split == "test" or tag == "s224_seed0"):
        raise SystemExit(f"{out} exists: test-dev is scored once per scout")
    crops, _ = rc.load(split, scout_dir(tag))
    s = confirmer(variant).score(np.asarray(crops))
    np.save(out, s)
    log(f"{tag} {split}: {len(s)} confirmer scores -> {out}")
    return s


def person_counts(split: str) -> np.ndarray:
    """Counted persons per frame (the same for every scout), cached."""
    path = OUT_DIR / f"{split}_persons.npy"
    if not path.exists():
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        np.save(path, np.array([len(p) for _, _, _, p in rc.frames(split)]))
    return np.load(path)


def chain(tag: str, variant: str = "r19") -> dict[str, Any]:
    """Scout + confirmer on test-dev at the threshold set on the scout's own val people."""
    _, vm = rc.load("val", scout_dir(tag))
    vs = np.load(score_path(tag, "val", variant))
    sel = ~vm["reserved"]
    thr = rc.threshold_at(vs[sel][vm["person"][sel]])
    _, tm = rc.load("test", scout_dir(tag))
    ts = np.load(score_path(tag, "test", variant))
    person, keep = tm["person"].astype(bool), ts >= thr
    return {"threshold": float(thr), "frame": tm["frame"], "person": person, "keep": keep,
            "auc": rc.auc(ts[person], ts[~person])}  # fmt: skip


def per_frame(c: dict, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """(people passed, false alarms passed) per frame."""

    def count(mask: np.ndarray) -> np.ndarray:
        return np.bincount(c["frame"][mask], minlength=n_frames).astype(float)

    return count(c["keep"] & c["person"]), count(c["keep"] & ~c["person"])


def compare(pairs: tuple = PAIRS, chains: dict | None = None,
            persons: np.ndarray | None = None, variant: str = "r19") -> dict[str, Any]:  # fmt: skip
    """Each pair (224 tag, 448 tag) on the same test-dev frames; ``chains``, ``persons``
    can be given (tests). ``variant``: the 448 chains' confirmer (R25b: ``OWN``)."""
    persons = person_counts("test") if persons is None else persons
    n_frames = len(persons)
    rng = np.random.default_rng(BOOT_SEED)
    counts = rng.multinomial(n_frames, np.full(n_frames, 1 / n_frames), size=N_BOOT)
    chains = chains or {t: chain(t, variant if "448" in t else "r19")
                        for pair in pairs for t in pair}  # fmt: skip
    rep: dict[str, Any] = {"variant": variant, "n_frames": n_frames,
                           "n_persons": int(persons.sum()), "seeds": {}}  # fmt: skip
    pooled = {"224": [0.0, np.zeros(N_BOOT)], "448": [0.0, np.zeros(N_BOOT)]}
    for a, b in pairs:
        row: dict[str, Any] = {}
        for tag, key in ((a, "224"), (b, "448")):
            c = chains[tag]
            people_f, false_f = per_frame(c, n_frames)
            people, false, false_b = people_f.sum(), false_f.sum(), counts @ false_f
            pooled[key][0] += false
            pooled[key][1] = pooled[key][1] + false_b
            row[key] = {"tag": tag, "threshold": c["threshold"], "auc": c["auc"],
                        "candidates_per_frame": len(c["keep"]) / n_frames,
                        "false_candidates_per_frame": int((~c["person"]).sum()) / n_frames,
                        "people_found": people / persons.sum(),
                        "false_per_frame": false / n_frames,
                        "false_passed_share": false / max(1, int((~c["person"]).sum())),
                        "_false_b": false_b}  # fmt: skip
        ratio_b = row["448"].pop("_false_b") / row["224"].pop("_false_b")
        row["ratio"] = [row["448"]["false_per_frame"] / row["224"]["false_per_frame"],
                        *np.percentile(ratio_b, [2.5, 97.5]).tolist()]  # fmt: skip
        row["people_diff"] = row["448"]["people_found"] - row["224"]["people_found"]
        rep["seeds"][a.split("_")[1]] = row
    ratio_b = pooled["448"][1] / pooled["224"][1]
    rep["pooled_ratio"] = [pooled["448"][0] / pooled["224"][0],
                           *np.percentile(ratio_b, [2.5, 97.5]).tolist()]  # fmt: skip
    rep["verdict"] = verdict(rep)
    return rep


def verdict(rep: dict[str, Any]) -> str:
    r, hi = rep["pooled_ratio"][0], rep["pooled_ratio"][2]
    if r >= KILL:
        return "KILL"
    rows = list(rep["seeds"].values())
    ok = (r <= POOLED_MAX and hi < 1 and len(rows) == len(PAIRS)
          and all(s["ratio"][0] <= SEED_MAX and s["people_diff"] >= -PEOPLE_DROP for s in rows))  # fmt: skip
    return "PASS" if ok else "NOT PASSED"


def render(rep: dict[str, Any]) -> str:
    pr = rep["pooled_ratio"]
    lines = [TITLES[rep.get("variant", "r19")], "",
             f"**{rep['verdict']}.** Pooled over seeds, the 448 chain passes {pr[0]:.2f}x "
             f"[{pr[1]:.2f}, {pr[2]:.2f}] the 224 chain's labelled false alarms per test-dev "
             f"frame (bars: pooled <= {POOLED_MAX}x with the upper bound below 1, each seed "
             f"<= {SEED_MAX}x, people found within {100 * PEOPLE_DROP:.0f} point; KILL at "
             f">= {KILL}x).", "",
             f"VisDrone2019-DET test-dev, {rep['n_frames']} frames, {rep['n_persons']} counted "
             f"persons, native scale. Scout threshold 0.5; {confirmers(rep)} Each confirmer's "
             "threshold is set per scout to keep 95% of that scout's val people.", "",
             "| Seed | Camera | Candidates per frame | False candidates per frame | "
             "Confirmer AUC | People found | False alarms passed per frame | "
             "Share of false alarms passed |", "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for seed, row in rep["seeds"].items():
        for key in ("224", "448"):
            c = row[key]
            lines.append(f"| {seed[-1]} | {key} px | {c['candidates_per_frame']:.2f} | "
                         f"{c.get('false_candidates_per_frame', float('nan')):.2f} | "
                         f"{c['auc']:.3f} | {100 * c['people_found']:.1f}% | "
                         f"{c['false_per_frame']:.3f} | {100 * c['false_passed_share']:.1f}% |")  # fmt: skip
    lines += [""]
    for seed, row in rep["seeds"].items():
        r = row["ratio"]
        lines.append(f"- Seed {seed[-1]}: 448 over 224 false alarms passed {r[0]:.2f}x "
                     f"[{r[1]:.2f}, {r[2]:.2f}], people found {100 * row['people_diff']:+.1f} "
                     "points.")  # fmt: skip
    steps = ("crops --split train`, `fit --seed 0|1|2`, `score --confirmer own`, `report-b"
             if rep.get("variant") == OWN else "crops`, `score`, `report")  # fmt: skip
    lines += ["", "Intervals: 95% frame bootstrap (2,000 resamples, the same frames for both "
              f"cameras). Reproduce: `python scripts/r25_camera.py {steps}`.", ""]  # fmt: skip
    return "\n".join(lines)


def confirmers(rep: dict[str, Any]) -> str:
    if rep.get("variant") != OWN:
        return "R19's confirmer (DINOv2-small fine-tuned) for every chain."
    own = rep["own_confirmer"]
    vp = ", ".join(f"seed {k} {100 * v:.1f}%" for k, v in own["val_pass"].items())
    return (
        "the 224 chains keep R19's confirmer; both 448 chains use R25b's, R19's recipe "
        "fitted on the 448 seed-0 scout's train candidates (seed "
        f"{own['seed']} kept: the lowest val pass at 95% keep; {vp})."
    )


def report(path: Path = Path("docs/results/r25_camera.md"),
           variant: str = "r19") -> dict[str, Any]:  # fmt: skip
    rep = compare(variant=variant)
    if variant == OWN:
        rep["own_confirmer"] = own_choice()
    path.write_text(render(rep), "utf-8")
    path.with_suffix(".json").write_text(json.dumps(rep, indent=1), "utf-8")
    return rep
