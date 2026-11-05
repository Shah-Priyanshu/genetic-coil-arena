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

R25c (goal G21c): R20b found that most labelled false alarms past a confirmer are people
VisDrone did not label, so R25/R25b's labelled counts may mostly measure label noise. R25c
counts only clear non-persons. From each of R25b's four chains (224 seeds 0/1 with R19's
confirmer, 448 seeds 0/1 with R25b's), ``N_SPOTS`` of its passed test-dev false alarms are
drawn at random (seed ``SPOT_SEED``); with ``REPEATS`` of them shown twice (consistency
only), they go on sheets in one random order with the chain hidden, and Claude subagents
label each spot blind: person / not_person / ambiguous at the marked centre (R20b's
protocol). Per chain, clear non-persons passed per frame = its passed false alarms per
frame x the share of its sample labelled not_person (per frame: N/n x the sampled
not_person spots in that frame, so the frame bootstrap carries the sampling). R25's bars
apply to that count: PASS if, pooled over seeds, the 448 chain passes <= 0.67x the 224
chain's (upper bound < 1), each seed <= 0.8x, people found within 1 point (R25b's); KILL at
>= 0.9x. Measured with no bar: each chain's people / unclear shares, label consistency on
the repeats.
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


# R25c: clear non-persons, labelled blind
N_SPOTS, REPEATS, SPOT_SEED = 400, 36, 2025
C_CHAINS = (("s224_seed0", "r19"), ("s448_seed0", OWN), ("s224_seed1", "r19"),
            ("s448_seed1", OWN))  # fmt: skip
CLASSES = ("person", "not_person", "ambiguous")
VISUAL_C = OUT_DIR / "visual_c"
LABELS_C = Path("docs/results/r25c_visual_check.json")


def passed_false(c: dict) -> np.ndarray:
    return np.flatnonzero(c["keep"] & ~c["person"])


def sample_c(chains: dict, seed: int = SPOT_SEED) -> list[tuple[str, int]]:
    """``N_SPOTS`` passed false alarms per chain, plus ``REPEATS`` of them again, in one
    random order: the sheet order."""
    rng = np.random.default_rng(seed)
    spots = [(tag, int(i)) for tag, _ in C_CHAINS
             for i in np.sort(rng.choice(passed_false(chains[tag]),
                                         min(N_SPOTS, len(passed_false(chains[tag]))),
                                         replace=False))]  # fmt: skip
    spots += [spots[k] for k in rng.choice(len(spots), REPEATS, replace=False)]
    return [spots[k] for k in rng.permutation(len(spots))]


def draw_sheets(pool: list[tuple[str, int]], out: Path, per: int = 12) -> int:
    """R20b's sheets: the centre at 4x and the whole 144-px view with the centre marked."""
    from PIL import Image, ImageDraw

    crops = {tag: rc.load("test", scout_dir(tag))[0] for tag, _ in C_CHAINS}
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("sheet_*.png"):
        old.unlink()
    h = rc.HALF
    for sheet in range(0, len(pool), per):
        img = Image.new("RGB", (3 * 600, 4 * 310), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        for k, (tag, i) in enumerate(pool[sheet : sheet + per]):
            x0, y0 = (k % 3) * 600, (k // 3) * 310
            view = Image.fromarray(np.asarray(crops[tag][i]))
            zoom = view.crop((h // 2, h // 2, 3 * h // 2, 3 * h // 2)).resize((288, 288))
            img.paste(zoom, (x0, y0 + 20))
            img.paste(view.resize((288, 288)), (x0 + 296, y0 + 20))
            cx, cy = x0 + 296 + 144, y0 + 20 + 144
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                draw.line((cx + 30 * dx, cy + 30 * dy, cx + 48 * dx, cy + 48 * dy),
                          fill=(255, 0, 255), width=2)  # fmt: skip
            draw.text((x0 + 4, y0 + 4), f"#{sheet + k:04d}  centre x4 | whole view",
                      fill=(0, 0, 0))  # fmt: skip
        img.save(out / f"sheet_{sheet // per:03d}.png")
    return (len(pool) + per - 1) // per


def sample_sheets(log: Any = print) -> None:
    if LABELS_C.exists():
        raise SystemExit(f"{LABELS_C} exists: the sample is fixed")
    chains = {tag: chain(tag, v) for tag, v in C_CHAINS}
    pool = sample_c(chains)
    n = draw_sheets(pool, VISUAL_C)
    (VISUAL_C / "pool.json").write_text(json.dumps({"seed": SPOT_SEED, "spots": pool}),
                                        "utf-8")  # fmt: skip
    log(f"{len(pool)} spots on {n} sheets in {VISUAL_C}")


def merge_labels() -> dict[str, Any]:
    """The subagents' ``labels_*.json`` ({spot number: class}) and the pool, into
    ``LABELS_C`` (kept in git)."""
    if LABELS_C.exists():
        return json.loads(LABELS_C.read_text("utf-8"))
    pool = json.loads((VISUAL_C / "pool.json").read_text("utf-8"))
    labels: dict[str, str] = {}
    for f in sorted(VISUAL_C.glob("labels_*.json")):
        for k, v in json.loads(f.read_text("utf-8")).items():
            labels[str(int(str(k).lstrip("#")))] = v
    missing = [n for n in range(len(pool["spots"])) if str(n) not in labels]
    bad = sorted({v for v in labels.values() if v not in CLASSES})
    if missing or bad:
        raise SystemExit(f"labels missing for {len(missing)} spots, unknown classes {bad}")
    rec = {**pool, "labels": labels}
    LABELS_C.write_text(json.dumps(rec), "utf-8")
    return rec


def clear_compare(rec: dict[str, Any], chains: dict, n_frames: int,
                  people_diff: dict[str, float]) -> dict[str, Any]:  # fmt: skip
    """Clear non-persons passed per frame, per chain and 448 over 224, from the labels."""
    first: dict[tuple[str, int], str] = {}
    pairs = []
    for n, (tag, i) in enumerate(rec["spots"]):
        lab, key = rec["labels"][str(n)], (tag, int(i))
        if key in first:
            pairs.append(first[key] == lab)
        else:
            first[key] = lab
    rng = np.random.default_rng(BOOT_SEED)
    counts = rng.multinomial(n_frames, np.full(n_frames, 1 / n_frames), size=N_BOOT)
    out: dict[str, Any] = {"chains": {}, "seeds": {}, "n_frames": n_frames,
                           "repeats_agree": [int(sum(pairs)), len(pairs)]}  # fmt: skip
    est: dict[str, tuple[float, np.ndarray]] = {}
    for tag, variant in C_CHAINS:
        c = chains[tag]
        mine = {i: lab for (t, i), lab in first.items() if t == tag}
        n_all, n = len(passed_false(c)), len(mine)
        clear = np.bincount([c["frame"][i] for i, lab in mine.items() if lab == "not_person"],
                            minlength=n_frames) * n_all / n  # fmt: skip
        est[tag] = (float(clear.sum()), counts @ clear)
        shares = {k: sum(lab == k for lab in mine.values()) for k in CLASSES}
        out["chains"][tag] = {"confirmer": variant, "passed_false": n_all, "sampled": n,
                              **shares, "false_per_frame": n_all / n_frames,
                              "clear_per_frame": clear.sum() / n_frames}  # fmt: skip
    pooled = [0.0, np.zeros(N_BOOT), 0.0, np.zeros(N_BOOT)]
    for a, b in PAIRS:
        ratio_b = est[b][1] / est[a][1]
        out["seeds"][a.split("_")[1]] = {
            "ratio": [est[b][0] / est[a][0], *np.percentile(ratio_b, [2.5, 97.5]).tolist()],
            "people_diff": people_diff[a.split("_")[1]]}  # fmt: skip
        pooled = [pooled[0] + est[b][0], pooled[1] + est[b][1], pooled[2] + est[a][0],
                  pooled[3] + est[a][1]]  # fmt: skip
    out["pooled_ratio"] = [pooled[0] / pooled[2],
                           *np.percentile(pooled[1] / pooled[3], [2.5, 97.5]).tolist()]  # fmt: skip
    out["verdict"] = verdict(out)
    return out


def render_c(rep: dict[str, Any]) -> str:
    pr, ag = rep["pooled_ratio"], rep["repeats_agree"]
    lines = ["# R25c: the 448-px camera, counting only clear non-persons", "",
             f"**{rep['verdict']}.** Pooled over seeds, the 448 chain passes {pr[0]:.2f}x "
             f"[{pr[1]:.2f}, {pr[2]:.2f}] the 224 chain's clear non-persons per test-dev frame "
             f"(R25's bars: pooled <= {POOLED_MAX}x with the upper bound below 1, each seed "
             f"<= {SEED_MAX}x, people found within {100 * PEOPLE_DROP:.0f} point; KILL at "
             f">= {KILL}x).", "",
             f"R25b's chains. Per chain {N_SPOTS} of its passed test-dev false alarms (VisDrone "
             f"labels) drawn at random (seed {SPOT_SEED}) and labelled blind by Claude "
             f"subagents, chain hidden; clear non-persons per frame = passed false alarms per "
             f"frame x the share labelled not a person.", "",
             "| Chain | Confirmer | Passed false alarms per frame | Sampled | Person | Unclear "
             "| Not a person | Clear non-persons per frame |", "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for tag, c in rep["chains"].items():
        conf = "R19's" if c["confirmer"] == "r19" else "its own (R25b)"
        lines.append(f"| {tag.replace('_', ', ').replace('s224', '224 px').replace('s448', '448 px')}"
                     f" | {conf} | {c['false_per_frame']:.3f} | {c['sampled']} | {c['person']} | "
                     f"{c['ambiguous']} | {c['not_person']} | {c['clear_per_frame']:.3f} |")  # fmt: skip
    lines += [""]
    for seed, row in rep["seeds"].items():
        r = row["ratio"]
        lines.append(f"- Seed {seed[-1]}: 448 over 224 clear non-persons passed {r[0]:.2f}x "
                     f"[{r[1]:.2f}, {r[2]:.2f}]; people found {100 * row['people_diff']:+.1f} "
                     "points (R25b).")  # fmt: skip
    lines += ["", f"Repeated spots labelled the same both times: {ag[0]} of {ag[1]}. "
              "Intervals: 95% frame bootstrap (2,000 resamples, the same frames for every "
              "chain), which also resamples the labelled spots. Reproduce: `python "
              "scripts/r25_camera.py sample-c`, label the sheets, `report-c`.", ""]  # fmt: skip
    return "\n".join(lines)


def report_c(path: Path = Path("docs/results/r25c_camera_clear.md")) -> dict[str, Any]:
    rec = merge_labels()
    chains = {tag: chain(tag, v) for tag, v in C_CHAINS}
    b = json.loads(Path("docs/results/r25b_camera_own.json").read_text("utf-8"))
    diff = {seed: row["people_diff"] for seed, row in b["seeds"].items()}
    rep = clear_compare(rec, chains, len(person_counts("test")), diff)
    path.write_text(render_c(rep), "utf-8")
    path.with_suffix(".json").write_text(json.dumps(rep, indent=1), "utf-8")
    return rep
