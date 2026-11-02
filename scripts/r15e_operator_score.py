"""R15e: score an operator's answers on the crops of ``r15e_operator_crops.py``.

The crop script labels a spot "person" when the detection was matched to a person by
R15b's scoring rule (greedy, one detection per person, centre inside the box). That is
right for detection accuracy but not for "is the ringed spot on a person?": a second
detection on the same person, or one on a head whose centre falls just outside the box,
counts as a false alarm there. This script re-runs the same seeded selection and also
labels each spot by whether any labelled person's box (VisDrone pedestrian or people,
counted or not) comes within the ring (``RING_PX`` tile pixels) of it. It scores the
answers under both labels, with Wilson intervals.

python scripts/r15e_operator_score.py --answers <folder of answer JSON files> [--out docs/results/r15e_operator.md]
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from genetic_coil_arena.experiments import r15_scout_brain as r15
from genetic_coil_arena.experiments import r15b_aerial_scout as r15b

OUT = r15.DATA_DIR / "r15e_operator"
RING_PX = 16 / 3  # the ring's radius on the 3x display, in tile pixels


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if not n:
        return (float("nan"),) * 3
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


def spot_labels(seed: int = 1519, tiles: int = 4000, threshold: float = 0.5,
                n: int = 200) -> list[dict]:  # fmt: skip
    """The crop script's selection, re-run, with the distance from each spot to the
    nearest labelled person box (0 inside)."""
    import torch

    rng = np.random.default_rng(seed)
    val = r15b.load_tiles(r15.DATA_DIR / "visdrone_tiles224", "val")
    native = np.flatnonzero(np.asarray(val["scale"]) == 1.0)
    pick = np.sort(rng.choice(native, size=min(tiles, len(native)), replace=False))
    images = np.asarray(val["images"])[pick]
    boxes = [r15.image_boxes(val, int(i)) for i in pick]
    model = r15.build_model(0.5)
    model.load_state_dict(torch.load(OUT.parent / "r15_models" / "r15b_coco_a0.5_seed0.pt",
                                     map_location="cpu"))  # fmt: skip
    dev = torch.device("cpu")
    to_input = r15b.input_fn(dev)
    preds = []
    for a in range(0, len(pick), 128):
        split = {
            "images": images[a : a + 128],
            "offsets": np.arange(len(images[a : a + 128]) + 1),
        }
        preds += r15b.predict(model, split, 224, dev, to_input, top=100)[0]
    table = r15b.hit_table(preds, boxes, "centre")
    keep = table["score"] >= threshold
    true_idx = np.flatnonzero(keep & (table["det_gt"] >= 0))
    false_idx = np.flatnonzero(keep & (table["det_gt"] < 0))
    k = n // 2
    chosen = np.concatenate([rng.choice(true_idx, k, replace=False),
                             rng.choice(false_idx, k, replace=False)])  # fmt: skip
    rng.shuffle(chosen)
    flat = np.concatenate(preds)
    out = []
    for j, d in enumerate(chosen):
        x, y, w, h, _ = flat[d]
        cx, cy = x + w / 2, y + h / 2
        g = boxes[int(table["det_tile"][d])]
        dist = float("inf")
        for bx, by, bw, bh in g[:, :4]:
            dx = max(bx - cx, 0.0, cx - (bx + bw))
            dy = max(by - cy, 0.0, cy - (by + bh))
            dist = min(dist, math.hypot(dx, dy))
        out.append({"id": f"c{j:03d}", "matched": bool(table["det_gt"][d] >= 0),
                    "dist_px": dist, "near_person": dist <= RING_PX})  # fmt: skip
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--answers", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("docs/results/r15e_operator.md"))
    args = p.parse_args()
    answers = {}
    for f in sorted(args.answers.glob("c*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        answers[f.stem] = d.get("data", d)
    truth = {
        t["id"]: t for t in json.loads((OUT / "truth.json").read_text(encoding="utf-8"))
    }
    spots = {s["id"]: s for s in spot_labels()}
    assert all(
        spots[i]["matched"] == truth[i]["person"] for i in truth
    ), "selection differs"
    adj = json.loads(
        Path("docs/results/r15e_adjudication.json").read_text(encoding="utf-8")
    )
    rows = []
    # clear cases only: R15b-matched people and spots judged a person by eye count as
    # people; only spots judged clearly not a person count as empty; ambiguous ones drop
    clear_p = {i for i in answers if spots[i]["matched"]} | set(adj["person"])
    clear_n = set(adj["not_person"])
    for name, pos, neg in (("clear cases (visual check of the false alarms)",
                            sorted(clear_p), sorted(clear_n)),):  # fmt: skip
        hit = sum(answers[i]["label"] == "person" for i in pos)
        fa = sum(answers[i]["label"] == "person" for i in neg)
        rows.append({"truth": name, "persons": len(pos), "hit": wilson(hit, len(pos)),
                     "hit_k": hit, "empties": len(neg), "false_accept": wilson(fa, len(neg)),
                     "fa_k": fa})  # fmt: skip
    for name, is_person in (("matched by R15b's scoring rule", lambda i: spots[i]["matched"]),
                            ("a labelled person within the ring", lambda i: spots[i]["near_person"])):  # fmt: skip
        pos = [i for i in answers if is_person(i)]
        neg = [i for i in answers if not is_person(i)]
        hit = sum(answers[i]["label"] == "person" for i in pos)
        fa = sum(answers[i]["label"] == "person" for i in neg)
        rows.append({"truth": name, "persons": len(pos), "hit": wilson(hit, len(pos)),
                     "hit_k": hit, "empties": len(neg), "false_accept": wilson(fa, len(neg)),
                     "fa_k": fa})  # fmt: skip
    ms = sorted(a["ms"] for a in answers.values())
    rep = {"rows": rows, "median_ms": ms[len(ms) // 2],
           "unsure": sum(a["label"] == "can't tell" for a in answers.values()),
           "spots": list(spots.values())}  # fmt: skip
    lines = ["# R15e: how often an operator confirms the scout brain's false alarms", "",
             f"200 ringed spots the scout brain flagged at ≥ 0.5 on VisDrone val (native "
             f"scale), answered in a median {rep['median_ms'] / 1000:.2f} s each; "
             f"{rep['unsure']} \"can't tell\".", "",
             "| Truth | Spots on a person | Operator says person | Spots without | Operator "
             "says person (false accept) |", "|---|---|---|---|---|"]  # fmt: skip
    for r in rows:
        h, f = r["hit"], r["false_accept"]
        lines.append(f"| {r['truth']} | {r['persons']} | {r['hit_k']} = {100 * h[0]:.0f}% "
                     f"[{100 * h[1]:.0f}, {100 * h[2]:.0f}] | {r['empties']} | {r['fa_k']} = "
                     f"{100 * f[0]:.0f}% [{100 * f[1]:.0f}, {100 * f[2]:.0f}] |")  # fmt: skip
    lines += ["", "Brackets: 95% Wilson intervals. One operator, crops from the scout's own "
              "view (a confirmer would look from closer), answered quickly as on a live feed. "
              "VisDrone leaves many riders and seated or shadowed people unlabelled, so the two "
              "label-based rows count real people as false alarms; the clear-cases row keeps "
              "only spots that are plainly a person or plainly not one "
              "(`docs/results/r15e_adjudication.json`; the visual check was not blind to the "
              "answers).", ""]  # fmt: skip
    md = "\n".join(lines)
    args.out.write_text(md, encoding="utf-8")
    args.out.with_suffix(".json").write_text(
        json.dumps(rep, indent=1), encoding="utf-8"
    )
    print(md)


if __name__ == "__main__":
    main()
