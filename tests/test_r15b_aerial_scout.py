"""R15b aerial scout: VisDrone parsing, tiling, matching and stratified/bootstrapped AP."""

import math

import numpy as np
import pytest

from genetic_coil_arena.experiments import r15b_aerial_scout as r15b


def test_parse_annotations_persons_and_ignore_regions():
    text = "10,20,5,12,1,1,0,0\n30,40,6,6,1,2,0,1\n0,0,50,50,0,0,0,0\n5,5,9,9,1,4,0,0\n"
    persons, ignore = r15b.parse_annotations(text)
    assert persons.tolist() == [[10, 20, 5, 12], [30, 40, 6, 6]]
    assert ignore.tolist() == [[0, 0, 50, 50]]


def test_tile_starts_cover_and_end_at_edge():
    assert r15b.tile_starts(200, 224) == [0]
    assert r15b.tile_starts(500, 224) == [0, 224, 276]
    assert r15b.tile_starts(448, 224) == [0, 224]


def test_clip_boxes_counts_half_inside():
    boxes = np.array([[10, 10, 20, 20], [215, 0, 20, 10], [220, 0, 20, 10],
                      [300, 0, 5, 5]], np.float32)  # fmt: skip
    out = r15b.clip_boxes(boxes, 0, 0, 224)
    assert len(out) == 3  # the last box is outside
    assert out[0].tolist() == [10, 10, 20, 20, 1]
    assert out[1, 4] == 0 and out[1, 2] == pytest.approx(9)  # 9 of 20 px inside
    assert out[2, 4] == 0


def test_plan_tiles_thins_empty_tiles_only():
    persons = np.array([[5, 5, 10, 20]], np.float32)
    rng = np.random.default_rng(0)
    full = r15b.plan_tiles(900, 500, persons, 224, 1.0, rng)
    assert len(full) == 5 * 3 + 3 * 2  # scale 1: 5x3 tiles, scale 0.5: 450x250 -> 3x2
    thin = r15b.plan_tiles(900, 500, persons, 224, 0.0, rng)
    assert [(s, x, y) for s, x, y, _ in thin] == [(1.0, 0, 0), (0.5, 0, 0)]


def test_match_prefers_counted_and_centre_mode():
    g = np.array([[0, 0, 10, 10, 0], [1, 1, 10, 10, 1]], np.float32)
    p = np.array([[0, 0, 10, 10, 0.9]], np.float32)
    assert r15b.match(p, g, "iou").tolist() == [1]
    tiny = np.array([[50, 50, 2, 2, 1]], np.float32)
    near = np.array([[48, 49, 8, 8, 0.5]], np.float32)  # centre (52, 53): within 4 px
    assert r15b.match(near, tiny, "centre").tolist() == [0]
    assert r15b.match(near, tiny, "iou").tolist() == [-1]
    far = np.array([[60, 60, 8, 8, 0.5]], np.float32)
    assert r15b.match(far, tiny, "centre").tolist() == [-1]


def test_ap_from_perfect_and_empty():
    assert r15b.ap_from(np.array([0.9, 0.8]), np.array([1, 1]), 2) == pytest.approx(1.0)
    assert r15b.ap_from(np.array([0.9, 0.8]), np.array([0, 1]), 2) == pytest.approx(
        0.25
    )
    assert r15b.ap_from(np.zeros(0), np.zeros(0, bool), 3) == 0.0
    assert math.isnan(r15b.ap_from(np.zeros(0), np.zeros(0, bool), 0))


def _split():
    gts = [
        np.array([[0, 0, 4, 4, 1], [100, 100, 40, 40, 1]], np.float32),
        np.array([[50, 50, 10, 10, 1], [0, 0, 10, 10, 0]], np.float32),
        np.zeros((0, 5), np.float32),
    ]
    offsets = np.cumsum([0] + [len(g) for g in gts])
    return gts, {
        "boxes": np.concatenate(gts), "offsets": offsets,
        "src": np.array([0, 0, 1]), "scale": np.array([1.0, 0.5, 1.0], np.float32),
    }  # fmt: skip


def test_tile_metrics_bins_groups_and_bootstrap():
    gts, split = _split()
    preds = [
        np.array([[100, 100, 40, 40, 0.9]], np.float32),  # hits the 40 px person
        np.array(
            [[50, 50, 10, 10, 0.8], [0, 0, 10, 10, 0.7]], np.float32
        ),  # 2nd: ignored
        np.array([[5, 5, 20, 20, 0.95]], np.float32),  # false alarm, 20 px
    ]
    top = np.array([0.9, 0.8, 0.95])
    m = r15b.tile_metrics(preds, top, split, n_boot=50, seed=1)
    pooled = m["pooled"]
    assert pooled["n_persons all"] == 3  # the uncounted box never counts
    assert pooled["n_persons 32+"] == 1 and pooled["ap50 32+"] == pytest.approx(1.0)
    assert pooled["ap50 0-8"] == 0.0  # the 4 px person is missed
    assert pooled["ap50 16-32"] != pooled["ap50 16-32"]  # no persons there: nan
    # all: hits at 0.9 and 0.8, the 20 px false alarm on top -> P = 0, 1/2, 2/3
    assert pooled["ap50 all"] == pytest.approx((1 / 3) * (2 / 3) * 2)
    assert m["scale 0.5"]["n_persons all"] == 1
    assert m["scale 0.5"]["ap50 all"] == pytest.approx(1.0)
    assert len(pooled["boot ap50"]) == 50
    mean, lo, hi = r15b.paired(pooled["boot ap50"], pooled["boot ap50"])
    assert mean == lo == hi == 0.0  # resamples without persons (nan) are skipped


def test_bootstrap_identity_weights_reproduce_ap():
    gts, split = _split()
    preds = [g[:, :4].tolist() for g in gts]
    preds = [np.array([b + [0.5] for b in p], np.float32).reshape(-1, 5) for p in preds]
    table = r15b.hit_table(preds, gts, "iou")
    frames = split["src"]
    w = np.ones(2, np.int64)
    plain = r15b.ap_bin(table)
    weighted = r15b.ap_bin(
        table, weights=(w[frames[table["det_tile"]]], w[frames[table["gt_tile"]]])
    )
    assert plain == pytest.approx(weighted) == pytest.approx(1.0)
