"""R15 scout brain: letterbox geometry, CenterNet targets and decoding, AP50 and AUC."""

import numpy as np
import pytest

from genetic_coil_arena.experiments import r15_scout_brain as r15

SIZE = 160


def test_person_boxes_scale_and_skip_crowds_and_other_classes():
    anns = [
        {"category_id": 1, "bbox": [10, 20, 30, 40], "iscrowd": 0},
        {"category_id": 1, "bbox": [0, 0, 100, 100], "iscrowd": 1},
        {"category_id": 3, "bbox": [0, 0, 10, 10], "iscrowd": 0},
    ]
    boxes = r15.person_boxes(anns, r15.letterbox_scale(640, 480, SIZE))
    assert boxes.shape == (1, 4) and np.allclose(boxes[0], [2.5, 5, 7.5, 10])


def _round_trip(boxes, flip=False):
    t = r15.targets([boxes], SIZE, np.array([flip]))
    g = SIZE // r15.STRIDE
    heat = t["heat"].copy()
    wh = np.zeros((1, 2, g, g), np.float32)
    off = np.zeros((1, 2, g, g), np.float32)
    iy, ix = np.divmod(t["idx"], g)
    wh[0, :, iy, ix] = t["wh"]
    off[0, :, iy, ix] = t["off"]
    return t, r15.decode(heat, wh, off, min_score=0.9)[0]


def test_targets_decode_round_trip():
    boxes = np.array([[20, 30, 24, 60], [100, 40, 40, 80]], np.float32)
    t, dec = _round_trip(boxes)
    assert t["heat"].max() == 1.0 and len(t["idx"]) == 2
    assert len(dec) == 2
    assert np.allclose(np.sort(dec[:, :4], axis=0), np.sort(boxes, axis=0), atol=1e-3)
    _, flipped = _round_trip(boxes[:1], flip=True)
    assert np.isclose(flipped[0, 0], SIZE - 20 - 24, atol=1e-3)


def test_ap50_perfect_partial_and_keep():
    gts = [np.array([[0, 0, 10, 10], [50, 50, 10, 40]], np.float32)]
    perfect = [np.array([[0, 0, 10, 10, 0.9], [50, 50, 10, 40, 0.8]], np.float32)]
    assert r15.ap50(perfect, gts) == pytest.approx(1.0)
    half = [np.array([[0, 0, 10, 10, 0.9], [100, 100, 5, 5, 0.95]], np.float32)]
    assert 0.2 < r15.ap50(half, gts) < 0.6
    tall_only = r15.ap50(perfect, gts, keep=lambda g: g[:, 3] >= 32)
    assert tall_only == pytest.approx(1.0)


def test_auc():
    assert r15.auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert r15.auc(np.array([0.5, 0.5]), np.array([0, 1])) == 0.5


def test_model_shapes_and_macs():
    torch = pytest.importorskip("torch")
    model = r15.build_model(0.25)
    out = model(torch.zeros(2, 3, SIZE, SIZE))
    assert out.shape == (2, 5, SIZE // r15.STRIDE, SIZE // r15.STRIDE)
    assert 10_000 < sum(p.numel() for p in model.parameters()) < 400_000
    assert r15.count_macs(model, SIZE) > 1_000_000
