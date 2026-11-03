"""R19: crops, distances and the confirmer read-out metrics."""

import numpy as np

from genetic_coil_arena.experiments import r15_scout_brain as r15
from genetic_coil_arena.experiments import r19_confirmer as rc


def test_cut_centres_the_spot_and_pads_outside_the_frame():
    img = np.zeros((100, 200, 3), np.uint8)
    img[50, 120] = 255
    out = rc.cut(img, 120, 50, half=10)
    assert out.shape == (20, 20, 3) and out[10, 10, 0] == 255
    edge = rc.cut(img, 2, 3, half=10)
    assert (edge[:7, :, 0] == r15.PAD).all() and (edge[:, :8, 0] == r15.PAD).all()
    assert (edge[7:, 8:, 0] == 0).all()


def test_box_dist_is_zero_inside_and_euclidean_outside():
    persons = np.array([[10, 10, 4, 6]], np.float32)
    assert rc.box_dist(persons, 12, 13) == 0.0
    assert np.isclose(rc.box_dist(persons, 17, 20), np.hypot(3, 4))
    assert rc.box_dist(np.zeros((0, 4), np.float32), 0, 0) == float("inf")


def test_auc_and_threshold():
    pos, neg = np.array([3.0, 4.0, 5.0]), np.array([1.0, 2.0, 3.0])
    assert np.isclose(rc.auc(pos, neg), (3 + 3 + 2.5) / 9)
    thr = rc.threshold_at(np.arange(100.0), keep=0.95)
    assert (np.arange(100.0) >= thr).sum() == 95


def test_frame_bootstrap_resamples_whole_frames():
    frame = np.array([0, 0, 1, 2, 2, 2])
    w = rc.frame_bootstrap(frame, n_boot=50)
    assert w.shape == (50, 6)
    assert (w[:, 0] == w[:, 1]).all() and (w[:, 3] == w[:, 5]).all()
    assert (w.sum(1) >= 0).all()
