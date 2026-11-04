"""R20 (closer look) and R21 (ground-station LoRA) helpers."""

import json

import numpy as np
import pytest

from genetic_coil_arena.experiments import r20_closer_look as rl


def test_chain_thresholds_keep_the_share_of_people():
    rng = np.random.default_rng(0)
    person = rng.random(4000) < 0.5
    a = rng.normal(person * 2.0, 1.0)
    b = rng.normal(person * 2.0, 1.0)
    ta, tb, q = rl.chain_thresholds(a, b, person, keep=0.95)
    kept = ((a >= ta) & (b >= tb))[person].mean()
    assert 0.95 <= kept < 0.96
    assert q > 0.95  # each model alone keeps more than the pair


def test_identical_models_need_no_stricter_share():
    rng = np.random.default_rng(1)
    person = rng.random(2000) < 0.5
    a = rng.normal(person * 2.0, 1.0)
    _, _, q = rl.chain_thresholds(a, a, person, keep=0.95)
    assert q <= 0.951  # one grid step for the quantile


def test_boot_ratio_and_spearman():
    num = np.array([1, 0, 1, 0], bool)
    den = np.array([1, 1, 1, 1], bool)
    w = np.ones((3, 4))
    assert np.allclose(rl.boot_ratio(num, den, w), 0.5)
    assert rl.spearman(np.arange(5.0), np.arange(5.0) ** 3) == pytest.approx(1.0)


def test_reduce_halves_the_frame():
    pytest.importorskip("PIL")
    img = np.zeros((10, 12, 3), np.uint8)
    img[:2, :2] = 200
    out = rl.reduce(img)
    assert out.shape == (5, 6, 3) and out[0, 0, 0] == 200


def test_first_json_and_target_reply():
    from genetic_coil_arena.experiments import r21_ground_lora as rg

    text = 'Sure!\n```json\n{"refusal": "", "params": {"behavior": "rth", "x": "}"}}\n```'
    assert json.loads(rg.first_json(text))["params"]["behavior"] == "rth"
    assert rg.first_json("no json here") == "no json here"
    ok = {"expect": "ok", "params": {"behavior": "hover", "alt_m": 6.1}}
    assert json.loads(rg.target_reply(ok)) == {"refusal": "", "clarify": "",
                                               "params": ok["params"]}  # fmt: skip
    ask = json.loads(rg.target_reply({"expect": "clarify", "tags": ["perimeter"]}))
    assert ask["clarify"] and not ask["refusal"]
    no = json.loads(rg.target_reply({"expect": "refuse", "tags": ["single-drone"]}))
    assert "drone" in no["refusal"]


def test_check_labels_flags_bad_items():
    from genetic_coil_arena.experiments import r21_ground_lora as rg

    items = [
        {"id": "x1", "text": "hover at 12 m", "expect": "ok",
         "params": {"behavior": "hover", "alt_m": 12}},
        {"id": "x2", "text": "hover at 12 m please", "expect": "ok",
         "params": {"behavior": "hover", "alt_m": 13}},
        {"id": "x3", "text": "circle the old pond", "expect": "ok",
         "params": {"behavior": "perimeter", "area": "pond"}},
        {"id": "x4", "text": "Hover at 12 m!", "expect": "clarify"},
    ]  # fmt: skip
    bad = dict(rg.check_labels(items))
    assert set(bad) == {"x2", "x3", "x4"}


def test_r22_folds_keep_sequences_together():
    from genetic_coil_arena.experiments import r22_clean_labels as rc

    names = np.array([f"{s:07d}_{k:05d}_d_0000001.jpg" for s in range(10) for k in range(3)])
    f = rc.folds(names)
    for s in range(10):
        assert len(set(f[s * 3 : s * 3 + 3])) == 1
    assert 0 < f.sum() < len(f)


def test_r22_pseudo_mask_and_boxes():
    from genetic_coil_arena.experiments import r22_clean_labels as rc

    meta = {"person": np.array([1, 1, 1, 1, 0, 0, 0, 0], bool),
            "dist": np.array([0, 0, 0, 0, 5.0, 0.0, 5.0, 5.0]),
            "frame": np.array([0, 0, 0, 0, 0, 0, 1, 1]),
            "fx": np.full(8, 50.0), "fy": np.full(8, 40.0), "size": np.full(8, 10.0)}  # fmt: skip
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.95, 0.95, 0.65, 0.1])
    mask, thr = rc.pseudo_mask(meta, score, np.zeros(8, int))
    assert list(np.flatnonzero(mask)) == [4]  # 5 is inside a box, 6 below, 7 far below
    boxes = rc.pseudo_boxes(meta, mask)
    assert np.allclose(boxes[0], [[45, 35, 10, 10]])


def test_r22_clear_rate():
    from genetic_coil_arena.experiments import r22_clean_labels as rc

    n_fa = np.array([4.0, 0.0])  # 2 false alarms per frame on average
    out = rc.clear_rate(n_fa, np.array([2.0, 0]), np.array([1.0, 0]), np.ones((1, 2)))
    assert out[0] == pytest.approx(1.0)  # half the sample is clear
