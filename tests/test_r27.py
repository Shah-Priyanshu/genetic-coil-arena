"""R27: read-out, variant choice and verdict for the VLM final check, on synthetic scores."""

import numpy as np

from genetic_coil_arena.experiments import r27_vlm_check as r27


def _data(n_people=200, n_clear=40, n_unclear=20):
    label = np.array(["person"] * n_people + ["not_person"] * n_clear + ["ambiguous"] * n_unclear)
    source = np.array(["blind"] * (n_people // 2) + ["visdrone"] * (n_people - n_people // 2)
                      + ["blind"] * (n_clear + n_unclear))  # fmt: skip
    return {"label": label, "source": source}


def _scores(data, sep, seed=0):
    rng = np.random.default_rng(seed)
    base = np.where(
        data["label"] == "person", sep, np.where(data["label"] == "not_person", 0, sep / 2)
    )
    return base + rng.normal(0, 1, len(base))


def test_threshold_keeps_people_and_read_out():
    d = _data()
    s = _scores(d, 4.0)
    thr = r27.threshold_at(s, d["label"])
    res = r27.evaluate(s, d, thr)
    assert abs(res["people_kept"][0] - 0.95) < 0.02
    assert res["n"] == {"person": 200, "not_person": 40, "ambiguous": 20}
    assert res["clear_rejected"][0] > 0.8 and res["auc"] > 0.95


def test_verdict_bars():
    good = {"clear_rejected": [0.6, 0.45, 0.73], "people_kept": [0.95, 0.93, 0.97],
            "people_kept_blind": [0.94, 0.9, 0.97], "people_kept_visdrone": [0.96, 0.93, 0.98]}  # fmt: skip
    assert r27.verdict(good) == "PASS"
    assert r27.verdict({**good, "clear_rejected": [0.55, 0.28, 0.7]}) == "NOT PASSED"
    assert r27.verdict({**good, "people_kept_blind": [0.88, 0.84, 0.92]}) == "NOT PASSED"
    assert r27.verdict({**good, "clear_rejected": [0.2, 0.1, 0.35]}) == "KILL"


def test_choose_picks_the_variant_rejecting_more():
    d = _data()
    variant, res = r27.choose({"plain": _scores(d, 1.0), "marked": _scores(d, 4.0)}, d)
    assert variant == "marked" and set(res) == {"plain", "marked"}


def test_view_marks_the_centre():
    crop = np.zeros((144, 144, 3), np.uint8)
    plain, marked = np.asarray(r27.view(crop, False)), np.asarray(r27.view(crop, True))
    assert plain.shape == (r27.SIZE, r27.SIZE, 3) and plain.max() == 0
    c = r27.SIZE // 2
    assert tuple(marked[c, c + 55]) == (255, 0, 255) and tuple(marked[c, c]) == (0, 0, 0)
