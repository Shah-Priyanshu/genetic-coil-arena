"""R25: the camera comparison's bookkeeping on synthetic chains."""

import numpy as np

from genetic_coil_arena.experiments import r25_camera as r25

N_FRAMES = 40


def _chain(false_per_frame: int, people_per_frame: int) -> dict:
    frame = np.repeat(np.arange(N_FRAMES), false_per_frame + people_per_frame)
    person = np.tile([False] * false_per_frame + [True] * people_per_frame, N_FRAMES)
    return {"threshold": 0.5, "frame": frame, "person": person,
            "keep": np.ones(len(frame), bool), "auc": 0.9}  # fmt: skip


def _rep(false448: int, people448: int = 3) -> dict:
    chains = {"s224_seed0": _chain(4, 3), "s224_seed1": _chain(4, 3),
              "s448_seed0": _chain(false448, people448), "s448_seed1": _chain(false448, people448)}  # fmt: skip
    return r25.compare(chains=chains, persons=np.full(N_FRAMES, 5))


def test_ratio_people_and_verdict():
    rep = _rep(2)
    assert rep["n_persons"] == 5 * N_FRAMES
    assert abs(rep["pooled_ratio"][0] - 0.5) < 1e-9
    row = rep["seeds"]["seed0"]
    assert abs(row["224"]["false_per_frame"] - 4) < 1e-9
    assert abs(row["448"]["people_found"] - 0.6) < 1e-9 and row["people_diff"] == 0
    assert rep["verdict"] == "PASS"


def test_bars():
    assert _rep(3)["verdict"] == "NOT PASSED"  # 0.75x: above the pooled bar
    assert _rep(4)["verdict"] == "KILL"
    assert _rep(2, people448=2)["verdict"] == "NOT PASSED"  # 20 points fewer people


def test_render_lists_both_cameras():
    text = r25.render(_rep(2))
    assert "**PASS.**" in text and "| 0 | 224 px |" in text and "| 1 | 448 px |" in text


def _chains_c(false448: int) -> dict:
    return {"s224_seed0": _chain(4, 3), "s224_seed1": _chain(4, 3),
            "s448_seed0": _chain(false448, 3), "s448_seed1": _chain(false448, 3)}  # fmt: skip


def test_sample_c_is_fixed_and_repeats_spots():
    chains = _chains_c(4)
    pool = r25.sample_c(chains)
    assert pool == r25.sample_c(chains)
    per = {t: sum(tag == t for tag, _ in pool) for t in chains}
    assert sum(per.values()) == 4 * 4 * N_FRAMES + r25.REPEATS  # every false alarm, fewer than N_SPOTS
    assert len(set(pool)) == 4 * 4 * N_FRAMES
    assert all(not chains[t]["person"][i] for t, i in pool)


def test_clear_compare_counts_only_not_person():
    chains = _chains_c(4)
    pool = r25.sample_c(chains)
    # half of the 224 chains' false alarms are clear non-persons, a quarter of the 448's
    labels = {str(n): ("not_person" if i % 4 < (2 if "224" in t else 1) else "person")
              for n, (t, i) in enumerate(pool)}  # fmt: skip
    rep = r25.clear_compare({"spots": pool, "labels": labels}, chains, N_FRAMES,
                            {"seed0": 0.0, "seed1": 0.0})  # fmt: skip
    assert abs(rep["pooled_ratio"][0] - 0.5) < 1e-9
    assert rep["repeats_agree"] == [r25.REPEATS, r25.REPEATS]
    assert rep["chains"]["s224_seed0"]["not_person"] == 2 * N_FRAMES
    assert rep["verdict"] == "PASS"
    text = r25.render_c(rep)
    assert text.startswith("# R25c:") and "| 224 px, seed0 | R19's |" in text
