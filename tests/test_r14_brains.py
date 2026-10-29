"""R14 step 2: learned swarm brains (population batching, training and evaluation smoke)."""

from dataclasses import replace

import numpy as np

from genetic_coil_arena.envs.swarm.brains import N_FEATURES
from genetic_coil_arena.experiments import r14_brains as rb

SMALL = replace(
    rb.BrainSettings(), generations=2, population=4, episodes=2, eval_every=1,
    val_episodes=4, progress_s=0,
)  # fmt: skip


def test_members_play_their_own_parameters():
    _, pol = rb.make_policy(SMALL, 3)
    n = pol.gas_net.n_params + pol.cam_net.n_params
    thetas = np.random.default_rng(0).normal(size=(2, n))
    both = rb.rollout(SMALL, 3, thetas, [11, 12, 13])
    one = rb.rollout(SMALL, 3, thetas[1:], [11, 12, 13])
    assert both.shape == (2, 3) and np.array_equal(both[1], one[0])


def test_features_shape_and_validity():
    cfg, pol = rb.make_policy(SMALL, 3)
    pol.set_params(
        *rb.split(np.zeros(pol.gas_net.n_params + pol.cam_net.n_params), pol)
    )
    from genetic_coil_arena.envs.swarm import SwarmEnv

    obs = SwarmEnv(cfg).reset([5])
    pol.reset([5])
    pol.act(obs)
    f, dest, valid, search = pol.features(
        obs.specialists["pos_est"].astype(float), np.zeros((1, 6, 6, 3))
    )
    assert f.shape == (1, 6, 8, N_FEATURES) and valid[..., 0].all()
    assert not valid[..., 1:7].any() and not search[..., 0].any()


def test_train_save_load_evaluate(tmp_path):
    res = rb.train_run(0, SMALL)
    rb.save_run(res, tmp_path)
    runs = rb.load_runs([tmp_path])
    assert len(runs) == 1 and runs[0]["theta"].shape == res["theta"].shape
    rep = rb.evaluate(runs, reps=1, per_rep=3)
    assert set(rep["by_k"]) == set(rb.EVAL_KS)
    assert "Sources" in rb.render(rep, runs)


def test_random_conditions_are_valid_configs():
    rng = np.random.default_rng(0)
    for _ in range(50):
        over = rb.random_condition(rng)
        cfg, pol = rb.make_policy(SMALL, 3, over)
        assert cfg.n_sources == over["n_sources"] and cfg.failures == over["failures"]
    for over in rb.VAL_MIX:
        rb.make_policy(SMALL, 3, over)


def test_randomised_training_and_robust_eval(tmp_path):
    from genetic_coil_arena.experiments import r14_robust as rr

    res = rb.train_run(1, replace(SMALL, randomise=True))
    assert res["settings"]["randomise"] and len(res["curve"]) == 3
    rb.save_run(res, tmp_path)
    runs = rb.load_runs([tmp_path])
    sets = {"step2": runs, "random": runs}
    conds = ["nominal", "1 drone down", "4 drones (2+2)"]
    cells = dict(rr.play_task(t) for t in rr.tasks(sets, 2, 3, conds))
    assert len(cells) == 3 * (len(rr.REFS) + 2)
    rep = rr.summarise(cells, sets)
    assert rep["conditions"] == conds and "robust" in rep
    r = rep["randomised"]  # identical sets: no difference, so 3b cannot pass
    assert r["nominal_minus_step2"] == 0.0 and not r["pass"]
    md = rr.render(rep, {"step2": 1, "random": 1}, 6)
    assert "3a, robust" in md and "4 drones (2+2) *" in md


def test_robust_verdicts_on_made_up_cells():
    from genetic_coil_arena.experiments import r14_robust as rr

    runs = [{"run": 0}, {"run": 1}]
    sets = {"step2": runs, "random": runs[:1]}
    cells = {}
    up = np.linspace(0.0, 0.02, 10)
    for cond, gain, rgain in (("nominal", 0.15, 0.14), ("loss 70%", 0.0, 0.1),
                              ("2 drones down", 0.1, 0.12)):  # fmt: skip
        cells[(cond, "hand 2B", -1)] = {"all": list(0.5 + up), "found": [2.0] * 10}
        cells[(cond, "sensed oracle", -1)] = {"all": [0.9] * 10, "found": [2.9] * 10}
        for r in runs:
            cells[(cond, "step2", r["run"])] = {"all": list(0.5 + gain + up),
                                                 "found": [2.5] * 10}  # fmt: skip
        cells[(cond, "random", 0)] = {"all": list(0.5 + rgain + up[::-1]),
                                       "found": [2.6] * 10}  # fmt: skip
    rep = rr.summarise(cells, sets)
    assert rep["robust"] == {"pass": False, "fails": ["loss 70%"]}
    assert rep["randomised"]["pass"]  # +6 points under stress, -1 at nominal
