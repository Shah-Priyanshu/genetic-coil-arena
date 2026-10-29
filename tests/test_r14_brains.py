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
