"""Drone swarm lab (R14): environment, radio, codec, scripted swarm and G13 verdict."""

from dataclasses import replace

import numpy as np
import pytest

from genetic_coil_arena.envs.swarm import (
    N_ACTIONS,
    HandCodec,
    ScriptedSwarm,
    SwarmConfig,
    SwarmEnv,
    heading_action,
    run_swarm,
)
from genetic_coil_arena.experiments import r14_swarm_talk as r14
from genetic_coil_arena.messages import MessageSpec

SEEDS = list(range(100, 140))


def test_codec_round_trip_and_bytes():
    codec = HandCodec(2, 5, 30.0)
    pos = np.array([[0.1, 29.9], [15.0, 7.4]])
    val = np.array([0.0, 0.49])
    packets = codec.encode(pos, val)
    assert packets.shape == (2, 2) and packets.min() >= 0 and packets.max() <= 255
    p, v = codec.decode(packets)
    assert np.all(np.abs(p - pos) <= codec.cell / 2 + 1e-9)
    assert v[0] == 0.0 and abs(v[1] - 0.49) < 0.05
    with pytest.raises(ValueError):
        HandCodec(1, 4, 30.0)  # no value bits left


def test_same_seed_same_episode_whatever_the_batch():
    env = SwarmEnv()
    a = env.reset([7, 8]).specialists["pos_est"][1]
    b = env.reset([8]).specialists["pos_est"][0]
    assert np.array_equal(a, b)
    assert np.all(env.source >= 8.0) and np.all(env.source <= 28.0)


def test_moves_radio_and_loss():
    cfg = replace(SwarmConfig(), loss=1.0)
    env = SwarmEnv(cfg)
    env.reset([1])
    before = env.pos.copy()
    msgs = np.full((1, 6, 4), 9, dtype=np.int64)
    obs, *_ = env.step(np.full((1, 6), 1), msgs)  # action 1: east
    assert np.allclose(env.pos[..., 0] - before[..., 0], 0.5) or np.any(env.pos == 30.0)
    assert not obs.specialists["inbox_ok"].any() and obs.specialists["inbox"].sum() == 0
    env = SwarmEnv(SwarmConfig())
    env.reset([1])
    obs, *_ = env.step(np.zeros((1, 6), dtype=np.int64), msgs)
    ok = obs.specialists["inbox_ok"][0]
    assert ok.sum() == 30 and not ok.diagonal().any()
    assert (obs.specialists["inbox"][0][ok] == 9).all()


def test_only_gas_drones_smell_and_camera_confirms():
    env = SwarmEnv()
    env.reset([3])
    env.pos[0, 3] = env.source[0]  # a camera drone over the source
    _, rewards, done, info = env.step(np.zeros((1, 6), dtype=np.int64))
    assert done[0] and info["confirmed"][0] and rewards[0, 0] == 1.0
    obs = env._observe(np.zeros((1, 6, 6, 4), np.int64), np.zeros((1, 6, 6), bool))
    assert (obs.specialists["gas"][0, 3:] == 0).all()


def test_heading_action():
    acts = heading_action(np.array([[1.0, 0.0], [0.0, 1.0], [0.1, 0.0]]), 0.25)
    assert acts.tolist() == [1, 3, 0] and acts.max() < N_ACTIONS


def test_talking_beats_silence_on_a_small_batch():
    cfg = replace(SwarmConfig(), message=MessageSpec("discrete", 8, 256))
    silent = run_swarm(SwarmEnv(cfg), ScriptedSwarm(cfg), SEEDS)
    talk = run_swarm(
        SwarmEnv(cfg),
        ScriptedSwarm(cfg, HandCodec(8, 24, cfg.arena), camera_radius=3.0),
        SEEDS,
    )
    assert talk["confirmed"].mean() > silent["confirmed"].mean()


def _cells(silent, full, one):
    d = r14.PRIMARY["drift"]
    return {
        r14.cell_key(0, 0.0, d): {"success": silent},
        r14.cell_key(8, 0.0, d): {"success": full},
        r14.cell_key(1, 0.3, d): {"success": one},
    }


@pytest.mark.parametrize(
    "one, verdict", [([0.75, 0.77], "GOOD SIGN"), ([0.88, 0.9], "RADIO DOES NOT BIND")]
)
def test_g13_verdict(one, verdict):
    cells = _cells([0.6, 0.62], [0.95, 0.93], one)
    assert r14.g13_verdict(cells)["verdict"] == verdict


def test_g13_talk_does_not_matter():
    cells = _cells([0.9, 0.92], [0.95, 0.93], [0.9, 0.9])
    assert r14.g13_verdict(cells)["verdict"] == "TALK DOES NOT MATTER"


def test_report_smoke():
    hand = {b: {"pos_bits": r14.POS_BITS[b][0], "radius": 2.0, "tune_success": 0.5}
            for b in r14.BUDGETS}  # fmt: skip
    cells = r14.run_grid(hand, per_rep=4, reps=2)
    md = r14.render(r14.build_report(hand, cells))
    assert "G13:" in md and "| 8 B |" in md
