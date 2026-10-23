"""Tests for the domain-agnostic batched multi-agent API (envs.base)."""

import math

import numpy as np
import pytest

from genetic_coil_arena.envs.base import EnvSpec, MessageSpec, Observations, StepResult


class TestMessageSpec:
    def test_discrete_shape_dtype_and_bits(self):
        spec = MessageSpec("discrete", length=2, vocab_size=8)
        zeros = spec.zeros(4, 3)
        assert zeros.shape == (4, 3, 2) and zeros.dtype == np.int64
        assert spec.bits == pytest.approx(6.0)

    def test_continuous_shape_dtype(self):
        spec = MessageSpec("continuous", dim=5)
        zeros = spec.zeros(2, 3)
        assert zeros.shape == (2, 3, 5) and zeros.dtype == np.float32
        assert math.isinf(spec.bits)

    def test_none_channel(self):
        spec = MessageSpec("none")
        assert spec.bits == 0.0 and spec.zeros(2, 3).shape == (2, 3, 0)
        with pytest.raises(ValueError):
            spec.validate(np.zeros((2, 3, 0)), 2, 3)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"kind": "smoke"},
            {"kind": "discrete", "vocab_size": 1},
            {"kind": "discrete", "length": 0},
            {"kind": "continuous", "dim": 0},
            {"interval": 0},
        ],
    )
    def test_invalid_specs(self, kwargs):
        with pytest.raises(ValueError):
            MessageSpec(**kwargs)

    def test_validate_discrete(self):
        spec = MessageSpec("discrete", length=1, vocab_size=4)
        ok = spec.validate(np.array([[[3], [0]]], dtype=np.int32), 1, 2)
        assert ok.dtype == np.int64
        with pytest.raises(ValueError, match="shape"):
            spec.validate(np.zeros((1, 3, 1), dtype=np.int64), 1, 2)
        with pytest.raises(ValueError, match="integers"):
            spec.validate(np.zeros((1, 2, 1), dtype=np.float32), 1, 2)
        with pytest.raises(ValueError, match=r"\[0, 4\)"):
            spec.validate(np.full((1, 2, 1), 4), 1, 2)

    def test_validate_continuous(self):
        spec = MessageSpec("continuous", dim=2)
        ok = spec.validate(np.zeros((1, 2, 2), dtype=np.float64), 1, 2)
        assert ok.dtype == np.float32
        with pytest.raises(ValueError, match="floats"):
            spec.validate(np.zeros((1, 2, 2), dtype=np.int64), 1, 2)


def test_step_result_unpacks_and_batch_size():
    obs = Observations(
        specialists={"x": np.zeros((3, 2, 4))}, centre={"y": np.zeros((3, 5))}
    )
    assert obs.batch_size == 3
    result = StepResult(obs, np.zeros((3, 2)), np.zeros(3, dtype=bool), {})
    o, r, d, info = result
    assert o is obs and r.shape == (3, 2) and d.shape == (3,) and info == {}


def test_env_spec_is_frozen():
    spec = EnvSpec(1, 3, {}, {}, MessageSpec(), 10)
    with pytest.raises(AttributeError):
        spec.n_actions = 4  # type: ignore[misc]
