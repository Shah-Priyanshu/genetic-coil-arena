"""Test the shared message conventions (genetic_coil_arena.messages)."""

import ast
from pathlib import Path

import numpy as np
import pytest

from genetic_coil_arena import envs, evaluation, messages
from genetic_coil_arena.messages import MessageSpec, infer_kind

MODULE = Path(messages.__file__)


def test_one_message_spec_everywhere():
    assert envs.MessageSpec is evaluation.MessageSpec is MessageSpec


def test_module_is_dependency_free():
    """Only the standard library and numpy, so every layer can import it."""
    allowed = {"math", "dataclasses", "typing", "numpy"}
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert {a.name.split(".")[0] for a in node.names} <= allowed
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0 and node.module.split(".")[0] in allowed


def test_receivers_are_optional_and_checked():
    free = MessageSpec("discrete", length=2, vocab_size=4)
    assert free.validate(np.zeros((5, 7, 2), dtype=np.int64)).shape == (5, 7, 2)
    fixed = MessageSpec.discrete(2, 4, n_receivers=3)
    assert fixed.zeros(5).shape == (5, 3, 2)
    with pytest.raises(ValueError, match="shape"):
        fixed.validate(np.zeros((5, 7, 2), dtype=np.int64))
    with pytest.raises(ValueError, match="shape"):
        fixed.validate(np.zeros((5, 3, 2), dtype=np.int64), batch_size=4)
    assert fixed.bits == fixed.capacity_bits == 4.0


def test_infer_kind():
    assert infer_kind(np.zeros(1, dtype=np.int32)) == "discrete"
    assert infer_kind(np.zeros(1, dtype=np.float32)) == "continuous"
    with pytest.raises(TypeError):
        infer_kind(np.zeros(1, dtype=bool))


def test_squad_env_spec_fills_in_receivers():
    from genetic_coil_arena.envs.squad import SquadConfig, SquadEnv

    env = SquadEnv(SquadConfig(n_snakes=4))
    assert env.spec.message.n_receivers == 4
    with pytest.raises(ValueError, match="n_receivers"):
        SquadConfig(n_snakes=2, message=MessageSpec.discrete(1, 8, n_receivers=3))
