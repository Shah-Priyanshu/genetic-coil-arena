"""Test envelopes: validation, JSON round trips and the import boundary."""

import ast
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from genetic_coil_arena.protocol import (
    PAYLOAD_TYPES,
    PROTOCOL_VERSION,
    Envelope,
    EnvelopeError,
    InMemoryTensorStore,
    canonical_json,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "genetic_coil_arena"
FORBIDDEN = (
    "snake_core",
    "evo",
    "envs",
    "torch",
    "jax",
    "tensorflow",
    "keras",
    "numba",
)


def make(**overrides):
    fields = {
        "sender": "a",
        "recipients": ("b",),
        "payload_type": "json",
        "payload": {"x": 1},
    }
    fields.update(overrides)
    return Envelope(**fields)


class TestEnvelopeFields:
    def test_has_spec_fields(self):
        names = set(make().to_dict())
        assert names == {
            "v",
            "msg_id",
            "trace_id",
            "parent_id",
            "sender",
            "recipients",
            "task_id",
            "clock",
            "payload_type",
            "payload_encoding",
            "payload",
        }

    def test_defaults(self):
        env = make()
        assert env.v == PROTOCOL_VERSION
        assert env.parent_id is None and env.task_id is None
        assert env.clock == 0
        assert len(env.msg_id) == 32 and len(env.trace_id) == 32
        assert env.manifest is None and not env.is_signed

    def test_unique_ids(self):
        assert make().msg_id != make().msg_id

    def test_frozen(self):
        env = make()
        with pytest.raises(FrozenInstanceError):
            env.sender = "c"

    def test_hashable_and_usable_in_sets(self):
        env = make()
        assert len({env, env}) == 1

    def test_single_recipient_string(self):
        assert make(recipients="b").recipients == ("b",)

    def test_payload_types(self):
        assert PAYLOAD_TYPES == ("json", "codes", "latent_ref")


class TestEnvelopeValidation:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"sender": ""},
            {"recipients": ()},
            {"recipients": ("b", "")},
            {"payload_type": "pickle"},
            {"clock": -1},
            {"clock": 1.5},
            {"clock": True},
            {"v": 2},
            {"v": 0},
            {"task_id": ""},
            {"parent_id": 3},
            {"payload": {"x": float("nan")}},
            {"payload": object()},
            {"payload_encoding": {"bad": object()}},
            {"payload_encoding": "x"},
        ],
    )
    def test_rejects(self, overrides):
        with pytest.raises(EnvelopeError):
            make(**overrides)

    def test_arrays_never_inlined(self):
        with pytest.raises(EnvelopeError, match="never inlined"):
            make(payload=np.zeros(3))
        with pytest.raises(EnvelopeError):
            make(payload_type="codes", payload=np.zeros(3, dtype=np.int64))

    def test_codes_payload_normalized(self):
        env = make(payload_type="codes", payload=[np.int64(1), 2, 3])
        assert env.payload == (1, 2, 3)
        assert all(type(s) is int for s in env.payload)

    @pytest.mark.parametrize("payload", ["abc", [1, 2.0], [True], 5])
    def test_codes_payload_rejects_non_int(self, payload):
        with pytest.raises(EnvelopeError):
            make(payload_type="codes", payload=payload)

    def test_codes_checked_against_encoding(self):
        make(payload_type="codes", payload=[0, 4], payload_encoding={"V": 5, "L": 2})
        with pytest.raises(EnvelopeError, match="outside"):
            make(payload_type="codes", payload=[0, 5], payload_encoding={"V": 5})
        with pytest.raises(EnvelopeError, match="L=3"):
            make(payload_type="codes", payload=[0, 1], payload_encoding={"L": 3})
        with pytest.raises(EnvelopeError, match="non-negative int"):
            make(payload_type="codes", payload=[0], payload_encoding={"V": "5"})

    def test_latent_ref_requires_reference_keys(self):
        with pytest.raises(EnvelopeError, match="latent_ref"):
            make(payload_type="latent_ref", payload={"key": "k"})
        ref = InMemoryTensorStore().put(np.ones(2, dtype=np.float32))
        assert make(payload_type="latent_ref", payload=ref).payload == ref


class TestSerialization:
    def test_json_round_trip(self):
        env = make(
            payload={"nested": [1, "two", None]}, task_id="t", parent_id="p", clock=7
        )
        again = Envelope.from_json(env.to_json())
        assert again == env

    def test_codes_round_trip(self):
        env = make(
            payload_type="codes", payload=(3, 1, 4), payload_encoding={"V": 8, "L": 3}
        )
        assert Envelope.from_json(env.to_json()) == env

    def test_latent_round_trip(self):
        ref = InMemoryTensorStore().put(np.arange(4, dtype=np.float32))
        env = make(payload_type="latent_ref", payload=ref)
        assert Envelope.from_json(env.to_json()) == env

    def test_canonical_json_is_deterministic(self):
        env = make(payload={"b": 1, "a": 2})
        assert env.to_json() == canonical_json(json.loads(env.to_json()))
        assert '"a":2,"b":1' in env.to_json()

    def test_manifest_only_when_signed(self):
        assert "manifest" not in make().to_dict()
        signed = make().with_manifest({"alg": "x"})
        assert signed.to_dict()["manifest"] == {"alg": "x"}
        assert Envelope.from_dict(signed.to_dict()) == signed

    def test_from_dict_rejects_missing_and_unknown(self):
        data = make().to_dict()
        with pytest.raises(EnvelopeError, match="missing"):
            Envelope.from_dict({k: v for k, v in data.items() if k != "clock"})
        with pytest.raises(EnvelopeError, match="unknown"):
            Envelope.from_dict({**data, "extra": 1})

    def test_from_dict_string_recipient(self):
        data = {**make().to_dict(), "recipients": "bob"}
        assert Envelope.from_dict(data).recipients == ("bob",)

    def test_future_version_rejected(self):
        data = {**make().to_dict(), "v": PROTOCOL_VERSION + 1}
        with pytest.raises(EnvelopeError, match="version"):
            Envelope.from_dict(data)

    @given(
        payload=st.recursive(
            st.none()
            | st.booleans()
            | st.integers()
            | st.text(max_size=5)
            | st.floats(allow_nan=False, allow_infinity=False),
            lambda inner: st.lists(inner, max_size=3)
            | st.dictionaries(st.text(max_size=3), inner, max_size=3),
            max_leaves=10,
        ),
        clock=st.integers(min_value=0, max_value=2**40),
    )
    def test_json_round_trip_property(self, payload, clock):
        env = make(payload=payload, clock=clock)
        again = Envelope.from_json(env.to_json())
        assert again.to_json() == env.to_json()


class TestReply:
    def test_reply_threads_ids(self):
        request = make(task_id="t", clock=4)
        answer = request.reply("b", payload={"ok": True})
        assert answer.recipients == ("a",)
        assert answer.parent_id == request.msg_id
        assert answer.task_id == "t" and answer.trace_id == request.trace_id
        assert answer.clock == 4 and answer.msg_id != request.msg_id

    def test_reply_overrides(self):
        answer = make().reply("b", payload=[1], recipients=["c", "d"], clock=9)
        assert answer.recipients == ("c", "d") and answer.clock == 9


def _imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, 0
        elif isinstance(node, ast.ImportFrom):
            yield node.module or "", node.level


@pytest.mark.parametrize("package", ["protocol", "runtime", "learning", "translator"])
def test_domain_agnostic_imports(package):
    files = sorted((PACKAGE_ROOT / package).glob("*.py"))
    assert files
    for path in files:
        for module, level in _imports(path):
            parts = module.split(".")
            if level == 0:
                assert parts[0] not in FORBIDDEN, f"{path.name} imports {module}"
                if parts[0] == "genetic_coil_arena":
                    assert parts[1] in (
                        "protocol",
                        "runtime",
                    ), f"{path.name} imports {module}"
            elif level >= 2:
                assert parts[0] in (
                    "protocol",
                    "runtime",
                ), f"{path.name} imports ..{module}"
