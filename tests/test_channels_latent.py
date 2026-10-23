"""Test the latent channel: wire formats, envelopes in a TensorStore, and the projector."""

import numpy as np
import pytest

from genetic_coil_arena.channels import (
    LATENT_CODEC,
    dequantize_int8,
    envelopes_to_latents,
    latent_encoding,
    latents_to_envelopes,
    message_wire_nbytes,
    quantize_int8,
    round_to_bfloat16,
    through_wire,
    wire_nbytes,
)
from genetic_coil_arena.protocol import (
    Envelope,
    EnvelopeError,
    InMemoryTensorStore,
    TensorDigestError,
)


@pytest.fixture
def messages():
    rng = np.random.default_rng(0)
    return rng.standard_normal((3, 4, 16)).astype(np.float32) * 50


class TestWire:
    def test_sizes(self):
        assert wire_nbytes(1, 2048) == 4096
        assert wire_nbytes(16, 2048) == 16 * 2048 * 2
        assert wire_nbytes(4, 2048, "float32") == 4 * 2048 * 4
        assert wire_nbytes(4, 2048, "int8") == 4 * 2048 + 4 * 2
        with pytest.raises(ValueError):
            wire_nbytes(1, 8, "fp8")

    def test_bfloat16_rounding(self):
        x = np.array([1.0, 0.1, -3.14159, 1e-3, 65504.0, 0.0], dtype=np.float32)
        r = round_to_bfloat16(x)
        assert r.dtype == np.float32
        np.testing.assert_array_equal(r[[0, 5]], [1.0, 0.0])  # exactly representable
        nonzero = x != 0
        assert np.all(np.abs(r - x)[nonzero] <= np.abs(x)[nonzero] * 2.0**-8)
        np.testing.assert_array_equal(round_to_bfloat16(r), r)  # idempotent
        # the low 16 bits are cleared
        assert not (r.view(np.uint32) & 0xFFFF).any()

    def test_bfloat16_ties_round_to_even(self):
        # 1 + 2^-8 sits exactly between bf16 neighbours 1 and 1 + 2^-7: rounds to 1
        assert round_to_bfloat16(np.array([1 + 2.0**-8], dtype=np.float32))[0] == 1.0
        # 1 + 3 * 2^-8 sits between 1 + 2^-7 and 1 + 2^-6: rounds to the even one
        assert round_to_bfloat16(np.array([1 + 3 * 2.0**-8], dtype=np.float32))[
            0
        ] == pytest.approx(1 + 2.0**-6)

    def test_int8_round_trip_error_is_bounded_by_the_scale(self, messages):
        codes, scales = quantize_int8(messages)
        assert codes.dtype == np.int8 and scales.shape == messages.shape[:-1]
        assert scales.dtype == np.float16
        error = np.abs(dequantize_int8(codes, scales) - messages)
        bound = scales.astype(np.float32)[..., None] / 2 * 1.01 + 1e-6
        # float16 scales are rounded, so allow a little slack beyond scale / 2
        assert np.all(error <= bound + np.abs(messages) * 2.0**-9)

    def test_int8_zero_vector_and_outliers(self):
        zeros = np.zeros((1, 2, 8), dtype=np.float32)
        np.testing.assert_array_equal(through_wire(zeros, "int8"), zeros)
        spiky = np.full((1, 1, 8), 0.01, dtype=np.float32)
        spiky[0, 0, 0] = (
            100.0  # one outlier flattens the rest: the cost of absmax scaling
        )
        out = through_wire(spiky, "int8")
        assert out[0, 0, 0] == pytest.approx(100.0, rel=1e-2)
        assert np.all(out[0, 0, 1:] == 0)

    def test_through_wire_float32_is_identity(self, messages):
        np.testing.assert_array_equal(through_wire(messages, "float32"), messages)


class TestEnvelopes:
    def test_round_trip(self, messages):
        store = InMemoryTensorStore("kv")
        envs = latents_to_envelopes(
            messages, store, "extractor", "normalizer", layer=14, pooling="last"
        )
        assert len(envs) == 3 and len(store) == 3
        back = envelopes_to_latents(envs, store, "normalizer", verify=True)
        assert back.dtype == np.float32 and back.shape == messages.shape
        np.testing.assert_array_equal(back, round_to_bfloat16(messages))

    def test_envelope_header_and_encoding(self, messages):
        store = InMemoryTensorStore()
        env = latents_to_envelopes(
            messages,
            store,
            "a",
            "b",
            task_ids=["t0", "t1", "t2"],
            trace_ids=["x0", "x1", "x2"],
            parent_ids=[None, "p", None],
            clock=7,
            layer=14,
        )[1]
        assert env.payload_type == "latent_ref" and env.recipients == ("b",)
        assert (env.task_id, env.trace_id, env.parent_id, env.clock) == (
            "t1",
            "x1",
            "p",
            7,
        )
        assert env.payload["shape"] == [4, 16] and env.payload["store"] == "memory"
        enc = env.payload_encoding
        assert enc == latent_encoding(4, 16, "bfloat16", layer=14)
        assert enc["codec"] == LATENT_CODEC and enc["wire_bytes"] == 4 * 16 * 2
        assert message_wire_nbytes(env) == 128
        assert Envelope.from_json(env.to_json()) == Envelope.from_dict(env.to_dict())

    def test_envelopes_never_inline_the_tensor(self, messages):
        env = latents_to_envelopes(messages, InMemoryTensorStore(), "a", "b")[0]
        assert len(env.to_json()) < 600  # a reference, not 4 x 16 floats of payload

    def test_int8_wire_dtype_is_applied_and_sized(self, messages):
        store = InMemoryTensorStore()
        envs = latents_to_envelopes(messages, store, "a", "b", wire_dtype="int8")
        back = envelopes_to_latents(envs, store)
        np.testing.assert_array_equal(back, through_wire(messages, "int8"))
        assert message_wire_nbytes(envs[0]) == 4 * 16 + 4 * 2

    def test_verify_detects_a_swapped_tensor(self, messages):
        store = InMemoryTensorStore()
        envs = latents_to_envelopes(messages, store, "a", "b")
        key = envs[0].payload["key"]
        tampered = np.array(store._tensors[key])
        tampered[0, 0] += 1.0
        tampered.setflags(write=False)
        store._tensors[key] = tampered
        envelopes_to_latents(envs, store)  # unverified reads do not notice
        with pytest.raises(TensorDigestError):
            envelopes_to_latents(envs, store, verify=True)

    def test_validation(self, messages):
        store = InMemoryTensorStore()
        with pytest.raises(ValueError, match="shape"):
            latents_to_envelopes(messages[0], store, "a", "b")
        with pytest.raises(TypeError):
            latents_to_envelopes(messages.astype(np.int64), store, "a", "b")
        with pytest.raises(ValueError, match="finite"):
            latents_to_envelopes(messages * np.nan, store, "a", "b")
        with pytest.raises(TypeError):
            latents_to_envelopes(messages, store, "a", ["b"])
        with pytest.raises(ValueError, match="task_ids"):
            latents_to_envelopes(messages, store, "a", "b", task_ids=["only one"])
        with pytest.raises(ValueError):
            latents_to_envelopes(messages, store, "a", "b", wire_dtype="fp8")

    def test_reader_rejects_other_envelopes(self, messages):
        store = InMemoryTensorStore()
        envs = latents_to_envelopes(messages, store, "a", "b")
        with pytest.raises(EnvelopeError, match="addressed"):
            envelopes_to_latents(envs, store, recipient="someone else")
        text = Envelope(sender="a", recipients="b", payload_type="json", payload={})
        with pytest.raises(EnvelopeError, match="latent_ref"):
            envelopes_to_latents([text], store)
        other = latents_to_envelopes(messages[:, :2], store, "a", "b")
        with pytest.raises(EnvelopeError, match="share k and D"):
            envelopes_to_latents([envs[0], other[0]], store)
        with pytest.raises(ValueError):
            envelopes_to_latents([], store)
