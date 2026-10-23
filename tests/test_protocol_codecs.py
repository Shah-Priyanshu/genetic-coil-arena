"""Test the array <-> envelope codecs and the tensor store."""

import numpy as np
import pytest

from genetic_coil_arena.protocol import (
    CODEC_VERSION,
    Envelope,
    InMemoryTensorStore,
    TensorDigestError,
    TensorStore,
    codes_to_envelopes,
    envelopes_to_codes,
    envelopes_to_vectors,
    tensor_digest,
    vectors_to_envelopes,
)

RECEIVERS = ["r0", "r1", "r2"]


@pytest.fixture
def codes():
    rng = np.random.default_rng(0)
    return rng.integers(0, 7, size=(4, 3, 5), dtype=np.int64)


@pytest.fixture
def vectors():
    rng = np.random.default_rng(1)
    return rng.standard_normal((2, 3, 6)).astype(np.float32)


class TestCodes:
    def test_round_trip(self, codes):
        batch = codes_to_envelopes(codes, 7, "speaker", RECEIVERS)
        back = envelopes_to_codes(batch, RECEIVERS)
        assert back.dtype == np.int64 and back.shape == (4, 3, 5)
        np.testing.assert_array_equal(back, codes)

    def test_layout_and_encoding(self, codes):
        batch = codes_to_envelopes(codes, 7, "speaker", RECEIVERS, clock=3)
        assert len(batch) == 4 and all(len(row) == 3 for row in batch)
        env = batch[2][1]
        assert env.recipients == ("r1",) and env.sender == "speaker"
        assert env.payload_type == "codes" and env.clock == 3
        assert env.payload == tuple(codes[2, 1].tolist())
        assert env.payload_encoding == {
            "codec": "codes",
            "codec_version": CODEC_VERSION,
            "V": 7,
            "L": 5,
        }

    def test_episode_ids_shared_within_episode(self, codes):
        batch = codes_to_envelopes(codes, 7, "s", RECEIVERS, task_ids=list("abcd"))
        for b, row in enumerate(batch):
            assert {e.task_id for e in row} == {"abcd"[b]}
            assert len({e.trace_id for e in row}) == 1
        assert len({row[0].trace_id for row in batch}) == 4

    def test_explicit_trace_and_parent_ids(self, codes):
        batch = codes_to_envelopes(
            codes,
            7,
            "s",
            RECEIVERS,
            trace_ids=["t0", None, "t2", "t3"],
            parent_ids=["p"] * 4,
        )
        assert batch[0][0].trace_id == "t0" and batch[0][0].parent_id == "p"
        assert (
            batch[1][0].trace_id == batch[1][2].trace_id
        )  # None -> one new id per episode

    def test_survives_json(self, codes):
        batch = codes_to_envelopes(codes, 7, "s", RECEIVERS)
        rebuilt = [[Envelope.from_json(e.to_json()) for e in row] for row in batch]
        np.testing.assert_array_equal(envelopes_to_codes(rebuilt), codes)

    @pytest.mark.parametrize(
        "bad", [np.zeros((2, 3)), np.zeros((1, 3, 2), dtype=np.float32)]
    )
    def test_rejects_bad_arrays(self, bad):
        with pytest.raises((ValueError, TypeError)):
            codes_to_envelopes(bad, 4, "s", RECEIVERS)

    def test_rejects_out_of_vocab(self, codes):
        with pytest.raises(ValueError, match=r"\[0, 3\)"):
            codes_to_envelopes(codes, 3, "s", RECEIVERS)
        with pytest.raises(ValueError):
            codes_to_envelopes(-np.ones((1, 3, 2), dtype=np.int64), 3, "s", RECEIVERS)

    def test_rejects_bad_recipients_or_ids(self, codes):
        with pytest.raises(ValueError, match="recipients"):
            codes_to_envelopes(codes, 7, "s", RECEIVERS[:2])
        with pytest.raises(ValueError, match="unique"):
            codes_to_envelopes(codes, 7, "s", ["a", "a", "b"])
        with pytest.raises(ValueError, match="task_ids"):
            codes_to_envelopes(codes, 7, "s", RECEIVERS, task_ids=["x"])

    def test_decode_checks(self, codes):
        batch = codes_to_envelopes(codes, 7, "s", RECEIVERS)
        with pytest.raises(ValueError, match="expected"):
            envelopes_to_codes(batch, ["r0", "r2", "r1"])
        with pytest.raises(ValueError):
            envelopes_to_codes([])
        with pytest.raises(ValueError, match="same number"):
            envelopes_to_codes([batch[0], batch[1][:2]])
        other = codes_to_envelopes(codes[:1, :, :3], 7, "s", RECEIVERS)
        with pytest.raises(ValueError, match="share"):
            envelopes_to_codes([batch[0], other[0]])

    def test_decode_rejects_wrong_codec_version(self, codes):
        env = codes_to_envelopes(codes[:1, :1], 7, "s", ["r"])[0][0]
        data = env.to_dict()
        data["payload_encoding"] = {**data["payload_encoding"], "codec_version": 99}
        with pytest.raises(ValueError, match="version"):
            envelopes_to_codes([[Envelope.from_dict(data)]])

    def test_decode_rejects_vector_envelopes(self, vectors):
        batch = vectors_to_envelopes(vectors, InMemoryTensorStore(), "s", RECEIVERS)
        with pytest.raises(ValueError, match="payload_type"):
            envelopes_to_codes(batch)


class TestVectors:
    def test_round_trip_by_reference(self, vectors):
        store = InMemoryTensorStore()
        batch = vectors_to_envelopes(vectors, store, "s", RECEIVERS)
        assert len(store) == 6
        env = batch[1][2]
        assert env.payload_type == "latent_ref"
        assert set(env.payload) == {"store", "key", "shape", "dtype", "sha256"}
        assert (
            env.payload_encoding["D"] == 6
            and env.payload_encoding["dtype"] == "float32"
        )
        assert "0." not in env.to_json().split('"payload"')[1][:200]  # not inlined
        back = envelopes_to_vectors(batch, store, RECEIVERS, verify=True)
        assert back.dtype == np.float32
        np.testing.assert_array_equal(back, vectors)

    def test_float64_is_stored_as_float32(self):
        store = InMemoryTensorStore()
        batch = vectors_to_envelopes(np.ones((1, 1, 2)), store, "s", ["r"])
        assert store.get(batch[0][0].payload).dtype == np.float32

    @pytest.mark.parametrize(
        "bad",
        [
            np.zeros((2, 3)),
            np.zeros((1, 1, 2), dtype=np.int64),
            np.full((1, 1, 2), np.nan),
        ],
    )
    def test_rejects_bad_arrays(self, bad):
        with pytest.raises((ValueError, TypeError)):
            vectors_to_envelopes(bad, InMemoryTensorStore(), "s", ["r"])

    def test_missing_tensor(self, vectors):
        store = InMemoryTensorStore()
        batch = vectors_to_envelopes(vectors, store, "s", RECEIVERS)
        store.delete(batch[0][0].payload)
        with pytest.raises(KeyError):
            envelopes_to_vectors(batch, store)

    def test_wrong_store(self, vectors):
        batch = vectors_to_envelopes(vectors, InMemoryTensorStore("a"), "s", RECEIVERS)
        with pytest.raises(KeyError, match="store"):
            envelopes_to_vectors(batch, InMemoryTensorStore("b"))


class TestTensorStore:
    def test_protocol_and_read_only_copy(self):
        store = InMemoryTensorStore()
        assert isinstance(store, TensorStore)
        source = np.arange(3, dtype=np.float32)
        ref = store.put(source)
        source[0] = 99
        stored = store.get(ref)
        assert stored[0] == 0
        with pytest.raises(ValueError):
            stored[0] = 1
        assert ref in store and len(store) == 1
        store.delete(ref)
        store.delete(ref)
        assert ref not in store

    def test_verify_detects_swapped_tensor(self):
        store = InMemoryTensorStore()
        ref = store.put(np.ones(3, dtype=np.float32))
        store._tensors[ref["key"]] = np.zeros(3, dtype=np.float32)
        store.get(ref)  # no check by default
        with pytest.raises(TensorDigestError):
            store.get(ref, verify=True)

    def test_digest_covers_dtype_and_shape(self):
        a = np.zeros(4, dtype=np.float32)
        assert tensor_digest(a) != tensor_digest(a.reshape(2, 2))
        assert tensor_digest(a) != tensor_digest(a.view(np.int32))
        assert tensor_digest(a) == tensor_digest(a.copy())

    def test_rejects_object_arrays_and_empty_name(self):
        with pytest.raises(TypeError):
            InMemoryTensorStore().put(np.array([object()]))
        with pytest.raises(ValueError):
            InMemoryTensorStore("")
