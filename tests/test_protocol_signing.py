"""Test HMAC signature manifests: tampering anywhere must be detected."""

import numpy as np
import pytest

from genetic_coil_arena.protocol import (
    Envelope,
    InMemoryTensorStore,
    SignatureError,
    check_signature,
    payload_digest,
    require_valid,
    sign,
    verify,
)

KEY = b"secret"


def make(**overrides):
    fields = {
        "sender": "a",
        "recipients": ("b",),
        "payload_type": "json",
        "payload": {"x": 1},
    }
    fields.update(overrides)
    return Envelope(**fields)


def tamper(env, **changes):
    """Rebuild a signed envelope with some serialized fields changed, keeping its manifest."""
    return Envelope.from_dict({**env.to_dict(), **changes})


class TestSignVerify:
    def test_valid_signature(self):
        signed = sign(make(), KEY)
        assert signed.is_signed
        assert signed.manifest["alg"] == "HMAC-SHA256"
        assert signed.manifest["key_id"] == "default"
        assert signed.manifest["payload_sha256"] == payload_digest(signed)
        assert verify(signed, KEY)
        assert require_valid(signed, KEY) is signed

    def test_signing_does_not_mutate(self):
        env = make()
        sign(env, KEY)
        assert env.manifest is None

    def test_survives_json_round_trip(self):
        signed = sign(make(payload_type="codes", payload=[1, 2]), KEY)
        assert verify(Envelope.from_json(signed.to_json()), KEY)

    def test_unsigned(self):
        assert check_signature(make(), KEY) == "unsigned"
        with pytest.raises(SignatureError, match="unsigned"):
            require_valid(make(), KEY)

    def test_wrong_key(self):
        assert check_signature(sign(make(), KEY), b"other") == "bad signature"

    def test_empty_key_refused(self):
        with pytest.raises(ValueError):
            sign(make(), b"")

    def test_keyring(self):
        signed = sign(make(), KEY, key_id="alice")
        assert verify(signed, {"alice": KEY, "bob": b"x"})
        assert check_signature(signed, {"bob": KEY}) == "unknown key id 'alice'"

    def test_resigning_replaces_manifest(self):
        signed = sign(sign(make(), b"old"), KEY)
        assert verify(signed, KEY) and not verify(signed, b"old")


class TestTampering:
    @pytest.mark.parametrize(
        "changes",
        [
            {"sender": "mallory"},
            {"recipients": ["c"]},
            {"task_id": "other"},
            {"trace_id": "other"},
            {"parent_id": "other"},
            {"clock": 99},
            {"msg_id": "replayed"},
            {"payload_encoding": {"codec": "other"}},
        ],
    )
    def test_header_tampering_breaks_mac(self, changes):
        signed = sign(make(), KEY)
        assert check_signature(tamper(signed, **changes), KEY) == "bad signature"

    def test_payload_tampering_detected(self):
        signed = sign(make(payload={"result": 42}), KEY)
        assert check_signature(tamper(signed, payload={"result": -1}), KEY) == (
            "payload digest mismatch"
        )

    def test_payload_and_digest_tampering_detected(self):
        signed = sign(make(payload={"result": 42}), KEY)
        forged = tamper(signed, payload={"result": -1})
        manifest = {**forged.manifest, "payload_sha256": payload_digest(forged)}
        assert check_signature(forged.with_manifest(manifest), KEY) == "bad signature"

    @pytest.mark.parametrize(
        "manifest",
        [
            {"alg": "none"},
            {"alg": "HMAC-SHA256"},
            {"alg": "HMAC-SHA256", "key_id": 5},
            {
                "alg": "HMAC-SHA256",
                "key_id": "default",
                "payload_sha256": "é",
                "mac": "é",
            },
            {"alg": "HMAC-SHA256", "key_id": "default", "payload_sha256": None},
        ],
    )
    def test_malformed_manifest_is_invalid_not_an_exception(self, manifest):
        assert not verify(make().with_manifest(manifest), KEY)


class TestLatentPayloads:
    def test_swapped_tensor_detected_with_store(self):
        store = InMemoryTensorStore()
        ref = store.put(np.ones(4, dtype=np.float32))
        signed = sign(make(payload_type="latent_ref", payload=ref), KEY)
        assert verify(signed, KEY, store)
        store._tensors[ref["key"]] = np.zeros(4, dtype=np.float32)
        assert verify(signed, KEY)  # the reference itself is intact
        assert check_signature(signed, KEY, store) == "tensor digest mismatch"

    def test_missing_tensor(self):
        store = InMemoryTensorStore()
        ref = store.put(np.ones(2, dtype=np.float32))
        signed = sign(make(payload_type="latent_ref", payload=ref), KEY)
        store.delete(ref)
        assert check_signature(signed, KEY, store) == "referenced tensor not found"

    def test_reference_digest_tampering_detected(self):
        store = InMemoryTensorStore()
        ref = store.put(np.ones(2, dtype=np.float32))
        signed = sign(make(payload_type="latent_ref", payload=ref), KEY)
        forged = tamper(signed, payload={**ref, "sha256": "0" * 64})
        assert check_signature(forged, KEY, store) == "payload digest mismatch"
