"""Test the backend interface, lazy imports and the Ollama backend (no model needed)."""

import json
import subprocess
import sys

import pytest

from genetic_coil_arena.backends import (
    CAP_GENERATE,
    CAP_STRUCTURED,
    GenerationResult,
    HFPeftBackend,
    LatentBackend,
    ModelBackend,
    OllamaBackend,
    OllamaError,
    kv_bytes_per_token,
)

SCHEMA = {
    "type": "object",
    "properties": {"tempo": {"enum": ["fast", "slow"]}},
    "required": ["tempo"],
    "additionalProperties": False,
}


class FakeServer:
    """Records requests and answers like Ollama's /api/chat."""

    def __init__(self, content='{"tempo":"fast"}', done_reason="stop"):
        self.calls = []
        self.content = content
        self.done_reason = done_reason

    def __call__(self, method, path, body, timeout):
        self.calls.append((method, path, body, timeout))
        if path == "/api/chat":
            return {
                "message": {"role": "assistant", "content": self.content},
                "done_reason": self.done_reason,
                "prompt_eval_count": 12,
                "eval_count": 5,
                "total_duration": 10,
                "load_duration": 1,
            }
        if path == "/api/ps":
            return {"models": [{"name": "m:1"}, {"model": "other"}]}
        return {}


def test_import_is_lazy_about_the_ml_stack():
    code = (
        "import sys, genetic_coil_arena.backends as b, genetic_coil_arena.channels as c;"
        "c.wire_nbytes(1, 2);"
        "bad = [m for m in ('torch', 'transformers', 'peft') if m in sys.modules];"
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_protocols_and_capabilities():
    hf, ollama = HFPeftBackend(), OllamaBackend(transport=FakeServer())
    assert isinstance(hf, ModelBackend) and isinstance(hf, LatentBackend)
    assert isinstance(ollama, ModelBackend) and not isinstance(ollama, LatentBackend)
    assert {"hidden_states", "kv_cache", "soft_prompt", "train"} <= hf.capabilities
    assert ollama.capabilities == {CAP_GENERATE, CAP_STRUCTURED}
    assert hf.base_id == "Qwen/Qwen3-1.7B" and hf.adapter_id is None


def test_a_plain_class_can_implement_model_backend():
    class Echo:
        base_id = "echo"
        capabilities = frozenset({CAP_GENERATE})

        def generate(self, prompts, adapter_id=None, **options):
            return [GenerationResult(text=p) for p in prompts]

    assert isinstance(Echo(), ModelBackend)
    assert not isinstance(Echo(), LatentBackend)


def test_kv_bytes_per_token_matches_the_r4_measurement():
    assert kv_bytes_per_token(28, 8, 128, 2) == 112 * 1024


def test_ollama_request_body_and_result():
    server = FakeServer()
    backend = OllamaBackend("m:1", transport=server, keep_alive="5m")
    (result,) = backend.generate(
        ["go fast"],
        system="sys",
        examples=[("slow please", '{"tempo":"slow"}')],
        schema=SCHEMA,
        max_new_tokens=50,
    )
    method, path, body, _ = server.calls[0]
    assert (method, path) == ("POST", "/api/chat")
    assert body["model"] == "m:1" and body["stream"] is False
    assert body["format"] == SCHEMA and body["keep_alive"] == "5m"
    assert body["think"] is False
    assert body["options"] == {"temperature": 0, "num_ctx": 2048, "num_predict": 50}
    assert [m["role"] for m in body["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert body["messages"][-1]["content"] == "go fast"
    assert result.text == '{"tempo":"fast"}'
    assert (result.n_prompt_tokens, result.n_new_tokens) == (12, 5)
    assert result.finish_reason == "stop" and result.meta["load_duration"] == 1


def test_ollama_without_schema_omits_format_and_reports_truncation():
    server = FakeServer(done_reason="length")
    (result,) = OllamaBackend(transport=server).generate(["hi", "again"])[:1]
    assert "format" not in server.calls[0][2]
    assert len(server.calls) == 2  # one call per prompt
    assert result.finish_reason == "length"


def test_ollama_generate_json_parses_and_validates():
    parsed, result = OllamaBackend(transport=FakeServer()).generate_json("x", SCHEMA)
    assert parsed == {"tempo": "fast"} and result.n_new_tokens == 5
    with pytest.raises(ValueError, match="violates the schema"):
        OllamaBackend(transport=FakeServer('{"tempo":"warp"}')).generate_json(
            "x", SCHEMA
        )
    with pytest.raises(ValueError, match="not JSON"):
        OllamaBackend(transport=FakeServer("fast!")).generate_json("x", SCHEMA)


def test_ollama_has_no_adapters():
    with pytest.raises(ValueError, match="no adapters"):
        OllamaBackend(transport=FakeServer()).generate(["x"], adapter_id="lora")


def test_ollama_server_state_calls():
    server = FakeServer()
    backend = OllamaBackend("m:1", transport=server)
    assert backend.is_up() and backend.loaded_models() == ["m:1", "other"]
    backend.unload()
    method, path, body, _ = server.calls[-1]
    assert (method, path) == ("POST", "/api/generate")
    assert body == {"model": "m:1", "keep_alive": 0}

    def down(method, path, body, timeout):
        raise OllamaError("refused")

    assert not OllamaBackend(transport=down).is_up()
    json.dumps(backend.request_body("x"))  # the body is plain JSON
