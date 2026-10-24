"""Ollama backend: text-only generation with JSON-schema structured output.

Meant for the translator (natural language to a GoalSpec), where a local model behind
Ollama is enough and the result must be machine-readable. It uses only the standard
library, speaks to ``/api/chat`` and keeps the model resident for ``keep_alive``.

R4 measured a 14B coder model at 0.6 s per call with 100% schema-valid output; at
9 GB it cannot stay resident beside LoRA training on a 12 GB card, so ``unload()``
(``keep_alive=0``) frees it.
"""

import json
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Optional

from .base import CAP_GENERATE, CAP_STRUCTURED, GenerationResult

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5-coder:14b"

Transport = Callable[[str, str, Optional[dict[str, Any]], float], dict[str, Any]]
"""``(method, path, body, timeout) -> parsed JSON response``"""


class OllamaError(RuntimeError):
    """Raised when the Ollama server cannot be reached or rejects a request."""


def http_transport(host: str) -> Transport:
    """A transport that talks to a real Ollama server over HTTP."""

    def call(
        method: str, path: str, body: Optional[dict[str, Any]], timeout: float
    ) -> dict[str, Any]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            host.rstrip("/") + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise OllamaError(f"HTTP {exc.code} from {path}: {detail}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise OllamaError(f"cannot reach Ollama at {host}: {exc}") from exc

    return call


class OllamaBackend:
    """``ModelBackend`` over a local Ollama server (text only).

    Ollama has no per-call adapters, so ``adapter_id`` must stay ``None``: pick another
    model tag to pick another specialist.

    Args:
        model: Ollama model tag; this is the backend's ``base_id``
        host: Server URL
        keep_alive: How long the server keeps the model loaded after a call (``"10m"``,
            ``0`` to unload at once, ``-1`` to keep it forever)
        timeout: Per-request timeout in seconds (the first call loads the model)
        options: Default sampling options (``temperature`` 0 and ``num_ctx`` 2048 unless
            overridden)
        think: Whether reasoning models may think; off by default, since the translator
            wants the answer only
        transport: Replacement for the HTTP transport (tests)
    """

    capabilities = frozenset({CAP_GENERATE, CAP_STRUCTURED})

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        host: str = DEFAULT_HOST,
        keep_alive: Any = "10m",
        timeout: float = 600.0,
        options: Optional[Mapping[str, Any]] = None,
        think: bool = False,
        transport: Optional[Transport] = None,
    ):
        self.base_id = model
        self.adapter_id: Optional[str] = None
        self.host = host
        self.keep_alive = keep_alive
        self.timeout = timeout
        self.options: dict[str, Any] = {"temperature": 0, "num_ctx": 2048}
        self.options.update(options or {})
        self.think = think
        self._transport = transport or http_transport(host)

    # ------------------------------------------------------------------ requests

    def request_body(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        examples: Sequence[tuple[str, str]] = (),
        schema: Optional[Mapping[str, Any]] = None,
        max_new_tokens: Optional[int] = None,
        keep_alive: Any = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        """The ``/api/chat`` body for one prompt (exposed for inspection and tests)."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        for user, assistant in examples:
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})
        messages.append({"role": "user", "content": prompt})

        merged = dict(self.options)
        merged.update(options or {})
        if max_new_tokens is not None:
            merged["num_predict"] = int(max_new_tokens)
        body: dict[str, Any] = {
            "model": self.base_id,
            "messages": messages,
            "stream": False,
            "options": merged,
            "keep_alive": self.keep_alive if keep_alive is None else keep_alive,
            "think": self.think,
        }
        if schema is not None:
            body["format"] = dict(schema)
        return body

    def generate(
        self,
        prompts: Sequence[str],
        adapter_id: Optional[str] = None,
        *,
        system: Optional[str] = None,
        examples: Sequence[tuple[str, str]] = (),
        schema: Optional[Mapping[str, Any]] = None,
        max_new_tokens: Optional[int] = None,
        keep_alive: Any = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> list[GenerationResult]:
        """Complete each prompt with one chat call.

        Args:
            prompts: User messages, one call each
            adapter_id: Must be ``None``
            system: System prompt shared by all calls
            examples: Few-shot ``(user, assistant)`` pairs shared by all calls
            schema: JSON Schema the reply must follow (Ollama structured output)
            max_new_tokens: Sets ``num_predict``
            keep_alive: Overrides the backend's default for these calls
            options: Sampling options merged over the backend's defaults

        Returns:
            One result per prompt. With ``schema`` the text is a JSON document; use
            :meth:`generate_json` to parse and check it.

        Raises:
            ValueError: ``adapter_id`` is not ``None``
            OllamaError: The server is unreachable or rejects the request
        """
        if adapter_id is not None:
            raise ValueError(
                "OllamaBackend has no adapters; choose another model tag instead"
            )
        results = []
        for prompt in prompts:
            body = self.request_body(
                prompt,
                system=system,
                examples=examples,
                schema=schema,
                max_new_tokens=max_new_tokens,
                keep_alive=keep_alive,
                options=options,
            )
            start = time.perf_counter()
            reply = self._transport("POST", "/api/chat", body, self.timeout)
            elapsed = time.perf_counter() - start
            message = reply.get("message") or {}
            done = reply.get("done_reason", "stop")
            results.append(
                GenerationResult(
                    text=message.get("content", ""),
                    n_prompt_tokens=int(reply.get("prompt_eval_count") or 0),
                    n_new_tokens=int(reply.get("eval_count") or 0),
                    finish_reason="length" if done == "length" else "stop",
                    latency_s=elapsed,
                    meta={
                        key: reply.get(key)
                        for key in (
                            "total_duration",
                            "load_duration",
                            "prompt_eval_duration",
                            "eval_duration",
                        )
                    },
                )
            )
        return results

    def generate_json(
        self, prompt: str, schema: Mapping[str, Any], **options: Any
    ) -> tuple[dict[str, Any], GenerationResult]:
        """One structured call: returns the parsed object and the raw result.

        The object is checked against ``schema`` when ``jsonschema`` is installed (Ollama
        constrains decoding but may not enforce every keyword, e.g. ``minItems``).

        Raises:
            ValueError: The reply is not JSON, or does not satisfy ``schema``
        """
        (result,) = self.generate([prompt], schema=schema, **options)
        try:
            parsed = json.loads(result.text)
        except ValueError as exc:
            raise ValueError(f"reply is not JSON: {result.text[:200]!r}") from exc
        try:
            import jsonschema
        except ImportError:  # pragma: no cover - jsonschema is a core dependency
            return parsed, result
        try:
            jsonschema.validate(parsed, dict(schema))
        except jsonschema.ValidationError as exc:
            raise ValueError(f"reply violates the schema: {exc.message}") from exc
        return parsed, result

    # ------------------------------------------------------------------ server state

    def is_up(self) -> bool:
        """Whether the server answers."""
        try:
            self._transport("GET", "/api/version", None, 5.0)
        except OllamaError:
            return False
        return True

    def loaded_models(self) -> list[str]:
        """Names of the models currently resident in the server."""
        reply = self._transport("GET", "/api/ps", None, 10.0)
        return [m.get("name") or m.get("model", "") for m in reply.get("models", [])]

    def unload(self) -> None:
        """Free the model now (``keep_alive=0``)."""
        self._transport(
            "POST",
            "/api/generate",
            {"model": self.base_id, "keep_alive": 0},
            60.0,
        )
