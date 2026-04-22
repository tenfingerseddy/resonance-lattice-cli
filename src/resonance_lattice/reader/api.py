# SPDX-License-Identifier: BUSL-1.1
"""API-backed reader (C3 of the v1.0.0 semantic-layer plan).

HTTP reader for users with API keys. Supports Anthropic's Messages API
and any OpenAI-compatible Chat Completions endpoint (OpenAI itself,
Together, Groq, local OpenAI-compatible servers, etc.).

Design contract:

1. **Zero new runtime deps.** Uses stdlib `urllib.request`. The
   Anthropic/OpenAI SDKs add retry, streaming, and async niceties
   but drag in pydantic / httpx for a synchronous one-shot call we
   don't need. Keeping `pyproject.toml` quiet matters for the v1.0.0
   install-and-go story.

2. **Env-var convention matches the repo.** Anthropic key is read
   from `CLAUDE_API` first (repo convention), falling back to
   `ANTHROPIC_API_KEY` for portability. OpenAI reads `OPENAI_API_KEY`.
   Callers can pass `api_key` explicitly to override both.

3. **Shared prompting with LocalReader.** `build_prompt` and
   `extract_citations` live in `reader.local` and are imported here
   so the two readers produce identical context packs and apply the
   same grounding invariant (out-of-range `[N]` markers dropped).
   One citation format, one system prompt, two transports.

4. **Fail-fast, informative errors.** HTTP failures carry the status
   code and a truncated body in the exception message so `rlat ask`
   can surface actionable remediation (`401 → check CLAUDE_API`,
   `429 → rate-limited, retry later`).
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any

from resonance_lattice.reader.base import Answer, Evidence, Reader
from resonance_lattice.reader.local import (
    DEFAULT_SYSTEM_PROMPT,
    build_prompt,
    extract_citations,
)

logger = logging.getLogger(__name__)


ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"
OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"

# Anthropic API version header. Pinning avoids silent behaviour shifts
# when they roll new versions. Update deliberately.
ANTHROPIC_API_VERSION = "2023-06-01"

DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.3
DEFAULT_TIMEOUT = 60.0


class APIReaderError(RuntimeError):
    """Raised on any HTTP / decoding failure from the upstream API.

    The message includes the provider, HTTP status (if any), and the
    first 500 chars of the response body. Callers catching this should
    treat it as terminal for the request — no retry at this layer.
    """


class APIReader(Reader):
    """Reader backed by a remote chat-completions API.

    Supports two providers:

      - "anthropic": Claude Messages API. Reads CLAUDE_API or
        ANTHROPIC_API_KEY. Recommended models: `claude-opus-4-7`,
        `claude-sonnet-4-6`.

      - "openai": OpenAI Chat Completions API and any OpenAI-
        compatible endpoint. Reads OPENAI_API_KEY. `base_url` lets
        callers point at alternative hosts (Together, Groq, local
        OpenAI-compatible servers) without code changes.

    Args:
        provider: "anthropic" or "openai".
        model: Provider-specific model id.
        api_key: Explicit key. Overrides env var lookup.
        base_url: Override the provider's default endpoint (OpenAI-
            compatible usage). Ignored for Anthropic — the Messages
            API endpoint shape is stable.
        max_tokens: Generation cap. Provider-wide default when unset.
        temperature: Sampling temperature.
        system_prompt: Override the default grounded-assistant prompt.
        timeout: HTTP timeout in seconds.

    Raises:
        ValueError: unknown provider or no API key available.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        system_prompt: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        if provider not in ("anthropic", "openai"):
            raise ValueError(
                f"Unknown provider: {provider!r}. "
                f"Expected one of: anthropic, openai."
            )
        if not model:
            raise ValueError("model must be a non-empty string")

        self._provider = provider
        self._model = model
        self._api_key = api_key or _resolve_api_key(provider)
        self._base_url = base_url
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._timeout = timeout
        self.name = f"{provider}-{model}"

    def answer(
        self, query: str, evidence: Sequence[Evidence],
    ) -> Answer:
        messages = build_prompt(query, evidence, self._system_prompt)
        t0 = time.perf_counter()
        generated_text = self._call(messages)
        latency_ms = (time.perf_counter() - t0) * 1000
        citations = extract_citations(generated_text, evidence)
        return Answer(
            query=query,
            text=generated_text.strip(),
            citations=citations,
            model=self.name,
            latency_ms=round(latency_ms, 2),
            evidence_used=len(evidence),
        )

    # ── HTTP dispatch ────────────────────────────────────────────────

    def _call(self, messages: list[dict[str, str]]) -> str:
        """Route to the right provider-shaped request."""
        if self._provider == "anthropic":
            return self._call_anthropic(messages)
        return self._call_openai(messages)

    def _call_anthropic(self, messages: list[dict[str, str]]) -> str:
        """Anthropic Messages API.

        Shape differs from OpenAI: system prompt is a top-level field,
        not a role. Extract it from our chat-shaped messages.
        """
        system = ""
        user_messages: list[dict[str, str]] = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                user_messages.append(m)

        body = {
            "model": self._model,
            "messages": user_messages,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        if system:
            body["system"] = system

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
            "content-type": "application/json",
        }

        data = _http_post(
            self._base_url or ANTHROPIC_ENDPOINT,
            headers=headers,
            body=body,
            timeout=self._timeout,
            provider="anthropic",
        )

        content = data.get("content") or []
        # Anthropic returns a list of content blocks; concatenate text ones.
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)

    def _call_openai(self, messages: list[dict[str, str]]) -> str:
        """OpenAI Chat Completions API (and OpenAI-compatible endpoints)."""
        body = {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        endpoint = (self._base_url or OPENAI_ENDPOINT).rstrip("/")
        # Accept either a bare host ("https://api.together.xyz") or a
        # full endpoint path — append the default suffix when the path
        # doesn't already target a chat completions route.
        if not endpoint.endswith("/chat/completions"):
            endpoint = endpoint + "/v1/chat/completions" if "/v1" not in endpoint else endpoint + "/chat/completions"

        headers = {
            "authorization": f"Bearer {self._api_key}",
            "content-type": "application/json",
        }

        data = _http_post(
            endpoint,
            headers=headers,
            body=body,
            timeout=self._timeout,
            provider="openai",
        )
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return str(message.get("content") or "")


def _resolve_api_key(provider: str) -> str:
    """Look up the API key from env vars using the repo's conventions.

    Anthropic: CLAUDE_API (repo convention) > ANTHROPIC_API_KEY.
    OpenAI:    OPENAI_API_KEY.

    Raises ValueError when no key is found so construction fails fast
    instead of letting the first request 401 halfway through a user flow.
    """
    if provider == "anthropic":
        key = os.environ.get("CLAUDE_API") or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "Anthropic API key not found. Set CLAUDE_API (repo "
                "convention) or ANTHROPIC_API_KEY, or pass api_key=..."
            )
        return key
    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY, "
                "or pass api_key=..."
            )
        return key
    raise ValueError(f"Unknown provider: {provider!r}")


def _http_post(
    url: str,
    *,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout: float,
    provider: str,
) -> dict[str, Any]:
    """POST JSON, return parsed JSON. Raises APIReaderError on any failure.

    Kept as a free function so tests can monkeypatch it without having
    to touch a network stack or mock urllib directly.
    """
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        # Surface both the status and the body preview — 401/429/400
        # respectively mean different things to the caller.
        body_preview = ""
        try:
            body_preview = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        raise APIReaderError(
            f"{provider} API HTTP {e.code}: {e.reason}. Body: {body_preview}"
        ) from e
    except urllib.error.URLError as e:
        raise APIReaderError(
            f"{provider} API request failed: {e.reason}"
        ) from e
    except Exception as e:
        raise APIReaderError(
            f"{provider} API request raised {type(e).__name__}: {e}"
        ) from e

    try:
        return json.loads(raw.decode("utf-8"))
    except Exception as e:
        preview = raw[:500].decode("utf-8", errors="replace")
        raise APIReaderError(
            f"{provider} API returned non-JSON: {preview}"
        ) from e
