"""DeepInfra HTTP client for the OpenAI-compatible chat completions endpoint.

Model + provider IDs are sourced from ``eval/config/model_candidates.yaml``
via the resolver pair (``_resolve_provider_model_id``,
``_resolve_candidate_family``). No literals like ``Qwen/Qwen3-...`` or
``deepinfra`` are allowed in this module — yaml drift must be visible at
the call site, not hidden behind a hardcoded fallback.

Auth comes exclusively from ``os.environ["DEEPINFRA_API_KEY"]``.
``_read_api_key`` raises ``ProviderError`` if the env var is missing or
empty; the value is never echoed in any error or log.

Retry policy mirrors design §6: exponential backoff on 429 and 5xx,
permanent failure on other 4xx, malformed responses, and exhausted
retries.

OpenRouter failover is deferred (design §6 "OpenRouter engages only
when ``LENSGRAPH_PROVIDER_FAILOVER=1``"). Slice 3 does not wire it.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "eval" / "config" / "model_candidates.yaml"
)
_DEEPINFRA_CHAT_URL = "https://api.deepinfra.com/v1/openai/chat/completions"


class ProviderError(RuntimeError):
    """Permanent provider failure.

    Triggers:
      * non-429 4xx response,
      * malformed 200 response (missing ``choices[0].message.content``),
      * max retries exhausted on 429/5xx/transport errors,
      * missing or empty ``DEEPINFRA_API_KEY``.
    """


@dataclass(frozen=True)
class ProviderResponse:
    raw_text: str
    latency_ms: int
    provider_model_id: str


def _load_yaml() -> dict:
    import yaml

    return yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))


def _lookup_option(candidate_id: str, *, component: str) -> dict:
    raw = _load_yaml()
    try:
        options = raw["candidates"][component]["options"]
    except KeyError as e:
        raise KeyError(
            f"candidates.{component}.options missing from model_candidates.yaml"
        ) from e
    for opt in options:
        if opt.get("id") == candidate_id:
            return opt
    raise KeyError(
        f"candidate_id={candidate_id!r} not found in candidates.{component}.options"
    )


def _resolve_provider_model_id(candidate_id: str, *, component: str) -> str:
    """Return the wire-level ``provider_model_id`` for a yaml candidate.

    ``component`` is the yaml group key
    (``'generator'`` | ``'judge'`` | ``'cheap_extraction'`` | ...)."""
    return _lookup_option(candidate_id, component=component)["provider_model_id"]


def _resolve_candidate_family(candidate_id: str, *, component: str) -> str:
    """Return the candidate's ``family`` field. Used by generate/api.py
    (Teammate B) for the cross-family judge invariant."""
    return _lookup_option(candidate_id, component=component)["family"]


def _resolve_candidate_provider(candidate_id: str, *, component: str) -> str:
    """Return the candidate's ``provider`` field (e.g. ``deepinfra`` /
    ``local`` / ``openrouter``). generate/api.py denormalises this into
    GeneratorCandidate / JudgeCandidate so AgentState never carries a
    yaml reference. Like the other two resolvers, the yaml is the only
    source of truth — drift surfaces here, not at request time."""
    return _lookup_option(candidate_id, component=component)["provider"]


def _read_api_key() -> str:
    """Return ``DEEPINFRA_API_KEY`` from env or raise ProviderError.

    The key value is NEVER included in the error message — only the env
    var name. Missing and empty are both errors."""
    key = os.environ.get("DEEPINFRA_API_KEY", "")
    if not key:
        raise ProviderError(
            "DEEPINFRA_API_KEY is missing or empty in the environment"
        )
    return key


def _client_factory() -> httpx.Client:
    """Module-attribute hook for tests to substitute an httpx.Client with
    a ``MockTransport``. Production default constructs a real client; per-
    request timeout is enforced in ``chat_completion`` so we don't pin one
    here."""
    return httpx.Client()


def _is_retryable_status(status: int) -> bool:
    return status == 429 or 500 <= status < 600


def _backoff_sleep(attempt: int, base: float) -> None:
    """Exponential backoff with uniform jitter ∈ [0, 0.5).

    ``attempt`` is 0-indexed (the just-failed attempt number). Sleeping
    is delegated to ``time.sleep`` so tests can monkeypatch it to a
    no-op."""
    if base <= 0:
        # Still call sleep so tests asserting "sleep was no-oped" don't
        # need to distinguish "skipped" from "instant" — keep semantics
        # identical to base>0 except for the zero-magnitude wait.
        time.sleep(0)
        return
    delay = base * (2**attempt) + random.uniform(0, 0.5)
    time.sleep(delay)


def chat_completion(
    candidate_id: str,
    messages: list[dict],
    *,
    component: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout_sec: float = 30.0,
    max_retries: int = 3,
    backoff_base_sec: float = 1.0,
) -> ProviderResponse:
    """POST to DeepInfra's OpenAI-compatible chat completions endpoint.

    ``candidate_id`` is the yaml ``id`` (e.g. ``'qwen3-235b-a22b-instruct'``).
    ``component`` is the yaml group key the candidate is found under.

    Retry budget: ``max_retries`` after the initial attempt (total
    attempts ≤ ``max_retries + 1``). 429 / 5xx / transport errors are
    retried with exponential backoff; other 4xx and malformed-200 are
    permanent.

    ``latency_ms`` is the total wall-clock for the call including any
    retry sleeps — design §6's "p95 latency ≤ 2.0s" minimum is total,
    non-streaming.
    """
    api_key = _read_api_key()
    provider_model_id = _resolve_provider_model_id(candidate_id, component=component)
    payload = {
        "model": provider_model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_status: int | None = None
    last_error_detail: str | None = None
    start = time.perf_counter()
    client = _client_factory()
    try:
        for attempt in range(max_retries + 1):
            try:
                response = client.post(
                    _DEEPINFRA_CHAT_URL,
                    json=payload,
                    headers=headers,
                    timeout=timeout_sec,
                )
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_status = None
                last_error_detail = type(e).__name__
                if attempt >= max_retries:
                    break
                _backoff_sleep(attempt, backoff_base_sec)
                continue

            status = response.status_code
            if status == 200:
                try:
                    body = response.json()
                    content = body["choices"][0]["message"]["content"]
                except (KeyError, IndexError, ValueError, TypeError) as e:
                    raise ProviderError(
                        f"malformed 200 response: {type(e).__name__}"
                    ) from e
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                return ProviderResponse(
                    raw_text=content,
                    latency_ms=elapsed_ms,
                    provider_model_id=provider_model_id,
                )

            if _is_retryable_status(status):
                last_status = status
                last_error_detail = f"status={status}"
                if attempt >= max_retries:
                    break
                _backoff_sleep(attempt, backoff_base_sec)
                continue

            # Non-retryable 4xx or 3xx — fail immediately. Do NOT log
            # response body verbatim (it may echo back the auth header
            # in some provider implementations).
            raise ProviderError(
                f"DeepInfra returned non-retryable status {status} "
                f"(candidate_id={candidate_id})"
            )

        raise ProviderError(
            f"DeepInfra exhausted max_retries={max_retries} "
            f"(last={last_error_detail or 'unknown'}, last_status={last_status})"
        )
    finally:
        client.close()
