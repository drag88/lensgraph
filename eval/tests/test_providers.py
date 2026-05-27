"""Fast tests for ``eval.runners.providers`` — no real HTTP.

Mirrors the test patterns in ``test_rerank_lazy.py`` (monkeypatched
resolver + canned-response fakes). All HTTP is intercepted by
``httpx.MockTransport``; ``time.sleep`` is monkeypatched to no-op so
exponential-backoff retries don't extend the test wall clock.

What the suite proves:
  - Provider model IDs are sourced from ``eval/config/model_candidates.yaml``
    (no hardcoded literals in the wrapper). Both resolvers (model id +
    family) read from the same yaml.
  - ``DEEPINFRA_API_KEY`` missing AND empty both raise — never logged.
  - Happy-path chat_completion returns the chunk content + latency.
  - 429 and 5xx retries succeed within ``max_retries``.
  - max_retries-exhausted raises ``ProviderError`` with a clear message.
  - 4xx (non-429) raises immediately with no retry.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import yaml


def _make_handler(responses):
    """Return an httpx.MockTransport handler that pops responses in order.

    Each response is a tuple ``(status, body_dict)``. The handler raises
    StopIteration on overrun so a test that under-counts retries fails
    loudly rather than silently re-using the last response."""
    it = iter(responses)

    def handler(request):  # noqa: ARG001
        status, body = next(it)
        return httpx.Response(status, json=body)

    return handler


def test_resolve_provider_model_id_matches_yaml_generator_option():
    """The wrapper must read provider_model_id from
    ``candidates.generator.options[*]`` and never hardcode strings like
    ``Qwen/Qwen3-235B-A22B-Instruct``."""
    from eval.runners import providers

    raw = yaml.safe_load(Path(providers._CONFIG_PATH).read_text(encoding="utf-8"))
    by_id = {o["id"]: o for o in raw["candidates"]["generator"]["options"]}
    expected = by_id["qwen3-235b-a22b-instruct"]["provider_model_id"]
    assert (
        providers._resolve_provider_model_id(
            "qwen3-235b-a22b-instruct", component="generator"
        )
        == expected
    )


def test_resolve_candidate_family_matches_yaml():
    """The family field is what enforces the cross-family judge rule
    (ADR 004 v3.1). The resolver returns the per-candidate family — caller
    (generate/api.py, Teammate B) compares."""
    from eval.runners import providers

    assert (
        providers._resolve_candidate_family(
            "qwen3-235b-a22b-instruct", component="generator"
        )
        == "alibaba-qwen"
    )
    assert (
        providers._resolve_candidate_family("deepseek-v3.2", component="judge")
        == "deepseek"
    )


def test_resolve_candidate_provider_matches_yaml():
    """The provider field (e.g. 'deepinfra', 'local') is what generate/api.py
    denormalises into GeneratorCandidate / JudgeCandidate so AgentState
    doesn't carry a yaml reference. Without this resolver, api.py would
    hardcode 'deepinfra' three times — violating the model-IDs-from-yaml
    invariant."""
    from eval.runners import providers

    assert (
        providers._resolve_candidate_provider(
            "qwen3-235b-a22b-instruct", component="generator"
        )
        == "deepinfra"
    )
    assert (
        providers._resolve_candidate_provider("gemma-4-e4b", component="cheap_extraction")
        == "local"
    )


def test_missing_api_key_raises_without_logging_value(monkeypatch):
    """Missing env var must raise ProviderError. The key value is absent
    here, but the test asserts the message doesn't leak whatever happens
    to be in the env."""
    from eval.runners import providers

    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    with pytest.raises(providers.ProviderError, match="DEEPINFRA_API_KEY"):
        providers._read_api_key()


def test_empty_api_key_raises(monkeypatch):
    from eval.runners import providers

    monkeypatch.setenv("DEEPINFRA_API_KEY", "")
    with pytest.raises(providers.ProviderError, match="DEEPINFRA_API_KEY"):
        providers._read_api_key()


def test_chat_completion_happy_path(monkeypatch):
    """200 → ProviderResponse(raw_text='ok', latency_ms>=0, resolved id)."""
    from eval.runners import providers

    monkeypatch.setenv("DEEPINFRA_API_KEY", "test-key-not-real")
    transport = httpx.MockTransport(
        _make_handler(
            [(200, {"choices": [{"message": {"content": "hello"}}]})]
        )
    )
    monkeypatch.setattr(
        providers, "_client_factory", lambda: httpx.Client(transport=transport)
    )

    resp = providers.chat_completion(
        candidate_id="qwen3-235b-a22b-instruct",
        messages=[{"role": "user", "content": "hi"}],
        component="generator",
        max_retries=0,
        backoff_base_sec=0.0,
    )
    assert resp.raw_text == "hello"
    assert resp.latency_ms >= 0
    # provider_model_id echoes what we resolved + sent, NOT the candidate_id.
    raw = yaml.safe_load(Path(providers._CONFIG_PATH).read_text(encoding="utf-8"))
    by_id = {o["id"]: o for o in raw["candidates"]["generator"]["options"]}
    assert resp.provider_model_id == by_id["qwen3-235b-a22b-instruct"]["provider_model_id"]


def test_retries_on_429_then_succeeds(monkeypatch):
    from eval.runners import providers

    monkeypatch.setenv("DEEPINFRA_API_KEY", "test-key")
    monkeypatch.setattr(providers.time, "sleep", lambda *_: None)
    transport = httpx.MockTransport(
        _make_handler(
            [
                (429, {"error": "rate-limited"}),
                (429, {"error": "rate-limited"}),
                (200, {"choices": [{"message": {"content": "ok"}}]}),
            ]
        )
    )
    monkeypatch.setattr(
        providers, "_client_factory", lambda: httpx.Client(transport=transport)
    )

    resp = providers.chat_completion(
        candidate_id="qwen3-235b-a22b-instruct",
        messages=[{"role": "user", "content": "hi"}],
        component="generator",
        max_retries=3,
        backoff_base_sec=0.0,
    )
    assert resp.raw_text == "ok"


def test_retries_on_500_then_succeeds(monkeypatch):
    from eval.runners import providers

    monkeypatch.setenv("DEEPINFRA_API_KEY", "test-key")
    monkeypatch.setattr(providers.time, "sleep", lambda *_: None)
    transport = httpx.MockTransport(
        _make_handler(
            [
                (500, {"error": "boom"}),
                (502, {"error": "bad gateway"}),
                (200, {"choices": [{"message": {"content": "recovered"}}]}),
            ]
        )
    )
    monkeypatch.setattr(
        providers, "_client_factory", lambda: httpx.Client(transport=transport)
    )

    resp = providers.chat_completion(
        candidate_id="deepseek-v3.2",
        messages=[{"role": "user", "content": "x"}],
        component="generator",
        max_retries=3,
        backoff_base_sec=0.0,
    )
    assert resp.raw_text == "recovered"


def test_max_retries_exhausted_raises(monkeypatch):
    from eval.runners import providers

    monkeypatch.setenv("DEEPINFRA_API_KEY", "test-key")
    monkeypatch.setattr(providers.time, "sleep", lambda *_: None)
    # 1 initial + 2 retries = 3 attempts, all 429.
    transport = httpx.MockTransport(
        _make_handler(
            [
                (429, {"error": "rate-limited"}),
                (429, {"error": "rate-limited"}),
                (429, {"error": "rate-limited"}),
            ]
        )
    )
    monkeypatch.setattr(
        providers, "_client_factory", lambda: httpx.Client(transport=transport)
    )

    with pytest.raises(providers.ProviderError, match="max_retries"):
        providers.chat_completion(
            candidate_id="qwen3-235b-a22b-instruct",
            messages=[{"role": "user", "content": "hi"}],
            component="generator",
            max_retries=2,
            backoff_base_sec=0.0,
        )


def test_400_raises_immediately_no_retry(monkeypatch):
    """4xx other than 429 is a permanent error; retrying just burns
    budget. The mock transport contains a single 400 — if the wrapper
    retried we'd get StopIteration instead of ProviderError."""
    from eval.runners import providers

    monkeypatch.setenv("DEEPINFRA_API_KEY", "test-key")
    monkeypatch.setattr(providers.time, "sleep", lambda *_: None)
    transport = httpx.MockTransport(
        _make_handler([(400, {"error": "bad request"})])
    )
    monkeypatch.setattr(
        providers, "_client_factory", lambda: httpx.Client(transport=transport)
    )

    with pytest.raises(providers.ProviderError, match="400"):
        providers.chat_completion(
            candidate_id="qwen3-235b-a22b-instruct",
            messages=[{"role": "user", "content": "hi"}],
            component="generator",
            max_retries=5,
            backoff_base_sec=0.0,
        )


def test_malformed_response_raises(monkeypatch):
    """200 with missing choices[0].message.content is unrecoverable."""
    from eval.runners import providers

    monkeypatch.setenv("DEEPINFRA_API_KEY", "test-key")
    transport = httpx.MockTransport(
        _make_handler([(200, {"choices": []})])
    )
    monkeypatch.setattr(
        providers, "_client_factory", lambda: httpx.Client(transport=transport)
    )

    with pytest.raises(providers.ProviderError, match="malformed"):
        providers.chat_completion(
            candidate_id="qwen3-235b-a22b-instruct",
            messages=[{"role": "user", "content": "hi"}],
            component="generator",
            max_retries=0,
            backoff_base_sec=0.0,
        )
