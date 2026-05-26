"""Fast wrapper-level tests for retrieve.rerank — no model load.

Mirrors test_bge_m3_lazy.py / test_colqwen_lazy.py: prove that
(a) the model id is sourced from model_candidates.yaml (no hardcoded
literal), (b) a yaml drift to a hosted provider fails loudly at boot,
(c) ``rerank()`` with empty candidates short-circuits before the heavy
FlagReranker weights would be pulled.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def test_resolve_model_id_matches_yaml_reranker_option():
    """_resolve_model_id reads from eval/config/model_candidates.yaml — no
    string is allowed to be hardcoded in the wrapper. v0 has exactly one
    reranker option (BGE-reranker-v2-m3); the resolver picks options[0]."""
    from retrieve import rerank

    raw = yaml.safe_load(Path(rerank._CONFIG_PATH).read_text(encoding="utf-8"))
    expected = raw["candidates"]["reranker"]["options"][0]["provider_model_id"]
    assert rerank._resolve_model_id() == expected


def test_resolve_model_id_rejects_non_local_provider(monkeypatch, tmp_path):
    """A yaml drift that flips the reranker to a hosted provider must raise —
    the wrapper's whole point is local CPU inference (ADR 004 + design §4)."""
    from retrieve import rerank

    bad = {
        "candidates": {
            "reranker": {
                "options": [
                    {
                        "id": "drifted-id",
                        "provider": "openrouter",
                        "provider_model_id": "vendor/drifted-model",
                        "family": "bge",
                    }
                ]
            }
        }
    }
    drift = tmp_path / "model_candidates.yaml"
    drift.write_text(yaml.safe_dump(bad), encoding="utf-8")
    monkeypatch.setattr(rerank, "_CONFIG_PATH", drift)

    with pytest.raises(ValueError, match="provider must be 'local'"):
        rerank._resolve_model_id()


def test_resolve_model_id_missing_options_raises(monkeypatch, tmp_path):
    """If the reranker section is renamed or emptied, surface a KeyError
    rather than letting it silently fall through."""
    from retrieve import rerank

    bad = {"candidates": {"reranker": {"options": []}}}
    drift = tmp_path / "model_candidates.yaml"
    drift.write_text(yaml.safe_dump(bad), encoding="utf-8")
    monkeypatch.setattr(rerank, "_CONFIG_PATH", drift)

    with pytest.raises(KeyError, match="empty"):
        rerank._resolve_model_id()


def test_rerank_empty_candidates_does_not_load_model(monkeypatch):
    """rerank(.., candidates=[]) must short-circuit before _reranker() is
    called. Order-independent: clear any cached singleton from prior tests
    and raise on call so a regression fails loudly rather than just being
    slow."""
    from retrieve import rerank

    rerank._reranker.cache_clear()

    def boom():
        raise AssertionError("_reranker() must not be called for empty candidates")

    monkeypatch.setattr(rerank, "_reranker", boom)

    out = rerank.rerank(conn=None, query="anything", candidates=[])
    assert out == []
