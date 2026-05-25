"""Fast wrapper-level tests for embed.colqwen — no model load.

Mirrors test_bge_m3_lazy.py: prove that (a) the model id is sourced
from model_candidates.yaml rather than hardcoded, (b) zero-input paths
short-circuit before the heavy ColQwen2.5 weights would be pulled, and
(c) yaml drift to a hosted provider or a missing candidate fails loudly
at boot.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def test_resolve_model_id_matches_yaml_visual_retrieval_colqwen():
    """_resolve_model_id reads from eval/config/model_candidates.yaml — no
    string is allowed to be hardcoded in the wrapper."""
    from embed import colqwen

    raw = yaml.safe_load(Path(colqwen._CONFIG_PATH).read_text(encoding="utf-8"))
    expected = next(
        opt["provider_model_id"]
        for opt in raw["candidates"]["visual_retrieval"]["options"]
        if opt["id"] == colqwen._CANDIDATE_ID
    )
    assert colqwen._resolve_model_id() == expected


def test_resolve_model_id_rejects_non_local_provider(monkeypatch, tmp_path):
    """A yaml drift that flips colqwen2.5 to a hosted provider must raise —
    the wrapper's whole point is local MPS inference."""
    from embed import colqwen

    bad = {
        "candidates": {
            "visual_retrieval": {
                "options": [
                    {
                        "id": colqwen._CANDIDATE_ID,
                        "provider": "openrouter",
                        "provider_model_id": "vidore/colqwen2.5-v0.2",
                    }
                ]
            }
        }
    }
    drift = tmp_path / "model_candidates.yaml"
    drift.write_text(yaml.safe_dump(bad), encoding="utf-8")
    monkeypatch.setattr(colqwen, "_CONFIG_PATH", drift)

    with pytest.raises(ValueError, match="provider must be 'local'"):
        colqwen._resolve_model_id()


def test_resolve_model_id_missing_candidate_raises(monkeypatch, tmp_path):
    """If someone renames the candidate id, surface a KeyError naming the id
    we expected rather than letting it silently fall through to defaults."""
    from embed import colqwen

    bad = {
        "candidates": {
            "visual_retrieval": {
                "options": [
                    {
                        "id": "some-other-id",
                        "provider": "local",
                        "provider_model_id": "vidore/colqwen2.5-v0.2",
                    }
                ]
            }
        }
    }
    drift = tmp_path / "model_candidates.yaml"
    drift.write_text(yaml.safe_dump(bad), encoding="utf-8")
    monkeypatch.setattr(colqwen, "_CONFIG_PATH", drift)

    with pytest.raises(KeyError, match=colqwen._CANDIDATE_ID):
        colqwen._resolve_model_id()


def test_encode_image_pooled_empty_does_not_load_model(monkeypatch):
    """encode_image_pooled([]) must short-circuit before _model() is called.
    Order-independent: clear any cached singleton and raise on call."""
    from embed import colqwen

    colqwen._model.cache_clear()

    def boom():
        raise AssertionError("_model() must not be called for empty input")

    monkeypatch.setattr(colqwen, "_model", boom)

    out = colqwen.encode_image_pooled([])
    assert out.shape == (0, colqwen.POOLED_DIM)


def test_encode_text_query_empty_does_not_load_model(monkeypatch):
    """encode_text_query('') and whitespace-only inputs must short-circuit."""
    from embed import colqwen

    colqwen._model.cache_clear()

    def boom():
        raise AssertionError("_model() must not be called for empty query")

    monkeypatch.setattr(colqwen, "_model", boom)

    assert colqwen.encode_text_query("").shape == (colqwen.POOLED_DIM,)
    assert colqwen.encode_text_query("   \n\t").shape == (colqwen.POOLED_DIM,)
