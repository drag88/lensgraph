"""Fast wrapper-level tests for retrieve.rerank — no model load.

Mirrors test_bge_m3_lazy.py / test_colqwen_lazy.py: prove that
(a) the model id is sourced from model_candidates.yaml (no hardcoded
literal), (b) a yaml drift to a hosted provider fails loudly at boot,
(c) ``rerank()`` with empty candidates short-circuits before the heavy
CrossEncoder weights would be pulled, (d) the activation pipeline is
sigmoid-once (NOT sigmoid-of-sigmoid).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from retrieve.types import FusedResult


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


class _FakeCrossEncoder:
    """Returns fixed raw logits regardless of the input pairs. Records the
    activation_fn kwarg so the test can prove rerank.py asked for raw
    logits (Identity) and not the model's default Sigmoid."""

    def __init__(self, raw_logits: list[float]) -> None:
        self._raw = np.asarray(raw_logits, dtype=np.float32)
        self.last_activation_fn = None

    def predict(self, pairs, convert_to_numpy=True, activation_fn=None):  # noqa: ARG002
        self.last_activation_fn = activation_fn
        assert len(pairs) == len(self._raw), (
            f"fake CrossEncoder configured for {len(self._raw)} pairs; "
            f"got {len(pairs)}"
        )
        return self._raw.copy()


def _stub_fused(chunk_id: int, text: str) -> FusedResult:
    """Minimal FusedResult for the activation-pipeline test."""
    return FusedResult(
        chunk_id=chunk_id,
        video_id="vid",
        start_sec=0.0,
        end_sec=10.0,
        text=text,
        score=0.0,
        rank=chunk_id,
        channel_ranks={"bm25": chunk_id},
    )


def test_rerank_score_is_sigmoid_of_raw_logit_not_sigmoid_of_sigmoid(monkeypatch):
    """Catches the double-sigmoid regression where rerank() would apply
    sigmoid AFTER CrossEncoder.predict() already applied the model's
    default Sigmoid activation. With activation_fn=Identity() bypassing
    the model activation, rerank_score must equal sigmoid(raw) — NOT
    sigmoid(sigmoid(raw)).

    Raw logits used: [0, 2, -2]. Expected rerank_scores after one
    sigmoid: [0.5, ~0.881, ~0.119]. Under the regressed
    sigmoid(sigmoid(x)) the same inputs would give [~0.622, ~0.707,
    ~0.530] — all pinned to (0.5, ~0.73)."""
    import torch

    from retrieve import rerank

    raw = [0.0, 2.0, -2.0]
    fake = _FakeCrossEncoder(raw)
    rerank._reranker.cache_clear()
    monkeypatch.setattr(rerank, "_reranker", lambda: fake)

    candidates = [
        _stub_fused(1, "alpha"),
        _stub_fused(2, "beta"),
        _stub_fused(3, "gamma"),
    ]
    out = rerank.rerank(conn=None, query="q", candidates=candidates, top_k=3)
    by_chunk = {r.chunk_id: r for r in out}

    expected = {1: 1 / (1 + math.exp(-0.0)),
                2: 1 / (1 + math.exp(-2.0)),
                3: 1 / (1 + math.exp(2.0))}
    for cid, want in expected.items():
        got = by_chunk[cid].rerank_score
        assert math.isclose(got, want, abs_tol=1e-5), (
            f"chunk_id={cid}: rerank_score={got!r} expected ~{want!r}. "
            "If got ~sigmoid(sigmoid(raw)) instead, the activation pipeline "
            "is double-sigmoiding."
        )

    assert isinstance(fake.last_activation_fn, torch.nn.Identity), (
        "rerank.py must pass activation_fn=torch.nn.Identity() to bypass "
        f"the model's default Sigmoid; got {fake.last_activation_fn!r}"
    )
