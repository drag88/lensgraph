"""Fast wrapper-level tests for embed.bge_m3 — no model load."""

from __future__ import annotations


def test_encode_empty_does_not_load_model(monkeypatch):
    """encode([]) must short-circuit before _model() is called. Order-
    independent: we clear any cached singleton from prior tests and
    monkeypatch _model with a raise-on-call sentinel so a regression
    would fail loudly rather than just be slow."""
    from embed import bge_m3

    bge_m3._model.cache_clear()  # drop any cached instance from earlier tests

    def boom():
        raise AssertionError("_model() must not be called for encode([])")

    monkeypatch.setattr(bge_m3, "_model", boom)

    out = bge_m3.encode([])
    assert out.dense.shape == (0, bge_m3.DENSE_DIM)
    assert out.sparse == []
    assert out.multi == []
