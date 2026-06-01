"""Fast tests for the strategy-aware retrieval filter.

Each text channel takes ``chunking_strategy`` (default ``fixed_window``) so the
chunking ablation can keep two strategies' chunks in one table without
retrieval mixing them. These tests mock the DB connection and stub the encoder
so they run with no Postgres and no BGE-M3 load.
"""

from __future__ import annotations

import numpy as np

from retrieve import bm25, dense, multivec, sparse


class _RecordingConn:
    """Captures every execute(sql, params); returns no rows."""

    def __init__(self):
        self.calls: list[tuple[str, tuple]] = []

    def execute(self, sql, params=()):
        self.calls.append((sql, params))
        return self

    def fetchall(self):
        return []


def _str_params(params) -> list[str]:
    """String params only — avoids ambiguous truth-value on numpy query vecs."""
    return [p for p in params if isinstance(p, str)]


def _stub_encode(monkeypatch, module):
    class _Out:
        dense = [np.zeros(4, dtype=np.float32)]
        sparse = [{0: 1.0}]
        multi = [np.zeros((1, 4), dtype=np.float32)]

    monkeypatch.setattr(module, "encode", lambda *_a, **_k: _Out())
    monkeypatch.setattr(module, "register_vector", lambda *_a, **_k: None, raising=False)


def test_dense_threads_strategy_and_defaults_fixed_window(monkeypatch):
    _stub_encode(monkeypatch, dense)
    conn = _RecordingConn()
    dense.retrieve(conn, "q", chunking_strategy="transcript_segment")
    sql, params = conn.calls[-1]
    assert "c.chunking_strategy = %s" in sql
    assert "transcript_segment" in _str_params(params)
    # default
    conn2 = _RecordingConn()
    dense.retrieve(conn2, "q")
    assert "fixed_window" in _str_params(conn2.calls[-1][1])


def test_sparse_threads_strategy(monkeypatch):
    monkeypatch.setattr(sparse, "encode", lambda *_a, **_k: type("O", (), {"sparse": [{0: 1.0}]})())
    monkeypatch.setattr(sparse, "register_vector", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(sparse, "SparseVector", lambda *a, **k: object())
    conn = _RecordingConn()
    sparse.retrieve(conn, "q", chunking_strategy="transcript_segment")
    sql, params = conn.calls[-1]
    assert "c.chunking_strategy = %s" in sql
    assert "transcript_segment" in _str_params(params)


def test_bm25_threads_strategy_precise_and_fallback():
    # precise path returns rows -> only precise SQL runs
    conn = _RecordingConn()
    bm25.retrieve(conn, "hello world", chunking_strategy="transcript_segment")
    sql, params = conn.calls[-1]
    assert "chunking_strategy = %s" in sql
    assert "transcript_segment" in _str_params(params)

    # fallback path: precise returns [] so the OR-expansion query runs too.
    # both executed queries must carry the strategy.
    conn2 = _RecordingConn()
    bm25.retrieve(conn2, "hello world", chunking_strategy="transcript_segment")
    for sql, params in conn2.calls:
        assert "chunking_strategy = %s" in sql
        assert "transcript_segment" in _str_params(params)


def test_multivec_forwards_strategy_to_prefilter(monkeypatch):
    seen = {}

    def _rec_dense(conn, q, *, top_k, chunking_strategy="fixed_window"):
        seen["dense"] = chunking_strategy
        return []

    def _rec_sparse(conn, q, *, top_k, chunking_strategy="fixed_window"):
        seen["sparse"] = chunking_strategy
        return []

    monkeypatch.setattr(multivec.dense, "retrieve", _rec_dense)
    monkeypatch.setattr(multivec.sparse, "retrieve", _rec_sparse)
    monkeypatch.setattr(
        multivec, "encode", lambda *_a, **_k: type("O", (), {"multi": [np.zeros((1, 4))]})()
    )
    out = multivec.retrieve(_RecordingConn(), "q", chunking_strategy="transcript_segment")
    assert out == []  # empty prefilter -> early return
    assert seen == {"dense": "transcript_segment", "sparse": "transcript_segment"}


def test_all_channels_default_to_fixed_window():
    import inspect

    for mod in (dense, sparse, multivec, bm25):
        sig = inspect.signature(mod.retrieve)
        assert sig.parameters["chunking_strategy"].default == "fixed_window"
