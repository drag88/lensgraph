"""Slow tests for embed.bge_m3 — model load + three-channel contract."""

from __future__ import annotations

import pytest

from embed.bge_m3 import DENSE_DIM, VOCAB_SIZE, encode, vocab_size

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def embedder():
    from embed.bge_m3 import _model

    return _model()


def test_vocab_size_is_250002(embedder):
    assert vocab_size() == VOCAB_SIZE


def test_dense_shape_is_1024(embedder):
    out = encode(["hello world"])
    assert out.dense.shape == (1, DENSE_DIM)


def test_sparse_is_non_empty_per_sentence(embedder):
    out = encode(["hello world", "another sentence about retrieval"])
    assert len(out.sparse) == 2
    assert all(len(s) > 0 for s in out.sparse)


def test_multi_token_count_matches_tokenizer_output(embedder):
    text = "hello world this is a retrieval sentence"
    out = encode([text])
    tokenizer_output = embedder.tokenizer(text, truncation=True, max_length=512)
    # BGE-M3 colbert_vecs keeps <s> (CLS) but strips the trailing </s> (EOS),
    # so per-token vector count is len(input_ids) - 1.
    assert out.multi[0].shape[0] == len(tokenizer_output["input_ids"]) - 1
    assert out.multi[0].shape[1] == DENSE_DIM


def test_empty_encode_returns_empty_without_model_load():
    out = encode([])
    assert out.dense.shape == (0, DENSE_DIM)
    assert out.sparse == []
    assert out.multi == []


def test_sparse_keys_are_int_and_in_vocab_range(embedder):
    """Wrapper normalizes BGE's str/np.float lexical_weights to plain
    dict[int, float]. Token ids must lie in [0, VOCAB_SIZE)."""
    out = encode(["hello world this is a retrieval sentence"])
    assert len(out.sparse) == 1
    assert out.sparse[0], "expected non-empty lexical weights"
    for token_id, weight in out.sparse[0].items():
        assert isinstance(token_id, int)
        assert isinstance(weight, float)
        assert 0 <= token_id < VOCAB_SIZE
