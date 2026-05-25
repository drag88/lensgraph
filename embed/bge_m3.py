"""BGE-M3 embedder — lazy singleton with batched 3-channel encode.

First call loads BAAI/bge-m3 (~2.3GB). Subsequent encodes reuse the
in-process instance. Returns dense (1024-d), sparse (lexical_weights
dict per sentence: token_id -> weight), and multi-vector (per-token
1024-d arrays per sentence — used for MaxSim retrieval).

Hardware: PyTorch picks MPS on Apple Silicon when available, else CPU.
Tokenizer is XLM-RoBERTa-large; vocab_size == 250002 is a load-bearing
contract (see eval/schemas + sparse_embeds sparsevec(250002) in
db/migrations/0002).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

VOCAB_SIZE = 250002
DENSE_DIM = 1024
_MODEL_ID = "BAAI/bge-m3"


@dataclass(frozen=True)
class BgeM3Output:
    """Three-channel encoding result for one or more sentences."""

    dense: np.ndarray              # (N, DENSE_DIM)
    sparse: list[dict[int, float]]  # lexical_weights per sentence
    multi: list[np.ndarray]         # per-token vectors per sentence (T_i, DENSE_DIM)


@lru_cache(maxsize=1)
def _model() -> Any:
    """Lazy singleton. Heavy import deferred until first call."""
    from FlagEmbedding import BGEM3FlagModel

    return BGEM3FlagModel(_MODEL_ID, use_fp16=True)


def vocab_size() -> int:
    """Return the tokenizer vocab size. Must be VOCAB_SIZE (250002)."""
    return _model().tokenizer.vocab_size


def encode(
    sentences: list[str],
    *,
    batch_size: int = 12,
    max_length: int = 512,
) -> BgeM3Output:
    """Encode a batch into all three channels. Empty input returns empty
    channels without loading the model."""
    if not sentences:
        return BgeM3Output(
            dense=np.zeros((0, DENSE_DIM), dtype=np.float32),
            sparse=[],
            multi=[],
        )
    raw = _model().encode(
        sentences,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )
    return BgeM3Output(
        dense=raw["dense_vecs"],
        sparse=raw["lexical_weights"],
        multi=raw["colbert_vecs"],
    )
