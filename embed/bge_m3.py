"""BGE-M3 embedder — lazy singleton with batched 3-channel encode.

First call loads the candidate identified by
``model_candidates.yaml::text_embeddings[bge-m3-all-channels]`` (~2.3GB).
Subsequent encodes reuse the in-process instance. Returns dense (1024-d),
sparse (lexical_weights dict per sentence: token_id -> weight), and
multi-vector (per-token 1024-d arrays per sentence — used for MaxSim
retrieval).

Hardware: PyTorch picks MPS on Apple Silicon when available, else CPU.
Tokenizer is XLM-RoBERTa-large; vocab_size == 250002 is a load-bearing
contract (see eval/schemas + sparse_embeds sparsevec(250002) in
db/migrations/0002).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

VOCAB_SIZE = 250002
DENSE_DIM = 1024

_CANDIDATE_ID = "bge-m3-all-channels"
_REQUIRED_CHANNELS = frozenset({"dense", "sparse", "multi_vector"})
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "eval" / "config" / "model_candidates.yaml"


@dataclass(frozen=True)
class BgeM3Output:
    """Three-channel encoding result for one or more sentences."""

    dense: np.ndarray  # (N, DENSE_DIM)
    sparse: list[dict[int, float]]  # lexical_weights per sentence
    multi: list[np.ndarray]  # per-token vectors per sentence (T_i, DENSE_DIM)


def _resolve_model_id() -> str:
    """Load model_candidates.yaml, return provider_model_id for the
    bge-m3-all-channels candidate. Validates provider=='local' and that
    channels include dense + sparse + multi_vector — any drift triggers
    a ValueError so the wrapper fails loudly at boot, not silently mid-encode."""
    import yaml

    raw = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    options = raw["candidates"]["text_embeddings"]["options"]
    for opt in options:
        if opt.get("id") != _CANDIDATE_ID:
            continue
        if opt.get("provider") != "local":
            raise ValueError(
                f"{_CANDIDATE_ID} provider must be 'local', got {opt.get('provider')!r}"
            )
        channels = set(opt.get("channels", []))
        missing = _REQUIRED_CHANNELS - channels
        if missing:
            raise ValueError(
                f"{_CANDIDATE_ID} channels missing: {sorted(missing)}; got {sorted(channels)}"
            )
        return opt["provider_model_id"]
    raise KeyError(f"candidate {_CANDIDATE_ID!r} not found in text_embeddings options")


def _normalize_sparse(raw_sparse: list[dict]) -> list[dict[int, float]]:
    """BGE-M3's lexical_weights uses string token ids and numpy floats.
    Coerce to dict[int, float] at the wrapper boundary so callers see
    the type the dataclass claims."""
    return [{int(k): float(v) for k, v in d.items()} for d in raw_sparse]


@lru_cache(maxsize=1)
def _model() -> Any:
    """Lazy singleton. Heavy import deferred until first call.
    Model id is resolved from model_candidates.yaml at first call — see
    _resolve_model_id."""
    from FlagEmbedding import BGEM3FlagModel

    return BGEM3FlagModel(_resolve_model_id(), use_fp16=True)


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
        sparse=_normalize_sparse(raw["lexical_weights"]),
        multi=raw["colbert_vecs"],
    )
