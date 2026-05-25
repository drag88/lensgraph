"""ColQwen2.5 image + text encoder for the visual retrieval channel.

Loads vidore/ColQwen2.5 via colpali-engine. Phase-1 step-19/20 exposes
only the pooled image encoder + a text query encoder — the two functions
the pooled-prefilter stage of retrieve/visual.py needs. Full per-patch
embeddings for MaxSim land in step 24.

Hardware: torch picks MPS on Apple Silicon when available, else CPU.
Modal remote inference is the deferred fallback (`docs/phase-1-design.md`
§7 lists MODAL_TOKEN_ID / MODAL_TOKEN_SECRET env vars but no wiring yet).

Pooled embedding dim: matches frames.pooled_embedding column type
vector(128) in db/migrations/0003_frames.sql. ColQwen2.5 emits 128-d
per-patch embeddings; the pooled vector is the mean across patches.
The dim is validated against the model output at first encode call —
mismatch aborts loudly rather than silently producing 0-vectors that
HNSW indexes will happily accept.

colpali-engine API path (probed 2026-05-25 on 0.3.16):
``from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

POOLED_DIM = 128


@lru_cache(maxsize=1)
def _model() -> Any:
    """Load ColQwen2.5 + its processor. Heavy import deferred until first call."""
    import torch
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = ColQwen2_5.from_pretrained(
        "vidore/colqwen2.5-v0.2",
        torch_dtype=torch.float16 if device == "mps" else torch.float32,
        device_map=device,
    ).eval()
    processor = ColQwen2_5_Processor.from_pretrained("vidore/colqwen2.5-v0.2")
    return model, processor, device


def encode_image_pooled(images: list) -> np.ndarray:
    """Encode a batch of PIL images. Returns (N, POOLED_DIM) float32 —
    one mean-pooled vector per image.

    Empty input returns shape (0, POOLED_DIM) without loading the model.
    """
    if not images:
        return np.zeros((0, POOLED_DIM), dtype=np.float32)

    import torch

    model, processor, device = _model()
    batch = processor.process_images(images).to(device)
    with torch.no_grad():
        # colpali_engine model output shape: (batch, num_patches, dim)
        patch_embs = model(**batch)
    pooled = patch_embs.mean(dim=1).to(torch.float32).cpu().numpy()

    if pooled.shape[1] != POOLED_DIM:
        raise RuntimeError(
            f"ColQwen2.5 pooled dim mismatch: got {pooled.shape[1]}, expected {POOLED_DIM}"
        )
    return pooled


def encode_text_query(query: str) -> np.ndarray:
    """Encode a text query for cosine search against pooled image embeds.

    Returns (POOLED_DIM,) float32. Empty / whitespace-only query returns
    a zero vector without loading the model — HNSW handles zero-norm via
    its distance operator (cosine is undefined for zero norm; pgvector
    returns NULL, ordered last).
    """
    if not query or not query.strip():
        return np.zeros((POOLED_DIM,), dtype=np.float32)

    import torch

    model, processor, device = _model()
    batch = processor.process_queries([query]).to(device)
    with torch.no_grad():
        token_embs = model(**batch)
    pooled = token_embs.mean(dim=1).to(torch.float32).cpu().numpy()[0]
    if pooled.shape[0] != POOLED_DIM:
        raise RuntimeError(
            f"ColQwen2.5 text-pooled dim mismatch: got {pooled.shape[0]}, expected {POOLED_DIM}"
        )
    return pooled
