"""ColQwen2.5 image + text encoder for the visual retrieval channel.

Loads the candidate identified by
``model_candidates.yaml::visual_retrieval[colqwen2.5]`` via colpali-engine.
Phase-1 step-19/20 exposed the pooled image encoder + a text query
encoder (for the HNSW prefilter stage of ``retrieve/visual.py``).
Phase-1 step-24/25 (this module's slice 2 form) adds the full per-patch
encoders + a ``pool_patches`` helper so the MaxSim refine pass and the
``embed_frames_handler`` consume the SAME patches that get pooled —
``frames.pooled_embedding`` and ``frame_patches`` can never disagree.

Hardware: torch picks MPS on Apple Silicon when available, else CPU.
Modal remote inference is the deferred fallback (`docs/phase-1-design.md`
§7 lists MODAL_TOKEN_ID / MODAL_TOKEN_SECRET env vars but no wiring yet).

Pooled embedding dim: matches frames.pooled_embedding column type
vector(128) in db/migrations/0003_frames.sql. ColQwen2.5 emits 128-d
per-patch embeddings; the pooled vector is the mean across patches.
The dim is validated against the model output at first encode call —
mismatch aborts loudly rather than silently producing 0-vectors that
HNSW indexes will happily accept.

**Variable per-image patch count.** ColQwen2.5's image encoder emits a
variable number of patches per image (depends on the image's aspect
ratio / token count after the vision encoder). When a batch has mixed
patch counts colpali-engine returns a padded ``(batch, max_patches, dim)``
tensor with an ``attention_mask`` on the input batch dict — we trim per
image with that mask before stacking back as a ragged 3-D ndarray
padded with zeros. The zero padding is benign for MaxSim downstream
(zero rows contribute zero similarity), but ``encode_image_patches``
returns the trimmed-then-uniformly-padded matrix when ``len(images) > 1``
and returns the exact (P, 128) matrix when ``len(images) == 1`` — the
embed_frames_handler always encodes one image at a time so it never
sees zero-padded rows.

colpali-engine API path (probed 2026-05-25 on 0.3.16):
``from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

POOLED_DIM = 128

_CANDIDATE_ID = "colqwen2.5"
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "eval" / "config" / "model_candidates.yaml"


def _resolve_model_id() -> str:
    """Load model_candidates.yaml, return provider_model_id for the
    colqwen2.5 candidate. Validates provider=='local' so a yaml drift to
    Modal/hosted is caught at boot, not silently mid-encode."""
    import yaml

    raw = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    options = raw["candidates"]["visual_retrieval"]["options"]
    for opt in options:
        if opt.get("id") != _CANDIDATE_ID:
            continue
        if opt.get("provider") != "local":
            raise ValueError(
                f"{_CANDIDATE_ID} provider must be 'local', got {opt.get('provider')!r}"
            )
        return opt["provider_model_id"]
    raise KeyError(f"candidate {_CANDIDATE_ID!r} not found in visual_retrieval options")


@lru_cache(maxsize=1)
def _model() -> Any:
    """Load ColQwen2.5 + its processor. Heavy import deferred until first
    call. Model id resolved from model_candidates.yaml — see _resolve_model_id."""
    import torch
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

    model_id = _resolve_model_id()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = ColQwen2_5.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "mps" else torch.float32,
        device_map=device,
    ).eval()
    processor = ColQwen2_5_Processor.from_pretrained(model_id)
    return model, processor, device


def encode_image_patches(images: list) -> np.ndarray:
    """Encode a batch of PIL images. Returns (N, P, POOLED_DIM) float32 —
    the full per-patch matrix, NOT pooled.

    Empty input returns shape (0, 0, POOLED_DIM) without loading the model.

    P is the variable per-image patch count. For batched input with mixed
    patch counts, rows past each image's true patch count are zero-padded
    (benign for MaxSim — zero rows contribute zero similarity). The
    embed_frames_handler encodes one image at a time, so it never sees
    padding rows; multi-image batching is a future optimization.
    """
    if not images:
        return np.zeros((0, 0, POOLED_DIM), dtype=np.float32)

    import torch

    model, processor, device = _model()
    batch = processor.process_images(images).to(device)
    with torch.no_grad():
        # colpali_engine model output shape: (batch, max_num_patches, dim)
        patch_embs = model(**batch)

    patches_np = patch_embs.to(torch.float32).cpu().numpy()

    if patches_np.ndim != 3:
        raise RuntimeError(
            f"ColQwen2.5 patches output must be 3-D (batch, patches, dim); "
            f"got shape {patches_np.shape}"
        )
    if patches_np.shape[2] != POOLED_DIM:
        raise RuntimeError(
            f"ColQwen2.5 patch dim mismatch: got {patches_np.shape[2]}, expected {POOLED_DIM}"
        )

    # Trim per-image padding using the input attention_mask. The processor
    # pads shorter images to max_num_patches in the batch; the attention
    # mask flags real-vs-pad positions. Trim then re-pad with zeros to a
    # uniform shape — this preserves the (N, P, D) contract while keeping
    # padding rows zero so MaxSim's max-over-rows is correctness-preserving.
    mask = batch.get("attention_mask")
    if mask is not None:
        mask_np = mask.cpu().numpy().astype(bool)
        true_counts = mask_np.sum(axis=1).tolist()
        if all(c == patches_np.shape[1] for c in true_counts):
            return patches_np  # nothing to trim — all images have full patches
        zeroed = np.zeros_like(patches_np)
        for i, n in enumerate(true_counts):
            zeroed[i, :n, :] = patches_np[i, :n, :]
        return zeroed
    return patches_np


def encode_image_pooled(images: list) -> np.ndarray:
    """Encode a batch of PIL images. Returns (N, POOLED_DIM) float32 —
    one mean-pooled vector per image.

    Empty input returns shape (0, POOLED_DIM) without loading the model.

    Implementation: delegates to encode_image_patches so the pooled vector
    is provably the mean of the same patches that ``embed_frames_handler``
    persists into ``frame_patches``. One encoder path, no drift between
    the HNSW prefilter input and the MaxSim refine input.
    """
    if not images:
        return np.zeros((0, POOLED_DIM), dtype=np.float32)

    patches = encode_image_patches(images)
    pooled = patches.mean(axis=1).astype(np.float32, copy=False)

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

    patches = encode_text_query_patches(query)
    pooled = patches.mean(axis=0).astype(np.float32, copy=False)
    if pooled.shape[0] != POOLED_DIM:
        raise RuntimeError(
            f"ColQwen2.5 text-pooled dim mismatch: got {pooled.shape[0]}, expected {POOLED_DIM}"
        )
    return pooled


def encode_text_query_patches(query: str) -> np.ndarray:
    """Encode a text query as per-token patches. Returns (T_q, POOLED_DIM)
    float32 — the matrix MaxSim consumes against ``frame_patches``.

    Empty / whitespace-only query returns shape (0, POOLED_DIM) without
    loading the model.
    """
    if not query or not query.strip():
        return np.zeros((0, POOLED_DIM), dtype=np.float32)

    import torch

    model, processor, device = _model()
    batch = processor.process_queries([query]).to(device)
    with torch.no_grad():
        token_embs = model(**batch)
    arr = token_embs.to(torch.float32).cpu().numpy()
    if arr.ndim != 3 or arr.shape[0] != 1:
        raise RuntimeError(
            f"ColQwen2.5 text patches output must be (1, T_q, dim); got shape {arr.shape}"
        )
    if arr.shape[2] != POOLED_DIM:
        raise RuntimeError(
            f"ColQwen2.5 text-patch dim mismatch: got {arr.shape[2]}, expected {POOLED_DIM}"
        )
    return arr[0]


def pool_patches(patches: np.ndarray) -> np.ndarray:
    """Mean-pool a (P, POOLED_DIM) patch matrix to (POOLED_DIM,).

    Internal helper so the embed_frames handler can compute the pooled
    vector from the SAME patches it persists — guarantees consistency
    between ``frames.pooled_embedding`` (HNSW prefilter input) and the
    ``frame_patches`` rows (MaxSim refine input).

    P must be >= 1; pooling a zero-row matrix is undefined (mean over
    nothing) and indicates upstream broke its contract.
    """
    if patches.ndim != 2 or patches.shape[1] != POOLED_DIM:
        raise ValueError(
            f"pool_patches expects (P, {POOLED_DIM}); got shape {patches.shape}"
        )
    if patches.shape[0] < 1:
        raise ValueError("pool_patches requires at least 1 patch row")
    return patches.mean(axis=0).astype(np.float32, copy=False)
