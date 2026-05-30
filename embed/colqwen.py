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


# LoRA projection-layer leaf names the colqwen2.5-v0.2 adapter targets. Used to
# locate the live modules and remap the adapter's stale keys onto them.
_LORA_TARGET_LEAVES = frozenset(
    {"down_proj", "gate_proj", "up_proj", "k_proj", "q_proj", "v_proj", "o_proj", "custom_text_proj"}
)


def _suffix_from_layers(path: str) -> str:
    """Return the stable module-path tail used to align an adapter key with a
    live module: everything from ``layers.<N>.`` onward, or the leaf name for
    top-level modules like ``custom_text_proj``."""
    import re

    m = re.search(r"layers\.\d+\..*$", path)
    return m.group(0) if m else path.split(".")[-1]


def _load_adapted_colqwen(adapter_id: str, device: str) -> Any:
    """Load ColQwen2.5 base + the ``colqwen2.5-v0.2`` LoRA adapter, remapping the
    adapter's stale parameter keys onto the live module names.

    Why this is not a plain ``ColQwen2_5.from_pretrained(adapter_id)``: that path
    is silently broken on this stack. The adapter was saved with text-decoder
    keys ``base_model.model.model.layers.*``, but under the installed
    transformers the live Qwen2.5-VL text decoder is ``language_model.layers.*``.
    PEFT injects LoRA into the live (``language_model.*``) modules but then fails
    to match the saved (``model.*``) weights onto them, so every LoRA layer is
    left at its random init. The result is a non-deterministic encoder: the same
    query encodes differently on each process load (cosine ~0.84 across loads),
    and queries no longer match the frame patches the encoder produced at ingest.
    See ``eval/reports/2026-05-30_visual_rerank_probe/methodology.mdx``.

    The fix remaps the saved adapter keys onto the live module paths by matching
    the ``layers.<N>.<...>`` suffix, then asserts the remapped key set is exactly
    what PEFT expects for this model — zero missing, zero unexpected. A future
    transformers/colpali bump that renames modules again will fail this assertion
    loudly instead of silently degrading retrieval to noise.
    """
    import json
    import tempfile

    import safetensors.torch as st
    import torch
    from colpali_engine.models import ColQwen2_5
    from huggingface_hub import hf_hub_download, snapshot_download
    from peft import PeftModel
    from peft.utils import get_peft_model_state_dict

    dtype = torch.float16 if device == "mps" else torch.float32
    cfg = json.loads(Path(hf_hub_download(adapter_id, "adapter_config.json")).read_text())
    base_id = cfg["base_model_name_or_path"]
    base = ColQwen2_5.from_pretrained(base_id, torch_dtype=dtype, device_map=device).eval()

    # Second half of the same rename bug: the base checkpoint stores the token
    # embeddings as ``model.embed_tokens.weight`` (old layout) and sets
    # ``tie_word_embeddings: True``. Under the installed transformers the live
    # module is ``language_model.embed_tokens`` and the tie logic leaves it at
    # its random init — so even with the LoRA fixed, every process gets a
    # different embedding table (the one parameter that differs across loads).
    # Load the real weights from the checkpoint explicitly.
    base_dir = Path(snapshot_download(base_id))
    weight_map = json.loads((base_dir / "model.safetensors.index.json").read_text())["weight_map"]
    embed_key = "model.embed_tokens.weight"
    embed_shard = st.load_file(str(base_dir / weight_map[embed_key]))[embed_key]
    embed_module = base.get_input_embeddings()
    with torch.no_grad():
        embed_module.weight.copy_(embed_shard.to(device=device, dtype=embed_module.weight.dtype))

    # Map each live target module to its layers-suffix so adapter keys can be
    # remapped onto whatever the current transformers calls the text decoder.
    live_by_suffix: dict[str, str] = {}
    for name, _module in base.named_modules():
        if name and name.rsplit(".", 1)[-1] in _LORA_TARGET_LEAVES and "visual" not in name:
            live_by_suffix.setdefault(_suffix_from_layers(name), name)

    adapter_dir = Path(snapshot_download(adapter_id))
    saved = st.load_file(str(adapter_dir / "adapter_model.safetensors"))
    remapped: dict[str, Any] = {}
    for key, value in saved.items():
        body = key.removeprefix("base_model.model.")
        for suf in (".lora_A.weight", ".lora_B.weight"):
            if body.endswith(suf):
                old_path, lora_suffix = body[: -len(suf)], suf
                break
        else:
            raise RuntimeError(f"unexpected adapter key shape: {key}")
        live_path = live_by_suffix.get(_suffix_from_layers(old_path))
        if live_path is None:
            raise RuntimeError(
                f"adapter key {key!r} has no live target module — the model layout "
                "changed; update the ColQwen adapter remap in embed/colqwen.py"
            )
        remapped[f"base_model.model.{live_path}{lora_suffix}"] = value

    with tempfile.TemporaryDirectory(prefix="colqwen25_v02_remap_") as tmp:
        tmp_dir = Path(tmp)
        tmp_dir.joinpath("adapter_config.json").write_text(
            (adapter_dir / "adapter_config.json").read_text()
        )
        st.save_file(remapped, str(tmp_dir / "adapter_model.safetensors"))
        model = PeftModel.from_pretrained(base, str(tmp_dir), torch_dtype=dtype)

    # Load-integrity gate: the keys we supplied must be exactly the keys PEFT
    # expects for this model. Any mismatch means the adapter did not fully apply.
    expected = set(get_peft_model_state_dict(model).keys())
    supplied = set(remapped.keys())
    missing = expected - supplied
    unexpected = supplied - expected
    if missing or unexpected:
        raise RuntimeError(
            f"ColQwen adapter load integrity failed: {len(missing)} missing, "
            f"{len(unexpected)} unexpected LoRA keys (missing sample: {sorted(missing)[:2]}, "
            f"unexpected sample: {sorted(unexpected)[:2]})"
        )
    return model.eval()


@lru_cache(maxsize=1)
def _model() -> Any:
    """Load ColQwen2.5 + its processor. Heavy import deferred until first
    call. Model id resolved from model_candidates.yaml — see _resolve_model_id.

    The model load goes through ``_load_adapted_colqwen`` rather than a plain
    ``from_pretrained`` because the v0.2 adapter's keys must be remapped onto the
    current module layout — see that function for the full rationale."""
    import torch
    from colpali_engine.models import ColQwen2_5_Processor

    model_id = _resolve_model_id()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = _load_adapted_colqwen(model_id, device)
    processor = ColQwen2_5_Processor.from_pretrained(model_id)
    return model, processor, device


def _encode_images_with_true_counts(images: list) -> tuple[np.ndarray, list[int]]:
    """Internal: run the ColQwen image encoder once and return
    (padded_patches, true_counts).

    ``padded_patches`` is the (N, max_P, POOLED_DIM) float32 tensor from
    the model. ``true_counts[i]`` is the count of REAL patch rows for
    image ``i`` (rows ``true_counts[i] .. max_P`` are zero-padding).

    Shared between ``encode_image_patches`` (returns the padded tensor
    as-is) and ``encode_image_pooled`` (averages only the real rows per
    image) so both consume identical model output without a second model
    call. The attention_mask from the processor's batch dict drives the
    true_counts list; if the processor stops emitting a mask, the true
    counts fall back to max_P (i.e. every row is treated as real — the
    pre-mask v0 behavior).
    """
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

    max_p = patches_np.shape[1]
    mask = batch.get("attention_mask")
    if mask is not None:
        mask_np = mask.cpu().numpy().astype(bool)
        true_counts = [int(c) for c in mask_np.sum(axis=1).tolist()]
    else:
        true_counts = [max_p] * patches_np.shape[0]

    # Zero out padding rows so downstream MaxSim (max-over-rows) stays
    # correctness-preserving even if a caller forgets the true_counts.
    if any(c < max_p for c in true_counts):
        zeroed = np.zeros_like(patches_np)
        for i, n in enumerate(true_counts):
            zeroed[i, :n, :] = patches_np[i, :n, :]
        patches_np = zeroed

    return patches_np, true_counts


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

    patches, _true_counts = _encode_images_with_true_counts(images)
    return patches


def encode_image_pooled(images: list) -> np.ndarray:
    """Encode a batch of PIL images. Returns (N, POOLED_DIM) float32 —
    one mean-pooled vector per image, averaging ONLY real patch rows.

    Empty input returns shape (0, POOLED_DIM) without loading the model.

    Implementation: pulls the padded patches AND the per-image true
    patch counts from the shared encoder, then averages each image's
    first ``true_counts[i]`` rows. A naive ``patches.mean(axis=1)``
    would include zero-padding rows from shorter images in a mixed
    batch and silently shrink their pooled vectors toward zero — wrong
    cosine-similarity behavior for the HNSW prefilter. The
    embed_frames_handler encodes one image at a time so it's not
    affected, but ad-hoc multi-image batching (callers staging the full
    corpus offline) must produce correct pooled vectors.
    """
    if not images:
        return np.zeros((0, POOLED_DIM), dtype=np.float32)

    patches, true_counts = _encode_images_with_true_counts(images)
    pooled = np.empty((patches.shape[0], POOLED_DIM), dtype=np.float32)
    for i, n in enumerate(true_counts):
        if n < 1:
            # Defensive: an image with zero real patches would otherwise
            # produce a NaN mean. Treat as zero pooled vector and let the
            # HNSW NULL-pooled WHERE clause filter it downstream.
            pooled[i] = 0.0
            continue
        pooled[i] = patches[i, :n, :].mean(axis=0)

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
