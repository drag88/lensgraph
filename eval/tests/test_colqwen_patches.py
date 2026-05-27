"""Slow ColQwen2.5 patch-encoder tests.

Phase-1 step 24 RED gate for ``embed/colqwen.py``'s patch-level encoders.
These tests load the real ColQwen2.5 weights (already cached from
slice-0 / slice-1 work), so each module run is ~30s cold model load.

Coverage:
  - ``encode_image_patches`` returns (N, P, 128) float32 with P > 0.
  - ``pool_patches(encode_image_patches([img])[0])`` matches
    ``encode_image_pooled([img])[0]`` to within float tolerance — the
    one-encoder-path refactor guarantee. Without this, the HNSW prefilter
    (pooled) and MaxSim refine (patches) could silently diverge.
  - ``encode_text_query_patches`` returns (T_q, 128) for a real query
    AND short-circuits to (0, 128) on empty / whitespace without
    loading the model. The lazy-cache-clear + monkeypatched _model
    pattern mirrors test_colqwen_lazy.

The pool_patches numerical check is fast (no model load) and lives at
module scope so it runs even when ColQwen weights aren't available.
"""

from __future__ import annotations

import numpy as np
import pytest

slow = pytest.mark.slow


# -- fast helpers (no model load) ----------------------------------------


def test_pool_patches_mean_aggregates_along_axis_0():
    """pool_patches is the mean over the patch axis. A (4, 128) array of
    distinct constants must reduce to the elementwise mean across rows."""
    from embed import colqwen

    patches = np.stack(
        [
            np.full((colqwen.POOLED_DIM,), 1.0, dtype=np.float32),
            np.full((colqwen.POOLED_DIM,), 2.0, dtype=np.float32),
            np.full((colqwen.POOLED_DIM,), 3.0, dtype=np.float32),
            np.full((colqwen.POOLED_DIM,), 4.0, dtype=np.float32),
        ]
    )
    pooled = colqwen.pool_patches(patches)
    assert pooled.shape == (colqwen.POOLED_DIM,)
    assert pooled.dtype == np.float32
    np.testing.assert_allclose(pooled, np.full((colqwen.POOLED_DIM,), 2.5, dtype=np.float32))


def test_pool_patches_rejects_wrong_dim():
    from embed import colqwen

    with pytest.raises(ValueError, match="pool_patches expects"):
        colqwen.pool_patches(np.zeros((4, colqwen.POOLED_DIM + 1), dtype=np.float32))


def test_pool_patches_rejects_empty():
    from embed import colqwen

    with pytest.raises(ValueError, match="at least 1 patch row"):
        colqwen.pool_patches(np.zeros((0, colqwen.POOLED_DIM), dtype=np.float32))


def test_encode_text_query_patches_empty_does_not_load_model(monkeypatch):
    """Mirrors test_colqwen_lazy's empty-query short-circuit guarantee for
    the new patch-level entry point."""
    from embed import colqwen

    colqwen._model.cache_clear()

    def boom():
        raise AssertionError("_model() must not be called for empty query")

    monkeypatch.setattr(colqwen, "_model", boom)

    assert colqwen.encode_text_query_patches("").shape == (0, colqwen.POOLED_DIM)
    assert colqwen.encode_text_query_patches("   \n\t").shape == (0, colqwen.POOLED_DIM)


def test_encode_image_patches_empty_does_not_load_model(monkeypatch):
    """encode_image_patches([]) must short-circuit before _model() is called."""
    from embed import colqwen

    colqwen._model.cache_clear()

    def boom():
        raise AssertionError("_model() must not be called for empty input")

    monkeypatch.setattr(colqwen, "_model", boom)

    out = colqwen.encode_image_patches([])
    assert out.shape == (0, 0, colqwen.POOLED_DIM)


def test_encode_image_pooled_averages_only_real_patches_in_mixed_batch(monkeypatch):
    """When a batch has mixed patch counts, encode_image_pooled must average
    ONLY the real (unpadded) rows per image — not the zero-padded rows.

    Catches the v0 regression where ``patches.mean(axis=1)`` summed zeros
    from the padded rows into the denominator, shrinking shorter images'
    pooled vectors toward zero (broken cosine similarity for the HNSW
    prefilter).

    Setup: monkeypatched ``_encode_images_with_true_counts`` returns a
    (2, 4, 128) padded matrix and true_counts=[2, 4]. Image 0 has rows
    [1, 3, 0, 0] (values broadcast to all 128 dims); image 1 has rows
    [5, 5, 5, 5]. Expected pooled[0] = mean([1, 3]) = 2 (NOT
    mean([1, 3, 0, 0]) = 1); expected pooled[1] = mean([5, 5, 5, 5]) = 5.
    """
    from embed import colqwen

    dim = colqwen.POOLED_DIM
    padded = np.zeros((2, 4, dim), dtype=np.float32)
    padded[0, 0, :] = 1.0
    padded[0, 1, :] = 3.0
    # padded[0, 2:, :] remain 0 (padding rows for the shorter image)
    padded[1, :, :] = 5.0
    true_counts = [2, 4]

    monkeypatch.setattr(
        colqwen,
        "_encode_images_with_true_counts",
        lambda images: (padded, true_counts),
    )

    # Two sentinel "PIL images" — they're never inspected (the encoder is
    # stubbed at the model call) but the count must match true_counts.
    pooled = colqwen.encode_image_pooled([object(), object()])

    assert pooled.shape == (2, dim)
    np.testing.assert_allclose(
        pooled[0], np.full((dim,), 2.0, dtype=np.float32),
        err_msg=(
            "pooled[0] must average only the first 2 real rows; including "
            "zero-padded rows would have given the all-1.0 vector "
            "((1+3+0+0)/4)."
        ),
    )
    np.testing.assert_allclose(
        pooled[1], np.full((dim,), 5.0, dtype=np.float32),
        err_msg="pooled[1] (full real patches) must equal the single-value mean 5.0",
    )


# -- slow model-load tests ------------------------------------------------


def _make_pil_image(color: tuple[int, int, int], size: tuple[int, int] = (224, 224)):
    """Synthesize a uniformly-colored PIL image — enough signal for ColQwen
    to emit deterministic patch counts; we don't assert on patch *content*."""
    from PIL import Image

    return Image.new("RGB", size, color)


@slow
def test_encode_image_patches_returns_3d_array_with_pooled_dim_128():
    """Single image → (1, P, 128). Two-image batch → (2, P, 128) for some P > 0.
    Float32 to match the column type ``vector(128)`` in 0003_frames.sql."""
    from embed import colqwen

    img_a = _make_pil_image((255, 0, 0))
    out_one = colqwen.encode_image_patches([img_a])
    assert out_one.ndim == 3
    assert out_one.shape[0] == 1
    assert out_one.shape[1] > 0
    assert out_one.shape[2] == colqwen.POOLED_DIM
    assert out_one.dtype == np.float32

    img_b = _make_pil_image((0, 255, 0))
    out_two = colqwen.encode_image_patches([img_a, img_b])
    assert out_two.ndim == 3
    assert out_two.shape[0] == 2
    assert out_two.shape[1] > 0
    assert out_two.shape[2] == colqwen.POOLED_DIM
    assert out_two.dtype == np.float32


@slow
def test_encode_image_patches_matches_pooled_via_pool_patches():
    """The refactor guarantee: encode_image_pooled([img])[0] must equal
    pool_patches(encode_image_patches([img])[0]) within float tolerance.

    Without this, ``frames.pooled_embedding`` (HNSW prefilter input) and
    the mean of ``frame_patches`` (the MaxSim refine input) could
    silently diverge — the visual channel would prefilter on a vector
    that does not summarize the patches it actually scores."""
    from embed import colqwen

    img = _make_pil_image((128, 128, 128))
    patches = colqwen.encode_image_patches([img])[0]
    via_helper = colqwen.pool_patches(patches)
    via_pooled = colqwen.encode_image_pooled([img])[0]
    np.testing.assert_allclose(via_helper, via_pooled, rtol=1e-5, atol=1e-5)


@slow
def test_encode_text_query_patches_returns_2d_array_with_pooled_dim():
    """A real query yields (T_q, 128) for T_q > 0 — the matrix MaxSim
    consumes against frame_patches."""
    from embed import colqwen

    out = colqwen.encode_text_query_patches("how does the encoder attend to slides")
    assert out.ndim == 2
    assert out.shape[0] > 0
    assert out.shape[1] == colqwen.POOLED_DIM
    assert out.dtype == np.float32
