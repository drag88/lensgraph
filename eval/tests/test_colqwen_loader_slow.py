"""Slow regression guard for the ColQwen2.5 adapter load.

These tests defend the fix in ``embed/colqwen._load_adapted_colqwen`` against the
two load bugs that silently turned visual retrieval into noise on this stack
(see ``eval/reports/2026-05-30_visual_rerank_probe/methodology.mdx`` and the
``loader_fix.mdx`` follow-up):

1. The ``colqwen2.5-v0.2`` LoRA adapter ships ``model.layers.*`` keys, but the
   live transformers exposes the text decoder as ``language_model.layers.*``, so
   PEFT left every LoRA layer at its random init.
2. ``tie_word_embeddings: True`` + the same rename left ``embed_tokens`` at its
   random init too — the dominant source of nondeterminism.

Both produced an encoder that was deterministic WITHIN a process but different on
every fresh load. So an in-process "encode twice, assert equal" check is NOT a
valid guard — it passes on the broken code. These tests assert the properties
that actually fail when the bug is present: cross-process determinism and strict
load integrity (zero missing / zero unexpected LoRA keys).

Real ColQwen2.5 weights load here (~30s cold; the cross-process test loads twice,
sequentially — never two ML processes at once), so the module is ``slow``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

slow = pytest.mark.slow

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PROBE_QUERY = "fixed cross-process determinism probe for ColQwen2.5 loader"

# Encode a fixed query and write the vector to argv[1]. Run as a separate process
# so its model load is fully independent of the in-test load.
_SUBPROCESS_SRC = (
    "import sys, numpy as np\n"
    "from embed.colqwen import encode_text_query\n"
    f"np.save(sys.argv[1], encode_text_query({_PROBE_QUERY!r}))\n"
)


@slow
def test_encoder_is_deterministic_across_processes(tmp_path):
    """The same query must encode identically in two independent processes.

    This is the guard the bug actually trips: with the broken load the LoRA and
    embed_tokens are randomly re-initialized per process, so two process loads
    disagree (observed cosine ~0.84). The subprocess runs first and exits —
    freeing its model — before the in-process encode, so only one ML model is
    ever resident at a time.
    """
    out = tmp_path / "subproc_qvec.npy"
    subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_SRC, str(out)],
        cwd=_REPO_ROOT,
        check=True,
        timeout=900,
    )
    other = np.load(out)

    from embed import colqwen

    colqwen._model.cache_clear()
    here = colqwen.encode_text_query(_PROBE_QUERY)

    assert here.shape == other.shape
    cosine = float(here @ other / (np.linalg.norm(here) * np.linalg.norm(other)))
    assert cosine > 0.9999, f"cross-process cosine {cosine:.6f} — encoder is non-deterministic"


@slow
def test_adapter_loads_with_zero_missing_or_unexpected_lora_keys():
    """``_load_adapted_colqwen`` raises on any LoRA key mismatch (its integrity
    gate). Reaching a loaded model means the remap covered every adapter key
    exactly, and a fully-LOADED adapter has every lora_B nonzero (standard LoRA
    inits B to zero, so an unloaded/random adapter would have differed)."""
    from embed import colqwen

    colqwen._model.cache_clear()
    model, _processor, _device = colqwen._model()
    lora_b = [p for n, p in model.named_parameters() if "lora_B" in n]
    assert lora_b, "no LoRA layers found — adapter did not apply"
    assert all(float(p.abs().max()) > 0 for p in lora_b), (
        "some lora_B is all-zero — adapter weights did not land (the remap regressed)"
    )


@slow
def test_token_embeddings_match_base_checkpoint():
    """The live embedding table must equal the base checkpoint's
    ``model.embed_tokens.weight``. A random re-init (the tie+rename bug) would
    not match — this is the guard for the embed_tokens half of the fix."""
    import json

    import safetensors.torch as st
    import torch
    from huggingface_hub import snapshot_download

    from embed import colqwen

    colqwen._model.cache_clear()
    model, _processor, _device = colqwen._model()

    base_dir = Path(snapshot_download("vidore/colqwen2.5-base"))
    weight_map = json.loads((base_dir / "model.safetensors.index.json").read_text())["weight_map"]
    key = "model.embed_tokens.weight"
    expected = st.load_file(str(base_dir / weight_map[key]))[key]
    live = model.get_input_embeddings().weight.detach().to("cpu", torch.float32)
    assert live.shape == expected.shape
    assert torch.allclose(live, expected.to(torch.float32), atol=1e-3), (
        "live embed_tokens != base checkpoint — the embed_tokens load regressed"
    )
