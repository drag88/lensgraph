"""BGE-reranker-v2-m3 cross-encoder over RRF-fused candidates.

Loads the candidate identified by
``model_candidates.yaml::candidates.reranker.options[0]`` via
``sentence_transformers.CrossEncoder``. The candidate id is never
hardcoded — yaml drift (a flip to a hosted provider, a renamed candidate)
is caught at first load with a clear error instead of silently mid-retrieve.

Why sentence-transformers and not FlagEmbedding's ``FlagReranker``:
FlagEmbedding 1.4.0 still calls ``tokenizer.prepare_for_model``, which was
removed from transformers' tokenizer API in the 5.x line. Our pinned
transformers is 5.9.0, so ``FlagReranker.compute_score`` crashes with
``AttributeError: XLMRobertaTokenizer has no attribute prepare_for_model``
on every call. ``sentence_transformers.CrossEncoder`` loads the same
BGE-reranker-v2-m3 checkpoint via the modern tokenizer API and is the
maintained path for cross-encoder reranking. Same model weights, same
scoring semantics — only the loader changes.

Score is the cross-encoder's normalized 0-1 sigmoid output. We pass
``activation_fn=torch.nn.Identity()`` to ``CrossEncoder.predict()`` so
the default model activation is bypassed (BGE-reranker-v2-m3 is
registered with a Sigmoid activation in its config; predict()'s default
``activation_fn=None`` means "use the model's"). Then we apply
``torch.sigmoid`` exactly once. Without the Identity override the score
would be ``sigmoid(sigmoid(raw_logit))``, which is monotonic but pinned
to the (0.5, ~0.73) sub-range — fine for ordering, broken as a 0-1
confidence contract.

The reranker runs CPU-only on M-series per design §4 "BGE-reranker-v2-m3
local CPU" — a hosted alternative kicks in only if the p95 reranker
minimums in ``model_candidates.yaml::candidates.reranker.minimums`` fail
at phase-2 bakeoff time.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from retrieve.types import FusedResult, RerankedResult

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "eval" / "config" / "model_candidates.yaml"


def _resolve_model_id() -> str:
    """Load model_candidates.yaml, return provider_model_id of
    ``candidates.reranker.options[0]``. Validates provider=='local' so a
    yaml drift to a hosted alternative is caught at boot. The reranker
    candidate list has exactly one option at v0; picking [0] is the
    convention `embed.bge_m3._resolve_model_id` and `embed.colqwen._resolve_model_id`
    mirror.

    Raises:
        KeyError: reranker options missing or empty.
        ValueError: top option's provider is not 'local'.
    """
    import yaml

    raw = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    try:
        options = raw["candidates"]["reranker"]["options"]
    except KeyError as e:
        raise KeyError("candidates.reranker.options missing from model_candidates.yaml") from e
    if not options:
        raise KeyError("candidates.reranker.options is empty in model_candidates.yaml")
    opt = options[0]
    if opt.get("provider") != "local":
        raise ValueError(
            f"reranker provider must be 'local', got {opt.get('provider')!r} "
            f"(candidate id: {opt.get('id')!r})"
        )
    return opt["provider_model_id"]


@lru_cache(maxsize=1)
def _reranker() -> Any:
    """Lazy singleton. CrossEncoder load is heavy (~2.3 GB BGE-reranker
    weights pulled on first call); deferring until first non-empty
    ``rerank()`` keeps fast tests fast.

    Model id resolved from model_candidates.yaml — see ``_resolve_model_id``.
    Device pinned to CPU per design §4; sentence-transformers picks the
    right dtype automatically.
    """
    from sentence_transformers import CrossEncoder

    return CrossEncoder(_resolve_model_id(), device="cpu")


def rerank(
    conn: Any,  # psycopg.Connection, kept Any to avoid a hard import in fast tests
    query: str,
    candidates: list[FusedResult],
    *,
    top_k: int = 8,
) -> list[RerankedResult]:
    """Re-rank a fused candidate list with the BGE-reranker-v2-m3 cross-encoder.

    Empty ``candidates`` short-circuits to ``[]`` WITHOUT loading the
    model — important for the lazy-load smoke test and for callers that
    handle an empty fused list (no candidates ranked any chunk).

    ``conn`` is unused at v0 — the input ``FusedResult`` already carries
    the chunk text. Kept on the signature so a future refactor that fetches
    long-form chunk text on demand (rather than carrying it in memory through
    RRF) does not require a call-site change.

    Algorithm: batch ``(query, candidate.text)`` pairs into
    ``CrossEncoder.predict(activation_fn=Identity())`` → sigmoid the raw
    logits to (0, 1) exactly once → sort descending (tie-break by chunk_id
    for determinism) → take top_k → emit RerankedResult rows with
    1-indexed rank.
    """
    if not candidates:
        return []

    import numpy as np
    import torch

    pairs: list[tuple[str, str]] = [(query, c.text) for c in candidates]
    # activation_fn=Identity() bypasses the model's default Sigmoid so we
    # receive raw logits and apply sigmoid ourselves — otherwise scores
    # would be sigmoid(sigmoid(logit)), pinned to (0.5, ~0.73). Ordering
    # survives the double-sigmoid but the public ``rerank_score`` contract
    # ("0-1 cross-encoder confidence") does not.
    raw_scores = _reranker().predict(
        pairs,
        convert_to_numpy=True,
        activation_fn=torch.nn.Identity(),
    )
    # CrossEncoder returns a 0-d numpy array for one pair, 1-d for many.
    # Normalize both shapes to a 1-d list.
    arr = np.atleast_1d(np.asarray(raw_scores, dtype=np.float32))
    scores_list = torch.sigmoid(torch.from_numpy(arr)).tolist()

    pre_ranked: list[tuple[FusedResult, float]] = list(zip(candidates, scores_list, strict=True))
    # Tie-break by chunk_id for determinism (mirrors retrieve.rrf.fuse).
    pre_ranked.sort(key=lambda x: (-x[1], x[0].chunk_id))
    top = pre_ranked[:top_k]

    return [
        RerankedResult(
            chunk_id=c.chunk_id,
            video_id=c.video_id,
            start_sec=c.start_sec,
            end_sec=c.end_sec,
            text=c.text,
            score=score,
            rank=i + 1,
            channel_ranks=dict(c.channel_ranks),
            rerank_score=score,
            rrf_score=c.score,
        )
        for i, (c, score) in enumerate(top)
    ]
