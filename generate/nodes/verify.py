"""Verify node — cross-family judge over top-3 reranked chunks.

The judge scores whether the top-3 reranked chunks plausibly contain
the answer to ``state['query']`` and proposes a refined query when it
doesn't. Confidence is a float in ``[0, 1]``.

If ``confidence < state['verify_confidence_threshold']`` AND
``state['iteration'] < 2``, the node increments ``iteration``, sets
``refined_query``, and the router routes back to Retrieve. Otherwise
the node clears ``refined_query`` (signal to the router "continue to
Generate").

On judge-side parse failure we degrade to confidence=0.5 and a reason
of ``"parse_failed: <detail>"``. A broken judge response must not block
the loop — better to continue to Generate with the current candidates
than to error the trace out.

The judge's family != generator's family invariant is enforced earlier
(in ``generate.api._resolve_candidate`` + the explicit cross-family
check in ``answer()``). This node assumes that invariant holds.
"""

from __future__ import annotations

import json

from eval.runners import providers
from generate.state import AgentState

_TOP_K_FOR_JUDGE = 3
_JUDGE_MAX_TOKENS = 256


def _format_chunks(chunks: list) -> str:
    """Inline the top-3 chunks with ``[video_id @ s.s-e.e]`` labels."""
    parts: list[str] = []
    for i, c in enumerate(chunks):
        parts.append(
            f"[{i + 1}] [{c.video_id} @ {c.start_sec:.1f}-{c.end_sec:.1f}]\n"
            f"{c.text}"
        )
    return "\n\n".join(parts)


def _judge_prompt(query: str, top3: list) -> list[dict]:
    """Build the judge prompt. Strict-JSON instruction; explicit
    confidence range; refined_query is OPTIONAL — empty string signals
    "no refinement needed"."""
    sys = (
        "You are a retrieval judge for a video RAG system. Decide whether the "
        "provided chunks plausibly contain enough evidence to answer the user "
        "query. Emit STRICT JSON only, no prose, no code fences:\n"
        '  {"confidence": <float in [0, 1]>, "reason": "<short>", '
        '"refined_query": "<query rewrite or empty string>"}\n'
        "confidence > 0.6 means 'yes, generate'; <= 0.6 means 'no, refine'."
    )
    user = (
        f"User query: {query!r}\n\nTop chunks:\n{_format_chunks(top3)}\n\n"
        "Score now."
    )
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]


def _parse_judge(raw: str) -> tuple[float, str, str | None]:
    """Return ``(confidence, reason, refined_query_or_None)``.

    On any parse failure: ``(0.5, "parse_failed: <detail>", None)``.
    """
    try:
        obj = json.loads(raw or "")
    except (json.JSONDecodeError, ValueError) as e:
        return 0.5, f"parse_failed: {e}", None
    if not isinstance(obj, dict):
        return 0.5, f"parse_failed: not an object ({type(obj).__name__})", None
    try:
        conf = float(obj.get("confidence", 0.5))
    except (TypeError, ValueError):
        return 0.5, "parse_failed: bad confidence", None
    conf = max(0.0, min(1.0, conf))  # clamp
    reason = obj.get("reason", "") if isinstance(obj.get("reason", ""), str) else ""
    refined = obj.get("refined_query")
    if not isinstance(refined, str) or not refined.strip():
        refined_query = None
    else:
        refined_query = refined.strip()
    return conf, reason, refined_query


def verify(state: AgentState, *, conn) -> AgentState:  # noqa: ARG001 (conn reserved)
    """Score top-3 reranked chunks; set verify_confidence + reason; on
    low confidence + iteration < 2, set refined_query and bump
    iteration. ``conn`` is unused (judge is HTTP) but kept on the
    signature so the graph wrapper can pass it uniformly."""
    judge = state["judge_candidate"]
    reranked = state.get("reranked") or []
    top3 = reranked[:_TOP_K_FOR_JUDGE]

    try:
        resp = providers.chat_completion(
            candidate_id=judge.candidate_id,
            messages=_judge_prompt(state["query"], top3),
            component="judge",
            max_retries=1,
            max_tokens=_JUDGE_MAX_TOKENS,
        )
        raw = resp.raw_text
    except providers.ProviderError as e:
        # Network failure → safe fallback so the graph doesn't dead-end.
        state["verify_confidence"] = 0.5
        state["verify_reason"] = f"judge_unavailable: {e}"
        state["refined_query"] = None
        return state

    conf, reason, refined = _parse_judge(raw)
    threshold = state["verify_confidence_threshold"]
    current_iter = state.get("iteration", 0)

    state["verify_confidence"] = conf
    state["verify_reason"] = reason

    if conf < threshold and current_iter < 2:
        # Engage loop: bump iteration; set refined_query (falling back to
        # the original query when the judge didn't propose a rewrite).
        state["refined_query"] = refined or state["query"]
        state["iteration"] = current_iter + 1
    else:
        state["refined_query"] = None
    return state
