"""Plan node — classifies the query and (optionally) decomposes it.

Two paths, picked by whether the caller supplied a ``planner_candidate``
on ``AgentState``:

1. ``planner_candidate is None`` — DETERMINISTIC PATH (default
   pre-bakeoff). No LLM call. Heuristic:

   * ``question_type = 'synthesis'`` if the lowered query matches a
     synthesis cue regex OR contains 2+ distinct speaker first-names
     from ``eval/corpora/<corpus_id>/talks.yaml``. Else
     ``question_type = 'single_clip'``.
   * ``sub_queries = [state['query']]``. Synthesis decomposition into
     multiple sub-queries requires an LLM; deterministic mode degrades
     to a single retrieval pass over the original query.

   Cheap, reproducible, free, and honest about its limits.

2. ``planner_candidate is not None`` — LLM PATH. Calls
   ``providers.chat_completion(planner_candidate.candidate_id,
   component='cheap_extraction', ...)`` with a structured prompt that
   asks for strict JSON ``{question_type, sub_queries: [...]}``. On
   parse failure, degrades to the deterministic path (and the trace
   span carries the degradation reason).

No silent LLM default. Caller must explicitly pass a planner_candidate
via ``PLANNER=...`` (see design §5 + §7) or accept the deterministic
path.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import yaml

from generate.state import AgentState

# Synthesis cue regex per design §5. ``\b`` anchors keep us off
# substring false-positives like "compared" → "comp" or "and" inside
# "android". The "and X (say|argue)" alternation catches sloppier
# synthesis phrasings.
_SYNTHESIS_CUE_RE = re.compile(
    r"\b(compare|contrast|both|and\s+\w+\s+(say|argue|claim|argues|says|claims))\b",
    re.IGNORECASE,
)

# Repo-root-relative — Path(__file__) is .../lensgraph/generate/nodes/plan.py.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CORPORA_DIR = _REPO_ROOT / "eval" / "corpora"


@lru_cache(maxsize=8)
def _speakers_for_corpus(corpus_id: str) -> tuple[tuple[str, ...], ...]:
    """Return ``(speaker_tokens, ...)`` — one tuple of lowercase name
    tokens per speaker in ``eval/corpora/<corpus_id>/talks.yaml``.

    Cached per process. Missing yaml or read failure → empty tuple so
    the synthesis-by-speakers heuristic degrades silently to the
    regex-only path.

    Parenthetical affiliations (``"Shubam Sabu (Google Cloud)"``) are
    stripped. Tokens shorter than 3 chars or in a small stop-word set
    are dropped — keeps false positives down on short queries like
    "the talk".
    """
    yaml_path = _CORPORA_DIR / corpus_id / "talks.yaml"
    try:
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, yaml.YAMLError):
        return ()
    if not isinstance(raw, list):
        return ()

    stop = {"the", "of", "and", "for", "with", "team", "labs", "co"}
    speakers: list[tuple[str, ...]] = []
    for entry in raw:
        speaker = (entry or {}).get("speaker", "") if isinstance(entry, dict) else ""
        if not isinstance(speaker, str):
            continue
        base = re.sub(r"\s*\([^)]*\)\s*$", "", speaker).strip()
        toks = tuple(
            tok.lower()
            for tok in re.split(r"[\s\-]+", base)
            if len(tok) >= 3 and tok.lower() not in stop
        )
        if toks:
            speakers.append(toks)
    return tuple(speakers)


def _count_speaker_mentions(query: str, corpus_id: str) -> int:
    """Count distinct speakers from ``talks.yaml`` whose name tokens
    appear in the query. Each speaker counts at most once — multiple
    token matches against the same speaker don't double-count."""
    speakers = _speakers_for_corpus(corpus_id)
    if not speakers:
        return 0
    q = query.lower()
    hits = 0
    for toks in speakers:
        if any(t in q for t in toks):
            hits += 1
    return hits


def _deterministic_classify(query: str, corpus_id: str) -> str:
    """Return ``'synthesis'`` if the query matches a synthesis cue or
    mentions 2+ speakers, else ``'single_clip'``."""
    if _SYNTHESIS_CUE_RE.search(query):
        return "synthesis"
    if _count_speaker_mentions(query, corpus_id) >= 2:
        return "synthesis"
    return "single_clip"


def _llm_plan(state: AgentState) -> tuple[str | None, list[str] | None, str | None]:
    """Call the cheap-extraction planner. Returns
    ``(question_type, sub_queries, error)``. On any failure
    ``(None, None, error_str)`` is returned and the caller degrades."""
    candidate = state.get("planner_candidate")
    if candidate is None:
        return None, None, "no planner_candidate"

    # Import here so the deterministic path stays free of provider deps.
    from eval.runners import providers

    prompt = (
        "You are a query planner for a video RAG system. Classify the user query "
        "into either 'single_clip' (answer in one talk) or 'synthesis' (requires "
        "comparing/combining 2+ talks). Then emit sub-queries: one for "
        "single_clip, two-or-more for synthesis.\n\n"
        f"User query: {state['query']!r}\n\n"
        'Reply with STRICT JSON, no prose: {"question_type": "single_clip"|"synthesis", '
        '"sub_queries": ["...", ...]}'
    )

    try:
        resp = providers.chat_completion(
            candidate_id=candidate.candidate_id,
            messages=[{"role": "user", "content": prompt}],
            component="cheap_extraction",
            max_retries=0,
            max_tokens=256,
        )
    except providers.ProviderError as e:
        return None, None, f"provider: {e}"

    raw = resp.raw_text or ""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as e:
        return None, None, f"json: {e}"
    qt = parsed.get("question_type")
    sq = parsed.get("sub_queries")
    if qt not in ("single_clip", "synthesis"):
        return None, None, f"bad question_type: {qt!r}"
    if not isinstance(sq, list) or not all(isinstance(s, str) and s for s in sq):
        return None, None, "bad sub_queries"
    return qt, sq, None


def plan(state: AgentState) -> AgentState:
    """Classify + (optionally) decompose. See module docstring."""
    query = state["query"]
    corpus_id = state.get("corpus_id", "ai_engineering_v0")

    # LLM path — opt-in, with graceful degradation.
    if state.get("planner_candidate") is not None:
        qt, sq, _err = _llm_plan(state)
        if qt is not None and sq is not None:
            state["question_type"] = qt  # type: ignore[typeddict-item]
            state["sub_queries"] = sq
            return state
        # fall through to deterministic

    # Deterministic path.
    state["question_type"] = _deterministic_classify(query, corpus_id)  # type: ignore[typeddict-item]
    state["sub_queries"] = [query]
    return state
