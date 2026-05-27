"""Typed state + Pydantic models for the LangGraph generation loop.

Design §5: ``AgentState`` is a ``TypedDict(total=False)`` so the graph can
build up the state field-by-field as nodes execute. Every non-primitive
value must be a Pydantic ``BaseModel`` (no raw dataclasses) so
``serialize_state_snapshot`` in ``generate/trace.py`` can ``.model_dump()``
them into ``trace_spans.input/output`` jsonb.

Candidate models (``GeneratorCandidate``, ``JudgeCandidate``,
``CheapExtractionCandidate``) are denormalised from
``eval/config/model_candidates.yaml`` so ``AgentState`` doesn't carry a
yaml reference around — the candidate is fully resolved by ``answer()``
before the graph starts.

``RetrievedChunk`` mirrors the fields the LangGraph nodes need from
``retrieve.types.FusedResult`` / ``RerankedResult``. ``from_fused`` /
``from_reranked`` are constructors so the conversion is one-shot and
typed.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from generate.parser import AnswerClaim, GenerationOutput


class GeneratorCandidate(BaseModel):
    """Resolved generator candidate from model_candidates.yaml.

    All fields denormalised from the yaml so ``AgentState`` doesn't carry
    a yaml reference. ``candidate_id`` is the yaml ``id``; the rest are
    looked up via ``_resolve_provider_model_id`` /
    ``_resolve_candidate_family`` in ``eval.runners.providers``."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    candidate_id: str
    provider_model_id: str
    family: str
    provider: str


class JudgeCandidate(GeneratorCandidate):
    """Same shape as ``GeneratorCandidate``; distinct type so test (f)
    [CrossFamilyViolationError] can express the rule with a type-checked
    invariant — judge_candidate.family must differ from
    generator_candidate.family."""


class CheapExtractionCandidate(GeneratorCandidate):
    """For the optional LLM Plan path. Distinct type so the planner can
    refuse to accept a non-cheap-extraction candidate at the type level."""


class RetrievedChunk(BaseModel):
    """The chunk shape ``AgentState`` carries through the graph.

    Mirrors the fields ``retrieve.types.FusedResult`` /
    ``RerankedResult`` carry. Built via ``from_fused`` / ``from_reranked``
    class methods so the conversion is one-shot.

    ``rerank_score`` is ``None`` for chunks that have only been through
    RRF (pre-rerank); ``rrf_score`` is ``None`` for chunks built outside
    the RRF path (rare; the test harness uses this for stub chunks).
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    chunk_id: int
    video_id: str
    start_sec: float
    end_sec: float
    text: str
    score: float
    rank: int
    channel_ranks: dict[str, int] = Field(default_factory=dict)
    rerank_score: float | None = None
    rrf_score: float | None = None

    @classmethod
    def from_fused(cls, fr: Any) -> RetrievedChunk:
        return cls(
            chunk_id=fr.chunk_id,
            video_id=fr.video_id,
            start_sec=fr.start_sec,
            end_sec=fr.end_sec,
            text=fr.text,
            score=float(fr.score),
            rank=int(fr.rank),
            channel_ranks=dict(fr.channel_ranks),
            rerank_score=None,
            rrf_score=float(fr.score),
        )

    @classmethod
    def from_reranked(cls, rr: Any) -> RetrievedChunk:
        return cls(
            chunk_id=rr.chunk_id,
            video_id=rr.video_id,
            start_sec=rr.start_sec,
            end_sec=rr.end_sec,
            text=rr.text,
            score=float(rr.score),
            rank=int(rr.rank),
            channel_ranks=dict(rr.channel_ranks),
            rerank_score=float(rr.rerank_score),
            rrf_score=float(rr.rrf_score),
        )


class ValidatedCitation(BaseModel):
    """Citation that passed Cite-node validation (≥50% overlap with a
    chunk in ``state['reranked']``). Carries the model-emitted span plus
    the matched ``chunk_id`` for phase-4 trace-viewer joins."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    video_id: str
    start_sec: float
    end_sec: float
    answer_claim_index: int
    matched_chunk_id: int


class InvalidCitation(BaseModel):
    """Citation that FAILED validation (no reranked chunk with ≥50%
    overlap on the cited ``video_id``). ``matched_chunk_id`` is omitted
    intentionally — there isn't one. ``reason`` is a short string for
    the trace viewer + phase-2 CitationAccuracy metric."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    video_id: str
    start_sec: float
    end_sec: float
    answer_claim_index: int
    reason: str


class AnswerResult(BaseModel):
    """Cite-node terminal output. Returned by ``answer()`` to callers and
    persisted into ``traces.final_citations`` (jsonb) for the trace
    viewer."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    trace_id: str
    query: str
    corpus_id: str
    answer: str | None
    claims: list[AnswerClaim] = Field(default_factory=list)
    valid_citations: list[ValidatedCitation] = Field(default_factory=list)
    invalid_citations: list[InvalidCitation] = Field(default_factory=list)
    abstain: bool = False
    parse_ok: bool = True
    iterations: int = 0
    latency_ms: int = 0


class AgentState(TypedDict, total=False):
    """LangGraph state. ``total=False`` because nodes populate fields
    incrementally. See design §5 for the per-node contract."""

    # Inputs (immutable through the run; set by answer() before graph.invoke)
    query: str
    corpus_id: str
    generator_candidate: GeneratorCandidate
    judge_candidate: JudgeCandidate
    planner_candidate: CheapExtractionCandidate | None
    verify_confidence_threshold: float

    # Plan output
    question_type: Literal["single_clip", "synthesis"]
    sub_queries: list[str]

    # Retrieve output (mutable across iterations)
    retrieved: list[RetrievedChunk]
    iteration: int
    refined_query: str | None

    # Rerank output
    reranked: list[RetrievedChunk]

    # Verify output
    verify_confidence: float
    verify_reason: str

    # Generate output
    raw_answer: str
    parsed: GenerationOutput | None
    parse_ok: bool

    # Cite output (terminal)
    final: AnswerResult

    # Tracing
    trace_id: str
