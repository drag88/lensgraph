"""Slow tests for each LangGraph node in isolation.

Live Postgres is required for retrieve + rerank (they read the corpus).
Verify / generate / cite get monkeypatched providers + stub state so
no real HTTP is made.

Each node test follows the pattern:
  - Build a minimal AgentState that satisfies the node's inputs.
  - Optionally monkeypatch providers.chat_completion.
  - Call the node directly (not via the graph).
  - Assert the state transition the node is responsible for.
"""

from __future__ import annotations

import json

import psycopg
import pytest

from db.conn import dsn as resolve_dsn
from generate.nodes import cite as cite_mod
from generate.nodes import generate as generate_mod
from generate.nodes import plan as plan_mod
from generate.nodes import rerank as rerank_mod
from generate.nodes import retrieve as retrieve_mod
from generate.nodes import verify as verify_mod
from generate.parser import GenerationOutput
from generate.state import (
    GeneratorCandidate,
    JudgeCandidate,
    RetrievedChunk,
)

pytestmark = pytest.mark.slow

# The corpus already in the dev DB (handoff says 212 chunks across 3 talks).
CORPUS_ID = "ai_engineering_v0"
# Tengyu's canonical exit-gate query (handoff: "tengyu-rag-library-analogy").
TENGYU_QUERY = (
    "How does Tengyu Ma use a library analogy to compare long-context, "
    "fine-tuning, and retrieval-augmented generation?"
)


@pytest.fixture(scope="module")
def conn():
    with psycopg.connect(resolve_dsn(), autocommit=True) as c:
        yield c


def _gen_candidate() -> GeneratorCandidate:
    return GeneratorCandidate(
        candidate_id="qwen3-235b-a22b-instruct",
        provider_model_id="Qwen/Qwen3-235B-A22B-Instruct",
        family="alibaba-qwen",
        provider="deepinfra",
    )


def _judge_candidate() -> JudgeCandidate:
    return JudgeCandidate(
        candidate_id="deepseek-v3.2",
        provider_model_id="deepseek-ai/DeepSeek-V3.2",
        family="deepseek",
        provider="deepinfra",
    )


# ---------------- Plan -------------------------------------------------


def test_plan_node_deterministic_path():
    state = {
        "query": "How does Tengyu Ma define long context?",
        "corpus_id": CORPUS_ID,
    }
    out = plan_mod.plan(state)
    assert out["question_type"] == "single_clip"
    assert out["sub_queries"] == ["How does Tengyu Ma define long context?"]


# ---------------- Retrieve --------------------------------------------


def test_retrieve_node_populates_state_with_real_corpus(conn):
    state = {
        "query": TENGYU_QUERY,
        "corpus_id": CORPUS_ID,
        "sub_queries": [TENGYU_QUERY],
        "iteration": 0,
        "refined_query": None,
    }
    out = retrieve_mod.retrieve(state, conn=conn)
    retrieved = out["retrieved"]
    assert len(retrieved) > 0
    # The contract: state['retrieved'] is list[RetrievedChunk].
    assert all(isinstance(r, RetrievedChunk) for r in retrieved)
    # Each chunk_id must be an int (trace stores ids, not text — see contract #6).
    assert all(isinstance(r.chunk_id, int) for r in retrieved)


# ---------------- Rerank ----------------------------------------------


def test_rerank_node_collapses_to_top8(conn):
    state = {
        "query": TENGYU_QUERY,
        "corpus_id": CORPUS_ID,
        "sub_queries": [TENGYU_QUERY],
        "iteration": 0,
        "refined_query": None,
    }
    state = retrieve_mod.retrieve(state, conn=conn)
    state = rerank_mod.rerank(state, conn=conn)
    reranked = state["reranked"]
    assert len(reranked) <= 8
    assert len(reranked) > 0
    # Reranker emits scores in [0, 1] via single sigmoid.
    for r in reranked:
        assert r.rerank_score is not None
        assert 0.0 <= r.rerank_score <= 1.0


# ---------------- Verify (mocked judge) -------------------------------


def _stub_chunk(chunk_id: int, video_id: str, s: float, e: float, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        video_id=video_id,
        start_sec=s,
        end_sec=e,
        text=text,
        score=0.9,
        rank=1,
        channel_ranks={"dense": 1},
        rerank_score=0.9,
        rrf_score=0.5,
    )


def test_verify_node_high_confidence_does_not_loop(monkeypatch):
    from eval.runners import providers

    def fake(**_k):
        return providers.ProviderResponse(
            raw_text=json.dumps(
                {"confidence": 0.92, "reason": "ok", "refined_query": ""}
            ),
            latency_ms=10,
            provider_model_id="deepseek-ai/DeepSeek-V3.2",
        )

    monkeypatch.setattr(providers, "chat_completion", fake)

    state = {
        "query": "q",
        "judge_candidate": _judge_candidate(),
        "reranked": [_stub_chunk(1, "v", 0.0, 10.0, "evidence")],
        "verify_confidence_threshold": 0.6,
        "iteration": 0,
    }
    out = verify_mod.verify(state, conn=None)
    assert out["verify_confidence"] == pytest.approx(0.92)
    assert out["iteration"] == 0  # unchanged
    assert out["refined_query"] is None


def test_verify_node_low_confidence_loops_once(monkeypatch):
    from eval.runners import providers

    def fake(**_k):
        return providers.ProviderResponse(
            raw_text=json.dumps(
                {"confidence": 0.2, "reason": "weak", "refined_query": "better q"}
            ),
            latency_ms=10,
            provider_model_id="deepseek-ai/DeepSeek-V3.2",
        )

    monkeypatch.setattr(providers, "chat_completion", fake)

    state = {
        "query": "q",
        "judge_candidate": _judge_candidate(),
        "reranked": [_stub_chunk(1, "v", 0.0, 10.0, "weak evidence")],
        "verify_confidence_threshold": 0.6,
        "iteration": 0,
    }
    out = verify_mod.verify(state, conn=None)
    assert out["iteration"] == 1
    assert out["refined_query"] == "better q"


# ---------------- Generate (mocked provider) --------------------------


def test_generate_node_parses_valid_json(monkeypatch):
    from eval.runners import providers

    def fake(**_k):
        return providers.ProviderResponse(
            raw_text=json.dumps(
                {
                    "answer": "Tengyu compares them with a library analogy.",
                    "claims": [{"text": "RAG is like consulting a library."}],
                    "citations": [
                        {
                            "video_id": "W_CYk2ogcDI",
                            "start_sec": 200.0,
                            "end_sec": 250.0,
                            "answer_claim_index": 0,
                        }
                    ],
                    "abstain": False,
                }
            ),
            latency_ms=10,
            provider_model_id="Qwen/Qwen3-235B-A22B-Instruct",
        )

    monkeypatch.setattr(providers, "chat_completion", fake)

    state = {
        "query": TENGYU_QUERY,
        "generator_candidate": _gen_candidate(),
        "reranked": [_stub_chunk(1, "W_CYk2ogcDI", 200.0, 250.0, "library analogy")],
    }
    out = generate_mod.generate(state)
    assert out["parse_ok"] is True
    parsed: GenerationOutput = out["parsed"]
    assert "library analogy" in parsed.answer.lower()
    assert len(parsed.claims) == 1
    assert len(parsed.citations) == 1


# ---------------- Cite ------------------------------------------------


def test_cite_node_validates_overlapping_citation():
    parsed = GenerationOutput(
        answer="x",
        claims=[{"text": "c"}],
        citations=[
            {
                "video_id": "v",
                "start_sec": 100.0,
                "end_sec": 110.0,
                "answer_claim_index": 0,
            }
        ],
        abstain=False,
        parse_ok=True,
        raw_response="",
    )
    state = {
        "query": "q",
        "corpus_id": "c",
        "trace_id": "t",
        "parsed": parsed,
        "reranked": [_stub_chunk(42, "v", 95.0, 115.0, "chunk text")],
        "iteration": 0,
    }
    out = cite_mod.cite(state)
    final = out["final"]
    assert final.abstain is False
    assert len(final.valid_citations) == 1
    assert final.valid_citations[0].matched_chunk_id == 42
    assert final.invalid_citations == []


def test_cite_node_flags_non_overlapping_citation_as_invalid():
    parsed = GenerationOutput(
        answer="x",
        claims=[{"text": "c"}],
        citations=[
            {
                "video_id": "v",
                "start_sec": 500.0,
                "end_sec": 510.0,
                "answer_claim_index": 0,
            }
        ],
        abstain=False,
        parse_ok=True,
        raw_response="",
    )
    state = {
        "query": "q",
        "corpus_id": "c",
        "trace_id": "t",
        "parsed": parsed,
        "reranked": [_stub_chunk(42, "v", 95.0, 115.0, "elsewhere")],
        "iteration": 0,
    }
    out = cite_mod.cite(state)
    final = out["final"]
    assert final.valid_citations == []
    assert len(final.invalid_citations) == 1
    assert final.invalid_citations[0].answer_claim_index == 0


def test_cite_node_rejects_out_of_bounds_answer_claim_index():
    """A citation whose answer_claim_index references a claim slot the
    model never emitted is a hallucination — must land in invalid_citations
    with a clear out-of-bounds reason, NOT be quietly accepted just
    because the timestamp happens to overlap a real chunk."""
    parsed = GenerationOutput(
        answer="x",
        claims=[{"text": "only-claim-zero"}],  # n_claims = 1; valid indices = {0}
        citations=[
            {
                "video_id": "v",
                "start_sec": 100.0,
                "end_sec": 110.0,
                "answer_claim_index": 5,  # out of bounds — would have been valid by overlap
            }
        ],
        abstain=False,
        parse_ok=True,
        raw_response="",
    )
    state = {
        "query": "q",
        "corpus_id": "c",
        "trace_id": "t",
        "parsed": parsed,
        "reranked": [_stub_chunk(42, "v", 95.0, 115.0, "chunk text")],
        "iteration": 0,
    }
    out = cite_mod.cite(state)
    final = out["final"]
    assert final.valid_citations == [], (
        "out-of-bounds answer_claim_index must NOT be accepted even when "
        "the timestamp overlaps a real chunk — the model invented a claim slot"
    )
    assert len(final.invalid_citations) == 1
    inv = final.invalid_citations[0]
    assert inv.answer_claim_index == 5
    assert "out of bounds" in inv.reason
    assert "claims emitted: 1" in inv.reason


def test_cite_node_emits_abstention_on_parse_failure():
    parsed = GenerationOutput(parse_ok=False, raw_response="garbage", error="bad json")
    state = {
        "query": "q",
        "corpus_id": "c",
        "trace_id": "t",
        "parsed": parsed,
        "reranked": [],
        "iteration": 0,
    }
    out = cite_mod.cite(state)
    final = out["final"]
    assert final.abstain is True
    assert final.answer is None
    assert final.valid_citations == []
    assert final.invalid_citations == []
