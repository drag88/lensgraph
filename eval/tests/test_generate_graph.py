"""Slow end-to-end tests for ``generate.graph`` + ``generate.api``.

Six sub-assertions from the brief, each as a separate test function:

  (a) trace + spans persisted in Postgres after a successful answer().
  (b) Verify→Retrieve loop fires when judge returns low confidence.
  (c) Iteration cap stops at 2 (max 3 retrieve spans even when judge
      always returns 0).
  (d) Abstention path produces final.abstain=True with answer=None.
  (e) BakeoffNotYetRunError raised when generator omitted AND no
      winner_locked row exists.
  (f) CrossFamilyViolationError raised when generator + judge share a
      family.

Mocking strategy: monkeypatch ``eval.runners.providers.chat_completion``
once per test with a component-aware fake. ``component='generator'``
returns a valid GenerationOutput JSON; ``component='judge'`` returns a
JSON with the confidence the test wants. The graph is otherwise driven
against the live corpus.

We re-bind the same fake into BOTH ``eval.runners.providers`` and any
node modules that imported the name at module load (verify, generate)
— this is a thin shim because ``from eval.runners import providers``
binds a module reference, so monkeypatching the module attribute is
sufficient.
"""

from __future__ import annotations

import json

import psycopg
import pytest

from db.conn import dsn as resolve_dsn
from generate.api import (
    BakeoffNotYetRunError,
    CrossFamilyViolationError,
    answer,
)

pytestmark = pytest.mark.slow

CORPUS_ID = "ai_engineering_v0"
TENGYU_QUERY = (
    "How does Tengyu Ma use a library analogy to compare long-context, "
    "fine-tuning, and retrieval-augmented generation?"
)


@pytest.fixture(autouse=True)
def _no_provider_failover(monkeypatch):
    """Belt-and-braces — make sure the test never accidentally exercises
    real DeepInfra."""
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    monkeypatch.delenv("LENSGRAPH_PROVIDER_FAILOVER", raising=False)


def _good_generator_payload() -> str:
    return json.dumps(
        {
            "answer": "Tengyu compares RAG, fine-tuning, and long context with a library analogy.",
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
    )


def _make_fake_chat_completion(
    *, judge_confidence: float = 0.95, judge_refined: str = "", abstain: bool = False
):
    """Build a component-aware fake provider call.

    judge_confidence controls the verify loop; abstain forces the
    generator to emit abstain=True (for test d)."""
    from eval.runners import providers

    judge_call_count = {"n": 0}

    def fake(**kwargs):
        component = kwargs.get("component")
        if component == "judge":
            judge_call_count["n"] += 1
            return providers.ProviderResponse(
                raw_text=json.dumps(
                    {
                        "confidence": judge_confidence,
                        "reason": "stub",
                        "refined_query": judge_refined,
                    }
                ),
                latency_ms=5,
                provider_model_id="deepseek-ai/DeepSeek-V3.2",
            )
        if component == "generator":
            if abstain:
                payload = json.dumps(
                    {
                        "answer": "",
                        "claims": [],
                        "citations": [],
                        "abstain": True,
                    }
                )
            else:
                payload = _good_generator_payload()
            return providers.ProviderResponse(
                raw_text=payload,
                latency_ms=20,
                provider_model_id="Qwen/Qwen3-235B-A22B-Instruct",
            )
        raise AssertionError(f"unexpected component: {component}")

    return fake, judge_call_count


def _patch_provider(monkeypatch, fake):
    """Patch chat_completion in every module that imported it."""
    from eval.runners import providers
    from generate.nodes import generate as g_mod
    from generate.nodes import verify as v_mod

    monkeypatch.setattr(providers, "chat_completion", fake)
    # verify.py and generate.py import the module, so the above is
    # sufficient — they look up providers.chat_completion at call time.
    # plan.py imports providers locally inside _llm_plan; deterministic
    # plan does not touch it. (Asserted in test_plan_deterministic.)
    assert v_mod.providers is providers
    assert g_mod.providers is providers


# --- (a) trace + spans persisted -----------------------------------


def test_a_trace_and_spans_persisted(monkeypatch):
    fake, _ = _make_fake_chat_completion(judge_confidence=0.95)
    _patch_provider(monkeypatch, fake)

    result = answer(
        TENGYU_QUERY,
        corpus_id=CORPUS_ID,
        generator_candidate="qwen3-235b-a22b-instruct",
        judge_candidate="deepseek-v3.2",
    )
    trace_id = result.trace_id

    with psycopg.connect(resolve_dsn(), autocommit=True) as conn:
        trace_row = conn.execute(
            "SELECT status, iterations, final_answer FROM traces WHERE trace_id = %s",
            (trace_id,),
        ).fetchone()
        assert trace_row is not None
        assert trace_row[0] == "completed"

        node_rows = conn.execute(
            "SELECT node_name FROM trace_spans WHERE trace_id = %s",
            (trace_id,),
        ).fetchall()
        names = [r[0] for r in node_rows]
        # Happy path with no loop: plan, retrieve, rerank, verify, generate, cite = 6.
        assert len(names) >= 6
        for required in ("plan", "retrieve", "rerank", "verify", "generate", "cite"):
            assert required in names, f"missing span: {required}"


# --- (b) Verify→Retrieve loop fires on low confidence --------------


def test_b_verify_to_retrieve_loop_fires(monkeypatch):
    # First judge call (iter 0) → confidence 0.2 → loop. Subsequent
    # calls → confidence 0.95 → continue. We engineer this via a
    # call-count switch.
    from eval.runners import providers

    judge_n = {"n": 0}

    def fake(**kwargs):
        component = kwargs.get("component")
        if component == "judge":
            judge_n["n"] += 1
            conf = 0.2 if judge_n["n"] == 1 else 0.95
            return providers.ProviderResponse(
                raw_text=json.dumps(
                    {
                        "confidence": conf,
                        "reason": "stub",
                        "refined_query": "tengyu library analogy long context",
                    }
                ),
                latency_ms=5,
                provider_model_id="deepseek-ai/DeepSeek-V3.2",
            )
        if component == "generator":
            return providers.ProviderResponse(
                raw_text=_good_generator_payload(),
                latency_ms=20,
                provider_model_id="Qwen/Qwen3-235B-A22B-Instruct",
            )
        raise AssertionError(f"unexpected component: {component}")

    _patch_provider(monkeypatch, fake)

    result = answer(
        TENGYU_QUERY,
        corpus_id=CORPUS_ID,
        generator_candidate="qwen3-235b-a22b-instruct",
        judge_candidate="deepseek-v3.2",
    )

    with psycopg.connect(resolve_dsn(), autocommit=True) as conn:
        retrieve_iters = conn.execute(
            "SELECT iteration FROM trace_spans WHERE trace_id = %s "
            "AND node_name = 'retrieve' ORDER BY started_at",
            (result.trace_id,),
        ).fetchall()
        iters = [r[0] for r in retrieve_iters]
        # We expect at least one retrieve span with iteration=1.
        assert any(i == 1 for i in iters), f"no looped retrieve span; iters={iters}"


# --- (c) iteration cap stops at 2 ----------------------------------


def test_c_iteration_cap_stops_at_2(monkeypatch):
    """Judge always returns 0.0 → loop max-twice → stop at iteration=2.
    Three retrieve spans expected (initial + 2 loops); a generate span
    must still be present (we exit the loop, not the graph)."""
    from eval.runners import providers

    def fake(**kwargs):
        component = kwargs.get("component")
        if component == "judge":
            return providers.ProviderResponse(
                raw_text=json.dumps(
                    {"confidence": 0.0, "reason": "no", "refined_query": "again"}
                ),
                latency_ms=5,
                provider_model_id="deepseek-ai/DeepSeek-V3.2",
            )
        if component == "generator":
            return providers.ProviderResponse(
                raw_text=_good_generator_payload(),
                latency_ms=20,
                provider_model_id="Qwen/Qwen3-235B-A22B-Instruct",
            )
        raise AssertionError(f"unexpected component: {component}")

    _patch_provider(monkeypatch, fake)

    result = answer(
        TENGYU_QUERY,
        corpus_id=CORPUS_ID,
        generator_candidate="qwen3-235b-a22b-instruct",
        judge_candidate="deepseek-v3.2",
    )

    with psycopg.connect(resolve_dsn(), autocommit=True) as conn:
        retrieve_iters = conn.execute(
            "SELECT iteration FROM trace_spans WHERE trace_id = %s "
            "AND node_name = 'retrieve' ORDER BY started_at",
            (result.trace_id,),
        ).fetchall()
        iters = [r[0] for r in retrieve_iters]
        # Initial retrieve (iter=0) + two loops (iter=1, iter=2).
        assert max(iters) == 2, f"iters={iters}"
        # Generate must still have fired (loop didn't dead-end).
        gen_rows = conn.execute(
            "SELECT count(*) FROM trace_spans WHERE trace_id = %s "
            "AND node_name = 'generate'",
            (result.trace_id,),
        ).fetchone()
        assert gen_rows[0] >= 1


# --- (d) abstention path -------------------------------------------


def test_d_abstention_path_returns_none_answer(monkeypatch):
    fake, _ = _make_fake_chat_completion(judge_confidence=0.95, abstain=True)
    _patch_provider(monkeypatch, fake)

    result = answer(
        TENGYU_QUERY,
        corpus_id=CORPUS_ID,
        generator_candidate="qwen3-235b-a22b-instruct",
        judge_candidate="deepseek-v3.2",
    )
    assert result.abstain is True
    assert result.answer is None
    assert result.valid_citations == []


# --- (e) BakeoffNotYetRunError -------------------------------------


def test_e_bakeoff_not_yet_run_error_when_generator_omitted(monkeypatch):
    """No winner_locked row exists in dev → answer() with generator
    omitted must raise. We don't even need to mock providers here —
    the resolver raises before any HTTP."""
    # If a prior test in the same run inserted a winner_locked row, this
    # would fail. Make sure no eval_runs winner exists.
    with psycopg.connect(resolve_dsn(), autocommit=True) as conn:
        winner = conn.execute(
            "SELECT count(*) FROM eval_runs WHERE summary->>'winner_locked' = 'true' "
            "AND summary->>'component' = 'generator'"
        ).fetchone()
        if winner[0] > 0:
            pytest.skip("a winner_locked generator row exists; cannot test the no-default path")

    with pytest.raises(BakeoffNotYetRunError, match=r"GENERATOR=|generator"):
        answer(
            TENGYU_QUERY,
            corpus_id=CORPUS_ID,
            generator_candidate=None,
            judge_candidate="deepseek-v3.2",
        )


# --- (f) CrossFamilyViolationError ---------------------------------


def test_f_cross_family_violation_raises():
    """Both candidates resolved to family='alibaba-qwen' → raise."""
    with pytest.raises(CrossFamilyViolationError, match="family"):
        answer(
            TENGYU_QUERY,
            corpus_id=CORPUS_ID,
            generator_candidate="qwen3-235b-a22b-instruct",
            judge_candidate="qwen3-235b-a22b-instruct",
        )
