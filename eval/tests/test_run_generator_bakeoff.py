"""Fast unit tests for the generator bakeoff (#2).

No live API, no DB: the provider/judge/retriever seams are stubbed. The
DB-persistence + real-provider path is exercised by the one manual
provisional run (see the report), not by ``make test``.
"""

from __future__ import annotations

from eval.runners import measure_generation as mg
from eval.runners.measure_generation import (
    GenerationRun,
    GoldQuery,
    JudgeResult,
    NegativeQuery,
    _p95_ms,
    _parse_judge_output,
)
from eval.runners.minimal_generation import RetrievedChunkLite
from generate.parser import AnswerClaim, Citation, GenerationOutput
from scripts.run_generator_bakeoff import (
    check_minimums,
    select_generator_winner,
)

_CHUNKS = [
    RetrievedChunkLite(chunk_id=1, video_id="V", start_sec=0.0, end_sec=10.0, text="alpha"),
    RetrievedChunkLite(chunk_id=2, video_id="V", start_sec=10.0, end_sec=20.0, text="beta"),
]


def _stub_retriever(conn, query):
    return _CHUNKS


def _gen_output(*, claims, citations, abstain=False, parse_ok=True):
    return GenerationOutput(
        answer="" if abstain else "an answer",
        claims=[AnswerClaim(text=c) for c in claims],
        citations=[
            Citation(video_id="V", start_sec=0.0, end_sec=10.0, answer_claim_index=i)
            for i in citations
        ],
        abstain=abstain,
        parse_ok=parse_ok,
    )


# ---- aggregation over dev_gold ------------------------------------------


def test_dev_gold_aggregation_mixes_grounded_abstain_parsefail():
    examples = [
        GoldQuery("ex-good", "q1"),
        GoldQuery("ex-partial", "q2"),
        GoldQuery("ex-abstain", "q3"),
        GoldQuery("ex-parsefail", "q4"),
    ]

    def gen(gen_id, q, chunks):
        if q == "q1":  # 2 claims both grounded, 1 accurate citation
            return GenerationRun(_gen_output(claims=["a", "b"], citations=[0]), 1200)
        if q == "q2":  # 2 claims, 1 grounded; 2 citations, 1 accurate
            return GenerationRun(_gen_output(claims=["a", "b"], citations=[0, 1]), 1800)
        if q == "q3":  # abstained (false refusal on answerable)
            return GenerationRun(_gen_output(claims=[], citations=[], abstain=True), 900)
        return GenerationRun(_gen_output(claims=[], citations=[], parse_ok=False), 1500)

    def judge(judge_id, system, *, question, chunks, parsed):
        if question == "q1":
            return JudgeResult(True, [True, True], [True], "raw")
        if question == "q2":
            return JudgeResult(True, [True, False], [True, False], "raw")
        raise AssertionError("judge should not be called for abstain/parsefail")

    summary, per_ex = mg.measure_generator_over_dev_gold(
        conn=None,
        generator_id="gemma-4-31b",
        judge_id="deepseek-v3.2",
        judge_system="SYS",
        examples=examples,
        retriever=_stub_retriever,
        generate_fn=gen,
        judge_fn=judge,
    )

    assert summary["n_dev_gold"] == 4
    # parse_ok on 3 of 4 (q4 failed)
    assert summary["json_parse_success"] == 0.75
    # one abstain of four
    assert summary["false_refusal_rate"] == 0.25
    # claims grounded: q1=1.0, q2=0.5 → mean 0.75 over the 2 judged examples
    assert summary["claims_grounded_rate"] == 0.75
    assert summary["n_judged_claims_examples"] == 2
    # citation accuracy: q1=1.0, q2=0.5 → mean 0.75
    assert summary["citation_accuracy"] == 0.75
    assert summary["n_judged_citation_examples"] == 2
    assert summary["p95_latency_ms"] == 1800
    assert per_ex["ex-good"]["claims_grounded_frac"] == 1.0
    assert per_ex["ex-partial"]["citation_accuracy_frac"] == 0.5
    assert per_ex["ex-abstain"]["abstain"] is True


def test_judge_failure_excluded_and_counted():
    examples = [GoldQuery("ex1", "q1")]

    def gen(gen_id, q, chunks):
        return GenerationRun(_gen_output(claims=["a"], citations=[0]), 1000)

    def judge(judge_id, system, *, question, chunks, parsed):
        return JudgeResult(judge_parse_ok=False, claim_supported=[], citation_accurate=[], raw="x")

    summary, per_ex = mg.measure_generator_over_dev_gold(
        conn=None,
        generator_id="g",
        judge_id="j",
        judge_system="S",
        examples=examples,
        retriever=_stub_retriever,
        generate_fn=gen,
        judge_fn=judge,
    )
    assert summary["judge_failures"] == 1
    assert summary["claims_grounded_rate"] is None
    assert summary["n_judged_claims_examples"] == 0
    assert per_ex["ex1"]["judge_parse_ok"] is False


# ---- abstention over negatives ------------------------------------------


def test_negative_refusal_rate():
    negatives = [NegativeQuery(f"neg{i}", f"q{i}") for i in range(4)]

    def gen(gen_id, q, chunks):
        # abstain on 3 of 4
        abstain = q != "q2"
        return GenerationRun(
            _gen_output(claims=[] if abstain else ["a"], citations=[], abstain=abstain), 800
        )

    summary, per_ex = mg.measure_abstention_over_negatives(
        conn=None,
        generator_id="g",
        negatives=negatives,
        retriever=_stub_retriever,
        generate_fn=gen,
    )
    assert summary["n_negatives"] == 4
    assert summary["corpus_negative_refusal"] == 0.75
    assert per_ex["neg2"]["abstain"] is False


# ---- selection rule ------------------------------------------------------


def test_select_cheapest_within_band():
    measurements = [
        {
            "candidate_id": "gemma-4-31b",
            "score": 0.90,
            "price_output_per_mtok_usd": 0.38,
            "meets_minimums": True,
        },
        {
            "candidate_id": "qwen3-235b-a22b-instruct",
            "score": 0.88,
            "price_output_per_mtok_usd": 0.10,
            "meets_minimums": True,
        },
    ]
    # qwen within 3pp of gemma (0.90 - 0.88 = 2pp) and cheaper → wins
    winner = select_generator_winner(measurements, tie_break_pp=3)
    assert winner["candidate_id"] == "qwen3-235b-a22b-instruct"


def test_select_outside_band_keeps_leader():
    measurements = [
        {
            "candidate_id": "gemma-4-31b",
            "score": 0.90,
            "price_output_per_mtok_usd": 0.38,
            "meets_minimums": True,
        },
        {
            "candidate_id": "qwen3-235b-a22b-instruct",
            "score": 0.80,
            "price_output_per_mtok_usd": 0.10,
            "meets_minimums": True,
        },
    ]
    # qwen is 10pp behind → outside the 3pp band; leader gemma wins
    winner = select_generator_winner(measurements, tie_break_pp=3)
    assert winner["candidate_id"] == "gemma-4-31b"


def test_select_none_when_no_minimums_met():
    measurements = [
        {
            "candidate_id": "gemma-4-31b",
            "score": 0.90,
            "price_output_per_mtok_usd": 0.38,
            "meets_minimums": False,
        },
        {
            "candidate_id": "qwen3-235b-a22b-instruct",
            "score": 0.88,
            "price_output_per_mtok_usd": 0.10,
            "meets_minimums": False,
        },
    ]
    assert select_generator_winner(measurements, tie_break_pp=3) is None


# ---- minimums check ------------------------------------------------------

_MINS = {
    "claims_supported_min": 0.85,
    "citation_accuracy_min": 0.85,
    "json_parse_success_min": 0.98,
    "corpus_negative_refusal_min": 0.90,
    "p95_latency_sec_max": 2.0,
}


def test_minimums_all_met():
    gen = {
        "claims_grounded_rate": 0.90,
        "citation_accuracy": 0.88,
        "json_parse_success": 1.0,
        "p95_latency_ms": 1500,
    }
    neg = {"corpus_negative_refusal": 0.95}
    met, checks = check_minimums(gen, neg, _MINS)
    assert met is True
    assert all(checks.values())


def test_minimums_latency_fail():
    gen = {
        "claims_grounded_rate": 0.90,
        "citation_accuracy": 0.88,
        "json_parse_success": 1.0,
        "p95_latency_ms": 5000,  # 5s > 2s max
    }
    neg = {"corpus_negative_refusal": 0.95}
    met, checks = check_minimums(gen, neg, _MINS)
    assert met is False
    assert checks["p95_latency_sec_max"] is False
    assert checks["claims_supported_min"] is True


def test_minimums_none_score_is_miss():
    gen = {
        "claims_grounded_rate": None,
        "citation_accuracy": None,
        "json_parse_success": 1.0,
        "p95_latency_ms": 1000,
    }
    neg = {"corpus_negative_refusal": 0.95}
    met, checks = check_minimums(gen, neg, _MINS)
    assert met is False
    assert checks["claims_supported_min"] is False


def test_minimums_none_latency_fails():
    # A generator we could not time must FAIL the latency minimum, never
    # silently pass via a 0ms coercion.
    gen = {
        "claims_grounded_rate": 0.95,
        "citation_accuracy": 0.95,
        "json_parse_success": 1.0,
        "p95_latency_ms": None,
    }
    neg = {"corpus_negative_refusal": 0.95}
    met, checks = check_minimums(gen, neg, _MINS)
    assert met is False
    assert checks["p95_latency_sec_max"] is False


def test_minimums_missing_latency_key_fails():
    gen = {
        "claims_grounded_rate": 0.95,
        "citation_accuracy": 0.95,
        "json_parse_success": 1.0,
        # p95_latency_ms absent entirely
    }
    neg = {"corpus_negative_refusal": 0.95}
    met, checks = check_minimums(gen, neg, _MINS)
    assert met is False
    assert checks["p95_latency_sec_max"] is False


# ---- judge output parsing ------------------------------------------------


def test_parse_judge_valid():
    raw = (
        '{"claim_assessments": [{"claim_index": 0, "supported": true},'
        '{"claim_index": 1, "supported": false}],'
        '"citation_assessments": [{"citation_index": 0, "accurate": true}]}'
    )
    r = _parse_judge_output(raw, n_claims=2, n_citations=1)
    assert r.judge_parse_ok is True
    assert r.claim_supported == [True, False]
    assert r.citation_accurate == [True]


def test_parse_judge_fenced():
    raw = '```json\n{"claim_assessments": [{"claim_index": 0, "supported": true}], "citation_assessments": []}\n```'
    r = _parse_judge_output(raw, n_claims=1, n_citations=0)
    assert r.judge_parse_ok is True
    assert r.claim_supported == [True]


def test_parse_judge_malformed():
    r = _parse_judge_output("not json at all", n_claims=2, n_citations=1)
    assert r.judge_parse_ok is False
    assert r.claim_supported == []


def test_parse_judge_out_of_range_index_defaults_false():
    # judge omits claim 1 and references a bogus index 5 → claim 1 stays False
    raw = '{"claim_assessments": [{"claim_index": 0, "supported": true}, {"claim_index": 5, "supported": true}], "citation_assessments": []}'
    r = _parse_judge_output(raw, n_claims=2, n_citations=0)
    assert r.claim_supported == [True, False]


def test_parse_judge_string_true_gets_no_credit():
    # Stringified booleans must NOT inflate: "true"/"false" both → no credit.
    raw = (
        '{"claim_assessments": [{"claim_index": 0, "supported": "true"},'
        '{"claim_index": 1, "supported": "false"}],'
        '"citation_assessments": [{"citation_index": 0, "accurate": "true"}]}'
    )
    r = _parse_judge_output(raw, n_claims=2, n_citations=1)
    # JSON parsed fine, so the response is "ok", but stringy values credit nothing.
    assert r.judge_parse_ok is True
    assert r.claim_supported == [False, False]
    assert r.citation_accurate == [False]


def test_parse_judge_numeric_truthy_gets_no_credit():
    # 1/0 are not JSON booleans → no credit (1 must not be coerced to True).
    raw = (
        '{"claim_assessments": [{"claim_index": 0, "supported": 1},'
        '{"claim_index": 1, "supported": 0}],'
        '"citation_assessments": []}'
    )
    r = _parse_judge_output(raw, n_claims=2, n_citations=0)
    assert r.claim_supported == [False, False]


def test_parse_judge_literal_false_stays_false():
    # Sanity: a genuine literal false stays false (no double-negation bug).
    raw = '{"claim_assessments": [{"claim_index": 0, "supported": false}], "citation_assessments": []}'
    r = _parse_judge_output(raw, n_claims=1, n_citations=0)
    assert r.claim_supported == [False]


# ---- p95 -----------------------------------------------------------------


def test_p95_small_n():
    assert _p95_ms([100, 200, 300, 400, 500]) == 500
    assert _p95_ms([]) == 0
    assert _p95_ms([42]) == 42
