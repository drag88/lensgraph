"""Real faithfulness + abstention measurement for the generator bakeoff.

Drives the **minimal_generation** harness path (not the LangGraph product
path): for one generator candidate, run a single provider call per
``dev_gold`` example over the retrieved chunks, parse the structured
output, then score it with a cross-family judge. Abstention is measured
separately over ``negative.jsonl`` corpus-scope rows.

Why minimal_generation and not the graph: ``db.repos.eval_runs.load_bakeoff_winner``
reads generator/judge selection ONLY from ``code_path='minimal_generation'``
rows (design §5). The graph's Plan/Verify/Cite stages are confounds for a
generator comparison and are comparison-only.

Metric definitions (see ``docs/eval-methodology.md`` Answer + Abstention tiers):

* ``claims_grounded_rate`` — over answered examples, the mean fraction of
  the model's OWN emitted claims that the judge finds supported by the
  retrieved chunk text. This is the faithfulness reading of the Answer
  tier's ``ClaimsSupported``: the model never sees gold, citations index
  the model's own claims (design §6), so the harness measures grounding
  of emitted claims, not recall of ``expected_claims``. The yaml minimum
  ``claims_supported_min`` is applied to this proxy in the provisional
  bakeoff; the distinction is documented in the report.
* ``citation_accuracy`` — over answered examples, the mean fraction of the
  model's citations whose cited span actually supports the referenced
  claim ("right answer, wrong source" catcher).
* ``json_parse_success`` — fraction of dev_gold generations that parsed.
* ``false_refusal_rate`` — fraction of dev_gold (answerable) examples the
  model abstained on. Should be low.
* ``p95_latency_ms`` — nearest-rank p95 over the dev_gold generation calls.
* ``corpus_negative_refusal`` — fraction of corpus-negative examples the
  model correctly abstained on.

The judge call and generator call go through ``providers.chat_completion``;
both are injectable (``generate_fn`` / ``judge_fn`` / ``retriever``) so the
fast tests can stub them with no live API.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import psycopg

from eval.runners import providers
from eval.runners.minimal_generation import RetrievedChunkLite
from generate.parser import GenerationOutput, parse_generation_output
from retrieve import bm25, dense, multivec, rrf, sparse

_CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpora" / "ai_engineering_v0"
_DEV_GOLD_PATH = _CORPUS_DIR / "dev_gold.jsonl"
_NEGATIVE_PATH = _CORPUS_DIR / "negative.jsonl"
_JUDGE_PROMPT_PATH = Path(__file__).resolve().parent / "judges" / "faithfulness_v1.md"

# The generator sees the top-N fused text chunks. 8 mirrors the product
# path's post-rerank top-8; the minimal harness skips the reranker (no
# local model load) and feeds the RRF top-8 directly.
_GENERATOR_TOP_K = 8


@dataclass(frozen=True)
class GoldQuery:
    example_id: str
    question: str


@dataclass(frozen=True)
class NegativeQuery:
    example_id: str
    question: str


def load_dev_gold_single_clip(path: Path = _DEV_GOLD_PATH) -> list[GoldQuery]:
    """Load single_clip dev_gold rows (the answerable generator eval set).

    Synthesis rows are reported separately per methodology and skipped
    here; negatives live in their own file."""
    out: list[GoldQuery] = []
    with path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            ex = json.loads(line)
            if ex.get("question_type") != "single_clip":
                continue
            out.append(GoldQuery(example_id=ex["id"], question=ex["question"]))
    return out


def load_corpus_negatives(path: Path = _NEGATIVE_PATH) -> list[NegativeQuery]:
    """Load corpus-scope negatives — the primary abstention signal
    (methodology: video-scope negatives test single-video reasoning and
    are not blended into corpus refusal)."""
    out: list[NegativeQuery] = []
    with path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            ex = json.loads(line)
            if ex.get("question_type") != "negative":
                continue
            if ex.get("negative_scope") != "corpus":
                continue
            out.append(NegativeQuery(example_id=ex["id"], question=ex["question"]))
    return out


def load_judge_prompt(path: Path = _JUDGE_PROMPT_PATH) -> str:
    return path.read_text(encoding="utf-8")


def judge_prompt_sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def retrieve_for_generation(
    conn: psycopg.Connection, query: str, *, top_k: int = _GENERATOR_TOP_K
) -> list[RetrievedChunkLite]:
    """RRF over the four text channels (bm25 + dense + sparse + multivec),
    truncated to ``top_k``. Visual + reranker are deliberately excluded:
    the minimal harness isolates generator quality from those stages and
    avoids a second local-model load."""
    b = bm25.retrieve(conn, query, top_k=30)
    d = dense.retrieve(conn, query, top_k=30)
    s = sparse.retrieve(conn, query, top_k=30)
    m = multivec.retrieve(conn, query, top_k=30)
    fused = rrf.fuse({"bm25": b, "dense": d, "sparse": s, "multivec": m}, top_k=top_k)
    return [
        RetrievedChunkLite(
            chunk_id=r.chunk_id,
            video_id=r.video_id,
            start_sec=r.start_sec,
            end_sec=r.end_sec,
            text=r.text,
        )
        for r in fused
    ]


# ---- generator + judge calls (injectable seams) --------------------------


@dataclass(frozen=True)
class GenerationRun:
    parsed: GenerationOutput
    latency_ms: int


def run_generator(candidate_id: str, query: str, chunks: list[RetrievedChunkLite]) -> GenerationRun:
    """One generator provider call → parse. Mirrors minimal_generation's
    prompt + parse contract (shared ``parse_generation_output``)."""
    from eval.runners.minimal_generation import _build_prompt

    resp = providers.chat_completion(
        candidate_id=candidate_id,
        messages=_build_prompt(query, chunks),
        component="generator",
        timeout_sec=90.0,
    )
    return GenerationRun(parsed=parse_generation_output(resp.raw_text), latency_ms=resp.latency_ms)


@dataclass(frozen=True)
class JudgeResult:
    judge_parse_ok: bool
    claim_supported: list[bool]
    citation_accurate: list[bool]
    raw: str


def _render_chunks(chunks: list[RetrievedChunkLite]) -> str:
    return "\n\n".join(
        f"[{i}] [{c.video_id} @ {c.start_sec:.1f}-{c.end_sec:.1f}] {c.text}"
        for i, c in enumerate(chunks)
    )


def _build_judge_messages(
    judge_system: str,
    *,
    question: str,
    chunks: list[RetrievedChunkLite],
    parsed: GenerationOutput,
) -> list[dict]:
    claims = [{"index": i, "text": c.text} for i, c in enumerate(parsed.claims)]
    citations = [
        {
            "index": i,
            "video_id": c.video_id,
            "start_sec": c.start_sec,
            "end_sec": c.end_sec,
            "answer_claim_index": c.answer_claim_index,
        }
        for i, c in enumerate(parsed.citations)
    ]
    user = (
        f"QUESTION:\n{question}\n\n"
        f"RETRIEVED_CHUNKS:\n{_render_chunks(chunks)}\n\n"
        f"ANSWER:\n{parsed.answer}\n\n"
        f"CLAIMS (0-indexed):\n{json.dumps(claims, ensure_ascii=False)}\n\n"
        f"CITATIONS (0-indexed):\n{json.dumps(citations, ensure_ascii=False)}\n\n"
        "Grade now. Emit STRICT JSON per the contract."
    )
    return [
        {"role": "system", "content": judge_system},
        {"role": "user", "content": user},
    ]


def _parse_judge_output(raw: str, *, n_claims: int, n_citations: int) -> JudgeResult:
    """Parse the judge JSON into aligned boolean lists.

    Robust to a ```json fence. Missing/extra/mis-indexed entries are
    handled by reading each assessment's declared index and defaulting an
    unseen index to ``False`` (an unscored claim is not credited). A
    malformed judge response sets ``judge_parse_ok=False`` and yields
    empty lists — the caller excludes that example from the aggregate and
    counts it as a judge failure.

    STRICT boolean policy: only a JSON literal ``true`` (Python ``True``)
    credits a claim/citation. Any other value — ``false``, the strings
    ``"true"``/``"false"``, ``1``/``0``, ``null``, or a missing key —
    yields NO credit (the entry stays ``False``). This can deflate a
    score but can NEVER inflate it, so a judge that emits stringified
    booleans cannot silently lift faithfulness above its true value."""
    text = raw.strip()
    if text.startswith("```"):
        # strip a single fence
        body = text.split("\n", 1)[1] if "\n" in text else ""
        body = body.rsplit("```", 1)[0]
        text = body.strip()
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return JudgeResult(judge_parse_ok=False, claim_supported=[], citation_accurate=[], raw=raw)
    if not isinstance(obj, dict):
        return JudgeResult(judge_parse_ok=False, claim_supported=[], citation_accurate=[], raw=raw)

    supported = [False] * n_claims
    for a in obj.get("claim_assessments", []) or []:
        try:
            idx = int(a["claim_index"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= idx < n_claims:
            # Strict: only JSON literal ``true`` credits. "true"/1/0 do not.
            supported[idx] = a.get("supported") is True

    accurate = [False] * n_citations
    for a in obj.get("citation_assessments", []) or []:
        try:
            idx = int(a["citation_index"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= idx < n_citations:
            # Strict: only JSON literal ``true`` credits. "true"/1/0 do not.
            accurate[idx] = a.get("accurate") is True

    return JudgeResult(
        judge_parse_ok=True,
        claim_supported=supported,
        citation_accurate=accurate,
        raw=raw,
    )


def run_judge(
    judge_id: str,
    judge_system: str,
    *,
    question: str,
    chunks: list[RetrievedChunkLite],
    parsed: GenerationOutput,
) -> JudgeResult:
    """One judge provider call → parse into per-item booleans."""
    messages = _build_judge_messages(judge_system, question=question, chunks=chunks, parsed=parsed)
    resp = providers.chat_completion(
        candidate_id=judge_id,
        messages=messages,
        component="judge",
        timeout_sec=90.0,
    )
    return _parse_judge_output(
        resp.raw_text, n_claims=len(parsed.claims), n_citations=len(parsed.citations)
    )


# ---- aggregation ---------------------------------------------------------


def _p95_ms(latencies: Sequence[int]) -> int:
    """Nearest-rank p95. With small n this is effectively the max; we
    report it honestly and the report quotes n."""
    if not latencies:
        return 0
    ordered = sorted(latencies)
    rank = max(0, int(round(0.95 * len(ordered) + 0.5)) - 1)
    rank = min(rank, len(ordered) - 1)
    return ordered[rank]


GenerateFn = Callable[[str, str, list[RetrievedChunkLite]], GenerationRun]
JudgeFn = Callable[..., JudgeResult]
Retriever = Callable[[psycopg.Connection, str], list[RetrievedChunkLite]]


def measure_generator_over_dev_gold(
    conn: psycopg.Connection,
    *,
    generator_id: str,
    judge_id: str,
    judge_system: str,
    examples: Sequence[GoldQuery],
    retriever: Retriever = retrieve_for_generation,
    generate_fn: GenerateFn = run_generator,
    judge_fn: JudgeFn = run_judge,
) -> tuple[dict, dict[str, dict]]:
    """Score one generator over the answerable dev_gold set.

    Returns ``(summary, per_example_detail)``. ``summary`` is the
    lightweight aggregate for ``eval_runs.summary``; ``per_example_detail``
    carries one ``eval_results``-bound record per example."""
    per_example: dict[str, dict] = {}
    latencies: list[int] = []
    parse_oks: list[bool] = []
    abstains: list[bool] = []
    claim_fracs: list[float] = []
    citation_fracs: list[float] = []
    judge_failures = 0

    for ex in examples:
        chunks = retriever(conn, ex.question)
        run = generate_fn(generator_id, ex.question, chunks)
        parsed = run.parsed
        latencies.append(run.latency_ms)
        parse_oks.append(parsed.parse_ok)
        abstains.append(parsed.abstain)

        rec: dict = {
            "question": ex.question,
            "parse_ok": parsed.parse_ok,
            "abstain": parsed.abstain,
            "latency_ms": run.latency_ms,
            "claim_count": len(parsed.claims),
            "citation_count": len(parsed.citations),
            "answer": parsed.answer,
        }

        if parsed.parse_ok and not parsed.abstain and parsed.claims:
            judged = judge_fn(
                judge_id,
                judge_system,
                question=ex.question,
                chunks=chunks,
                parsed=parsed,
            )
            if judged.judge_parse_ok:
                cf = sum(judged.claim_supported) / len(judged.claim_supported)
                claim_fracs.append(cf)
                rec["claims_grounded_frac"] = cf
                rec["claim_supported"] = judged.claim_supported
                if parsed.citations:
                    af = sum(judged.citation_accurate) / len(judged.citation_accurate)
                    citation_fracs.append(af)
                    rec["citation_accuracy_frac"] = af
                    rec["citation_accurate"] = judged.citation_accurate
                else:
                    rec["citation_accuracy_frac"] = None
            else:
                judge_failures += 1
                rec["judge_parse_ok"] = False
        per_example[ex.example_id] = rec

    n = len(examples)
    summary = {
        "n_dev_gold": n,
        "json_parse_success": (sum(parse_oks) / n) if n else 0.0,
        "false_refusal_rate": (sum(abstains) / n) if n else 0.0,
        "p95_latency_ms": _p95_ms(latencies),
        "claims_grounded_rate": (sum(claim_fracs) / len(claim_fracs)) if claim_fracs else None,
        "citation_accuracy": (
            (sum(citation_fracs) / len(citation_fracs)) if citation_fracs else None
        ),
        "n_judged_claims_examples": len(claim_fracs),
        "n_judged_citation_examples": len(citation_fracs),
        "judge_failures": judge_failures,
    }
    return summary, per_example


def measure_abstention_over_negatives(
    conn: psycopg.Connection,
    *,
    generator_id: str,
    negatives: Sequence[NegativeQuery],
    retriever: Retriever = retrieve_for_generation,
    generate_fn: GenerateFn = run_generator,
) -> tuple[dict, dict[str, dict]]:
    """Score corpus_negative_refusal: fraction of corpus negatives the
    generator correctly abstains on."""
    per_example: dict[str, dict] = {}
    refusals: list[bool] = []
    for ex in negatives:
        chunks = retriever(conn, ex.question)
        run = generate_fn(generator_id, ex.question, chunks)
        refused = bool(run.parsed.abstain)
        refusals.append(refused)
        per_example[ex.example_id] = {
            "question": ex.question,
            "abstain": refused,
            "parse_ok": run.parsed.parse_ok,
            "latency_ms": run.latency_ms,
            "answer": run.parsed.answer,
        }
    n = len(negatives)
    summary = {
        "n_negatives": n,
        "corpus_negative_refusal": (sum(refusals) / n) if n else 0.0,
    }
    return summary, per_example
