"""Internal bakeoff harness — NOT the product path.

The LangGraph graph (``generate/graph.py``) is the product path. This
harness exists so phase-2's generator+judge bakeoff can isolate generator
quality from Plan/Verify/Cite confounds — one provider call → one
``eval_runs`` row → one ``eval_results`` row.

Per design §6:
  * The model never sees gold labels; citations index into the model's
    OWN claims list via ``answer_claim_index``.
  * The shared ``parse_generation_output`` (``generate/parser.py``) is
    the single source of JSON parse semantics for both this harness
    and the LangGraph generate node — drift here means the bakeoff
    and the langgraph comparison measure different things.
  * No retry on parse failure. ``parse_ok=False`` is real bakeoff signal.
  * Provider errors (max-retries-exhausted, malformed-200, non-retryable
    4xx) propagate; the harness does NOT swallow them into a synthetic
    ``parse_ok=False`` row because that conflates infrastructure failure
    with model failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import yaml

from db.repos import eval_runs
from eval.runners import providers
from generate.parser import parse_generation_output


@dataclass(frozen=True)
class RetrievedChunkLite:
    """Minimum chunk shape the harness needs.

    Phase-2's bakeoff runner builds these from any retrieval source
    (the product path uses ``retrieve/types.py::RetrievedChunk``; this
    harness deliberately doesn't depend on that file because it's not
    on the product path)."""

    chunk_id: int
    video_id: str
    start_sec: float
    end_sec: float
    text: str


_SYSTEM_PROMPT = """\
You answer questions about engineering conference talks. You will be given a question and a list of timestamped transcript chunks. Decompose your answer into atomic claims; cite each claim by 0-indexed position into your OWN claims list (NOT into anything you weren't shown).

Emit STRICT JSON only — no prose outside the object, no code fences — conforming exactly to:

  {
    "answer": "<prose answer>",
    "claims": [{"text": "<atomic claim>"}, {"text": "<atomic claim>"}, ...],
    "citations": [
      {"video_id": "<vid>", "start_sec": <float>, "end_sec": <float>, "answer_claim_index": <int>},
      ...
    ],
    "abstain": false
  }

If the chunks do NOT contain the answer, emit exactly:

  {"answer": "", "claims": [], "citations": [], "abstain": true}

Each claim is an OBJECT with a "text" string field (not a bare string). Every claim must be cited at least once."""


def _build_prompt(query: str, chunks: list[RetrievedChunkLite]) -> list[dict]:
    rendered = "\n\n".join(
        f"[{c.video_id} @ {c.start_sec:.1f}-{c.end_sec:.1f}] {c.text}"
        for c in chunks
    )
    user = f"Question: {query}\n\nChunks:\n{rendered}"
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _load_candidate_set() -> dict:
    """Snapshot the candidate yaml at run time so the eval_runs row is
    reproducible without re-reading the file."""
    return yaml.safe_load(providers._CONFIG_PATH.read_text(encoding="utf-8"))


def generate_for_bakeoff(
    *,
    conn,
    example_id: str,
    query: str,
    chunks: list[RetrievedChunkLite],
    candidate_id: str,
    chunking_strategy: str = "fixed_window",
    embedding_model_id: str = "bge-m3-all-channels",
    run_id: str,
) -> dict:
    """Run one provider call → parse → write one eval_runs + one
    eval_results row.

    ``candidate_id`` is the yaml ``id`` (e.g. ``'qwen3-235b-a22b-instruct'``);
    the provider call resolves it to the wire-level ``provider_model_id``
    internally. ``run_id`` is caller-owned — the caller iterates over
    dev_gold and chooses how to scope ``run_id`` (one per sweep, one per
    example, etc.).

    Provider errors (``providers.ProviderError``) propagate. They mean
    infrastructure broke, not that the model failed — conflating them
    would inflate the parse_ok=False count in phase-2 metrics.
    """
    messages = _build_prompt(query, chunks)
    # ProviderError raises here and propagates — see module docstring.
    # 90s timeout mirrors generate/nodes/generate.py — Qwen3-235B /
    # DeepSeek V3.2 over top-8 chunks exceed the providers default 30s.
    resp = providers.chat_completion(
        candidate_id=candidate_id,
        messages=messages,
        component="generator",
        timeout_sec=90.0,
    )

    parsed = parse_generation_output(resp.raw_text)

    summary = {
        "component": "minimal_generation_sweep",
        "candidate_id": candidate_id,
        "parse_ok": parsed.parse_ok,
        "latency_ms": resp.latency_ms,
        "abstain": parsed.abstain,
    }
    eval_runs.insert_run(
        conn,
        run_id=run_id,
        run_date=date.today(),
        code_path="minimal_generation",
        chunking_strategy=chunking_strategy,
        embedding_model_id=embedding_model_id,
        generator_model_id=candidate_id,
        judge_model_id=None,
        judge_prompt_hash=None,
        candidate_set_yaml=_load_candidate_set(),
        summary=summary,
    )

    system_output = parsed.model_dump()
    metrics = {
        "parse_ok": parsed.parse_ok,
        "latency_ms": resp.latency_ms,
        "abstain": parsed.abstain,
        "claim_count": len(parsed.claims),
        "citation_count": len(parsed.citations),
    }
    eval_runs.insert_result(
        conn,
        run_id=run_id,
        example_id=example_id,
        system_output=system_output,
        metrics=metrics,
    )

    return {
        "run_id": run_id,
        "example_id": example_id,
        "parse_ok": parsed.parse_ok,
        "latency_ms": resp.latency_ms,
        "parsed": system_output,
    }
