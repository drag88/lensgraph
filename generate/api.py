"""Public ``answer()`` entry point and the ``make answer`` CLI.

No silent model defaults anywhere (ADR 004 v3.1). Resolution rule per
component:

  * generator / judge: explicit candidate_id  > load_bakeoff_winner
                       > raise ``BakeoffNotYetRunError``.
  * cheap_extraction (planner): explicit candidate_id > None.
    ``None`` is valid — the Plan node uses its deterministic path.

Cross-family invariant: whenever both generator + judge are resolved,
``judge.family != generator.family`` is asserted. Violation raises
``CrossFamilyViolationError`` with a copy-pasteable corrective.

The CLI (``python -m generate.api`` invoked by ``make answer``) reads
``QUERY`` / ``GENERATOR`` / ``JUDGE`` / ``PLANNER`` / ``CORPUS_ID``
from ``os.environ`` to match the project's existing ``make ingest`` /
``make bakeoff-prep`` env-passthrough pattern.

Exit codes (CLI):
  0  — successful answer printed to stdout.
  2  — missing required env var ``QUERY``.
  3  — ``BakeoffNotYetRunError``.
  4  — ``CrossFamilyViolationError``.
  1  — any other unhandled exception (Python default).
"""

from __future__ import annotations

import os
import sys
import time

import psycopg

from db.conn import dsn
from db.repos import eval_runs as eval_runs_repo
from db.repos import traces as traces_repo
from eval.runners import providers
from generate.graph import build_graph
from generate.state import (
    AgentState,
    AnswerResult,
    CheapExtractionCandidate,
    GeneratorCandidate,
    JudgeCandidate,
)
from generate.trace import new_uuid7, verify_threshold_from_env


class BakeoffNotYetRunError(RuntimeError):
    """Raised when ``answer()`` is called without an explicit candidate
    for ``generator`` or ``judge`` AND no locked bakeoff winner exists
    for that component. The message is copy-pasteable per design §5."""


class CrossFamilyViolationError(RuntimeError):
    """Raised when generator.family == judge.family at run start.
    ADR 004 v3.1 anti-preference-leakage invariant."""


def _example_candidates_from_yaml() -> tuple[str, str]:
    """Build a copy-pasteable (GENERATOR=, JUDGE=) example pair, sourced
    from ``model_candidates.yaml``. Picks the first generator candidate
    and the first judge candidate from a different family so the
    suggested command honours the cross-family invariant.

    Sourcing the IDs from yaml rather than hardcoding them keeps the
    no-silent-defaults grep test green and lets the message stay
    current when the candidate set is updated."""
    raw = providers._load_yaml()
    gen_opt = raw["candidates"]["generator"]["options"][0]
    gen_id = gen_opt["id"]
    gen_family = gen_opt["family"]
    judge_opts = raw["candidates"]["judge"]["options"]
    judge_id = next(
        (o["id"] for o in judge_opts if o["family"] != gen_family),
        judge_opts[0]["id"],
    )
    return gen_id, judge_id


def _bakeoff_not_yet_run_message(component: str, corpus_id: str) -> str:
    gen_id, judge_id = _example_candidates_from_yaml()
    return (
        f"BakeoffNotYetRunError: no locked winner for component={component!r} "
        f"in corpus {corpus_id!r}. Pass it explicitly, e.g.:\n\n"
        f'    make answer QUERY="..." GENERATOR={gen_id} JUDGE={judge_id}\n\n'
        "Once the phase-2 generator+judge bakeoff lands a winner_locked=true "
        "row in eval_runs, this fallback path auto-resolves."
    )


def _cross_family_message(
    *,
    generator: GeneratorCandidate,
    judge: JudgeCandidate,
    generator_explicit: bool,
    judge_explicit: bool,
) -> str:
    explicit_label = (
        "explicit" if generator_explicit else "auto-resolved from bakeoff winner"
    )
    judge_label = (
        "explicit" if judge_explicit else "auto-resolved from bakeoff winner"
    )
    # Pick a corrective: ask the user to override the auto-resolved side
    # (or, if both are explicit, the judge by convention).
    if generator_explicit and not judge_explicit:
        fix_hint = f"JUDGE=<id from a non-{generator.family}-family option>"
    elif judge_explicit and not generator_explicit:
        fix_hint = f"GENERATOR=<id from a non-{judge.family}-family option>"
    else:
        fix_hint = f"JUDGE=<id from a non-{generator.family}-family option>"
    return (
        f"CrossFamilyViolationError: judge family {judge.family!r} ({judge_label}) "
        f"shares family with generator {generator.candidate_id!r} ({explicit_label}). "
        "ADR 004 v3.1 requires cross-family judging.\n\n"
        f"    make answer QUERY=\"...\" {fix_hint}"
    )


def _resolve_generator(
    *,
    candidate_id: str | None,
    conn: psycopg.Connection,
    corpus_id: str,
) -> tuple[GeneratorCandidate, bool]:
    """Return (resolved, explicit?) for the generator component."""
    explicit = candidate_id is not None
    if candidate_id is None:
        candidate_id = eval_runs_repo.load_bakeoff_winner(
            conn, component="generator"
        )
    if candidate_id is None:
        raise BakeoffNotYetRunError(
            _bakeoff_not_yet_run_message("generator", corpus_id)
        )
    return (
        GeneratorCandidate(
            candidate_id=candidate_id,
            provider_model_id=providers._resolve_provider_model_id(
                candidate_id, component="generator"
            ),
            family=providers._resolve_candidate_family(
                candidate_id, component="generator"
            ),
            provider=providers._resolve_candidate_provider(
                candidate_id, component="generator"
            ),
        ),
        explicit,
    )


def _resolve_judge(
    *,
    candidate_id: str | None,
    conn: psycopg.Connection,
    corpus_id: str,
) -> tuple[JudgeCandidate, bool]:
    explicit = candidate_id is not None
    if candidate_id is None:
        candidate_id = eval_runs_repo.load_bakeoff_winner(
            conn, component="judge"
        )
    if candidate_id is None:
        raise BakeoffNotYetRunError(
            _bakeoff_not_yet_run_message("judge", corpus_id)
        )
    return (
        JudgeCandidate(
            candidate_id=candidate_id,
            provider_model_id=providers._resolve_provider_model_id(
                candidate_id, component="judge"
            ),
            family=providers._resolve_candidate_family(
                candidate_id, component="judge"
            ),
            provider=providers._resolve_candidate_provider(
                candidate_id, component="judge"
            ),
        ),
        explicit,
    )


def _resolve_planner(
    *,
    candidate_id: str | None,
) -> CheapExtractionCandidate | None:
    """Planner asymmetric resolution (design §5): explicit > None.
    ``None`` engages the deterministic Plan path."""
    if candidate_id is None:
        return None
    return CheapExtractionCandidate(
        candidate_id=candidate_id,
        provider_model_id=providers._resolve_provider_model_id(
            candidate_id, component="cheap_extraction"
        ),
        family=providers._resolve_candidate_family(
            candidate_id, component="cheap_extraction"
        ),
        provider=providers._resolve_candidate_provider(
            candidate_id, component="cheap_extraction"
        ),
    )


def answer(
    query: str,
    *,
    corpus_id: str = "ai_engineering_v0",
    generator_candidate: str | None = None,
    judge_candidate: str | None = None,
    planner_candidate: str | None = None,
) -> AnswerResult:
    """Run the LangGraph loop end-to-end. Synchronous.

    Resolves candidates → opens a psycopg connection → starts the trace
    → builds the graph → invokes → flushes spans → ends the trace →
    returns the terminal ``AnswerResult``.

    Raises:
      * ``BakeoffNotYetRunError`` if generator or judge cannot resolve.
      * ``CrossFamilyViolationError`` if judge.family == generator.family.
      * ``ValueError`` if ``query`` is empty.
    """
    if not query or not query.strip():
        raise ValueError("query must be non-empty")

    trace_id = new_uuid7()
    span_buffer: list[dict] = []
    started_perf = time.perf_counter()

    with psycopg.connect(dsn(), autocommit=True) as conn:
        gen, gen_explicit = _resolve_generator(
            candidate_id=generator_candidate,
            conn=conn,
            corpus_id=corpus_id,
        )
        judge, judge_explicit = _resolve_judge(
            candidate_id=judge_candidate,
            conn=conn,
            corpus_id=corpus_id,
        )
        if judge.family == gen.family:
            raise CrossFamilyViolationError(
                _cross_family_message(
                    generator=gen,
                    judge=judge,
                    generator_explicit=gen_explicit,
                    judge_explicit=judge_explicit,
                )
            )
        planner = _resolve_planner(candidate_id=planner_candidate)

        # Persist the trace skeleton so spans have a FK to attach to.
        traces_repo.start_trace(
            conn,
            trace_id=trace_id,
            query=query,
            corpus_id=corpus_id,
            generator_model_id=gen.provider_model_id,
        )

        state: AgentState = {
            "query": query,
            "corpus_id": corpus_id,
            "generator_candidate": gen,
            "judge_candidate": judge,
            "planner_candidate": planner,
            "verify_confidence_threshold": verify_threshold_from_env(),
            "iteration": 0,
            "refined_query": None,
            "trace_id": trace_id,
        }
        graph = build_graph(conn=conn, span_buffer=span_buffer)
        try:
            final_state: AgentState = graph.invoke(state)
        except BaseException:
            # Best-effort flush spans + mark the trace errored.
            traces_repo.flush_spans(conn, span_buffer)
            elapsed_ms = int((time.perf_counter() - started_perf) * 1000)
            traces_repo.end_trace(
                conn,
                trace_id=trace_id,
                status="error",
                final_answer=None,
                final_citations=None,
                iterations=int(state.get("iteration", 0)),
                latency_ms=elapsed_ms,
            )
            raise

        traces_repo.flush_spans(conn, span_buffer)
        elapsed_ms = int((time.perf_counter() - started_perf) * 1000)

        result_obj: AnswerResult = final_state["final"]
        # Refresh elapsed/iterations on the returned AnswerResult.
        result = result_obj.model_copy(
            update={
                "iterations": int(final_state.get("iteration", 0)),
                "latency_ms": elapsed_ms,
            }
        )
        traces_repo.end_trace(
            conn,
            trace_id=trace_id,
            status="completed",
            final_answer=result.answer,
            final_citations=[c.model_dump() for c in result.valid_citations],
            iterations=result.iterations,
            latency_ms=elapsed_ms,
        )
        return result


def _print_result(result: AnswerResult) -> None:
    print(f"trace_id: {result.trace_id}")
    print(f"abstain: {result.abstain}")
    print(f"iterations: {result.iterations}")
    print(f"latency_ms: {result.latency_ms}")
    print()
    print(result.answer or "(no answer)")
    print()
    print(f"valid_citations ({len(result.valid_citations)}):")
    for c in result.valid_citations:
        print(
            f"  - {c.video_id} @ {c.start_sec:.1f}-{c.end_sec:.1f} "
            f"→ chunk {c.matched_chunk_id}"
        )
    if result.invalid_citations:
        print(f"invalid_citations ({len(result.invalid_citations)}):")
        for c in result.invalid_citations:
            print(
                f"  - {c.video_id} @ {c.start_sec:.1f}-{c.end_sec:.1f} "
                f"({c.reason})"
            )


def _cli() -> int:
    query = os.environ.get("QUERY")
    if not query:
        print("ERROR: QUERY=... required", file=sys.stderr)
        return 2
    try:
        result = answer(
            query=query,
            corpus_id=os.environ.get("CORPUS_ID", "ai_engineering_v0"),
            generator_candidate=os.environ.get("GENERATOR") or None,
            judge_candidate=os.environ.get("JUDGE") or None,
            planner_candidate=os.environ.get("PLANNER") or None,
        )
    except BakeoffNotYetRunError as e:
        print(str(e), file=sys.stderr)
        return 3
    except CrossFamilyViolationError as e:
        print(str(e), file=sys.stderr)
        return 4
    _print_result(result)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
