"""eval_runs + eval_results repository.

Three concerns:

  1. ``insert_run`` — append a row to ``eval_runs`` for a code_path run
     (``minimal_generation``, ``embeddings_bakeoff``, eventually
     ``langgraph_loop``). Caller owns ``run_id`` uniqueness; we never
     auto-generate.
  2. ``insert_result`` — append per-example output + metrics keyed by
     ``(run_id, example_id)``. PK violation on duplicate is intentional
     contract — re-running the same example for the same run should
     fail loudly so callers know to roll a new ``run_id``.
  3. ``lock_bakeoff_winner`` / ``load_bakeoff_winner`` — the bakeoff
     "winner" lock lives in the ``summary jsonb`` column rather than a
     separate table. Per design §5: the lock is queried with
     ``summary->>'winner_locked' = 'true'`` so a phase-2 selection rule
     can flip it without a migration. ``load_bakeoff_winner`` returns
     ``None`` on no-lock (caller wraps into
     ``BakeoffNotYetRunError``); the repo is a pure SQL adapter.

The ``code_path`` column distinguishes the harness path
(``minimal_generation``) from the full LangGraph product path
(``langgraph_loop``). Per design §5 SQL, selection ONLY reads from
``minimal_generation`` rows — phase-3 langgraph runs are comparison-only
and never inform selection.
"""

from __future__ import annotations

import json
from datetime import date

import psycopg
from psycopg.types.json import Jsonb


def insert_run(
    conn: psycopg.Connection,
    *,
    run_id: str,
    run_date: date,
    code_path: str,
    chunking_strategy: str,
    embedding_model_id: str,
    generator_model_id: str | None = None,
    judge_model_id: str | None = None,
    judge_prompt_hash: str | None = None,
    candidate_set_yaml: dict,
    summary: dict | None = None,
) -> None:
    """Append a row to ``eval_runs``. Caller is responsible for
    ``run_id`` uniqueness (it's the PK; psycopg raises on collision)."""
    conn.execute(
        """
        INSERT INTO eval_runs (
            run_id, run_date, code_path, chunking_strategy,
            embedding_model_id, generator_model_id, judge_model_id,
            judge_prompt_hash, candidate_set_yaml, summary
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            run_id,
            run_date,
            code_path,
            chunking_strategy,
            embedding_model_id,
            generator_model_id,
            judge_model_id,
            judge_prompt_hash,
            Jsonb(candidate_set_yaml),
            Jsonb(summary if summary is not None else {}),
        ),
    )


def insert_result(
    conn: psycopg.Connection,
    *,
    run_id: str,
    example_id: str,
    system_output: dict,
    metrics: dict,
) -> None:
    """Append a row to ``eval_results``. ``(eval_run_id, example_id)``
    is the PK; duplicates raise ``UniqueViolation`` by design."""
    conn.execute(
        """
        INSERT INTO eval_results (eval_run_id, example_id, system_output, metrics)
        VALUES (%s, %s, %s, %s)
        """,
        (run_id, example_id, Jsonb(system_output), Jsonb(metrics)),
    )


def lock_bakeoff_winner(
    conn: psycopg.Connection,
    *,
    component: str,
    candidate_id: str,
    run_id: str,
    chunking_strategy: str = "fixed_window",
    extra_summary: dict | None = None,
) -> None:
    """Mark ``run_id`` as the winner for ``component`` by merging into
    its ``summary jsonb``.

    Idempotent: re-running with the same ``(component, candidate_id,
    run_id)`` after the lock is already set yields the same JSON value,
    so the UPDATE is a structural no-op (we don't gate on a WHERE
    clause; we always rewrite the merged value — Postgres rewrites the
    row but the column ends up byte-equal). Tests assert byte-equal
    summary across two calls.

    ``chunking_strategy`` is the v0 lock dimension — phase 2's selection
    SQL filters by it (design §5 SQL). ``extra_summary`` is a free-form
    bag for measurement metadata (per-channel scores, leader gap,
    runner-up) that the embeddings bakeoff scaffold uses.
    """
    merge: dict = {
        "winner_locked": True,
        "component": component,
        "winner_candidate_id": candidate_id,
    }
    if extra_summary:
        merge.update(extra_summary)

    conn.execute(
        """
        UPDATE eval_runs
           SET summary = COALESCE(summary, '{}'::jsonb) || %s::jsonb
         WHERE run_id = %s
           AND chunking_strategy = %s
        """,
        (json.dumps(merge), run_id, chunking_strategy),
    )


def load_bakeoff_winner(
    conn: psycopg.Connection,
    *,
    component: str,
    chunking_strategy: str = "fixed_window",
) -> str | None:
    """Return the locked winner's ``candidate_id`` (yaml id), or ``None``.

    Design §5 SQL: ``ORDER BY created_at DESC LIMIT 1``. We do NOT
    filter on ``code_path`` here even though design specifies
    ``code_path='minimal_generation'`` for generator/judge selection;
    that filter belongs in the caller (generate/api.py) per component,
    because phase-1 step 35a uses ``code_path='embeddings_bakeoff'``
    for the text_embeddings lock. Keeping the repo agnostic lets the
    same function serve both."""
    row = conn.execute(
        """
        SELECT summary->>'winner_candidate_id'
          FROM eval_runs
         WHERE summary->>'winner_locked' = 'true'
           AND summary->>'component' = %s
           AND chunking_strategy = %s
         ORDER BY created_at DESC
         LIMIT 1
        """,
        (component, chunking_strategy),
    ).fetchone()
    if row is None or row[0] is None:
        return None
    return row[0]
