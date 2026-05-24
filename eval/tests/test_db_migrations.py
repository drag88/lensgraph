"""Schema-shape test for migrations 0001..0005.

Slow: requires a live Postgres reachable at $POSTGRES_DSN (default
postgresql://lensgraph:lensgraph@localhost:5432/lensgraph). Spins up an
ephemeral DB per session, applies all migrations once, then asserts the
catalog matches the contract in docs/phase-1-design.md §2 — especially the
load-bearing sparsevec(250002) dimension.
"""

from __future__ import annotations

import psycopg
import pytest

from db.conn import dsn as resolve_dsn
from db.migrate import apply

pytestmark = pytest.mark.slow

TEST_DB_NAME = "lensgraph_test_migrations"


def _swap_db(dsn: str, new_db: str) -> str:
    head, _, _ = dsn.rpartition("/")
    return f"{head}/{new_db}"


@pytest.fixture(scope="module")
def test_db():
    admin = resolve_dsn()
    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(f"DROP DATABASE IF EXISTS {TEST_DB_NAME}")
        conn.execute(f"CREATE DATABASE {TEST_DB_NAME}")
    test_dsn = _swap_db(admin, TEST_DB_NAME)
    apply(test_dsn)
    yield test_dsn
    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(f"DROP DATABASE IF EXISTS {TEST_DB_NAME}")


def _column_type(conn: psycopg.Connection, table: str, col: str) -> str:
    row = conn.execute(
        """
        SELECT format_type(a.atttypid, a.atttypmod)
        FROM pg_attribute a
        JOIN pg_class c ON a.attrelid = c.oid
        WHERE c.relname = %s AND a.attname = %s AND a.attnum > 0
        """,
        (table, col),
    ).fetchone()
    return row[0] if row else ""


def test_extensions_installed(test_db):
    with psycopg.connect(test_db) as c:
        rows = c.execute(
            "SELECT extname FROM pg_extension WHERE extname IN ('vector', 'pgmq')"
        ).fetchall()
    assert {r[0] for r in rows} == {"vector", "pgmq"}


def test_all_tables_exist(test_db):
    expected = {
        "schema_migrations",
        "talks",
        "ingest_step_status",
        "chunks",
        "dense_embeds",
        "sparse_embeds",
        "chunk_token_embeds",
        "frames",
        "frame_patches",
        "eval_runs",
        "eval_results",
        "traces",
        "trace_spans",
    }
    with psycopg.connect(test_db) as c:
        rows = c.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        ).fetchall()
    present = {r[0] for r in rows}
    missing = expected - present
    assert not missing, f"missing tables: {missing}"


def test_sparse_embeds_sparsevec_250002(test_db):
    """XLM-RoBERTa-large vocab size = 250002 is a load-bearing contract (§2)."""
    with psycopg.connect(test_db) as c:
        assert _column_type(c, "sparse_embeds", "embedding") == "sparsevec(250002)"


def test_dense_embeds_vector_1024(test_db):
    with psycopg.connect(test_db) as c:
        assert _column_type(c, "dense_embeds", "embedding") == "vector(1024)"


def test_chunk_token_embeds_vector_1024(test_db):
    with psycopg.connect(test_db) as c:
        assert _column_type(c, "chunk_token_embeds", "embedding") == "vector(1024)"


def test_frames_pooled_vector_128(test_db):
    with psycopg.connect(test_db) as c:
        assert _column_type(c, "frames", "pooled_embedding") == "vector(128)"


def test_frame_patches_vector_128(test_db):
    with psycopg.connect(test_db) as c:
        assert _column_type(c, "frame_patches", "embedding") == "vector(128)"


def test_chunks_tsv_is_tsvector(test_db):
    with psycopg.connect(test_db) as c:
        assert _column_type(c, "chunks", "tsv") == "tsvector"


def test_talks_format_tags_is_text_array(test_db):
    with psycopg.connect(test_db) as c:
        assert _column_type(c, "talks", "format_tags") == "text[]"


def test_chunks_end_sec_check_rejects_inverted_span(test_db):
    """The end_sec > start_sec CHECK is a structural invariant; verify it fires."""
    fake_sha = "a" + ("0" * 63)
    with psycopg.connect(test_db) as c, c.transaction():
        c.execute(
            """
            INSERT INTO talks(video_id, title, speaker, url, duration_sec, format_tags,
                              license, captions_source, transcript_path, transcript_sha256,
                              accessed_at)
            VALUES ('mig-test-vid', 't', 's', 'http://x', 100, ARRAY['narrative'],
                    'mit', 'youtube_auto', 'tx/test.vtt', %s, now())
            """,
            (fake_sha,),
        )
    with psycopg.connect(test_db) as c, pytest.raises(psycopg.errors.CheckViolation):
        c.execute(
            """
            INSERT INTO chunks(video_id, chunking_strategy, start_sec, end_sec,
                               text, token_count)
            VALUES ('mig-test-vid', 'fixed_window', 10.0, 5.0, 'inverted', 1)
            """
        )


def test_schema_migrations_records_all_versions(test_db):
    with psycopg.connect(test_db) as c:
        rows = c.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
    assert [r[0] for r in rows] == [
        "0001_init",
        "0002_chunks_embeds",
        "0003_frames",
        "0004_eval_runs",
        "0005_traces",
    ]


def test_apply_is_idempotent(test_db):
    """Re-applying against an already-migrated DB is a no-op."""
    assert apply(test_db) == []
