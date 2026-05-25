"""Embeddings repository tests.

Slow: requires a live Postgres reachable at $POSTGRES_DSN. Spins an ephemeral
test DB per module, applies all migrations once, then exercises the embeds
repo against it.

The sparsevec 1-based-index tests (test_sparse_token_id_zero_maps_to_index_one,
test_sparse_token_id_max_maps_to_index_dim) are load-bearing — without the
+1 shift in upsert_sparse, BGE-M3 token id VOCAB_SIZE-1 would exceed the
sparsevec(VOCAB_SIZE) dimension and crash at INSERT.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import psycopg
import pytest

from chunking.types import Chunk
from db.conn import dsn as resolve_dsn
from db.migrate import apply
from db.repos import chunks as chunks_repo
from db.repos import embeds
from db.repos.talks import Talk
from db.repos.talks import upsert as upsert_talk
from embed.bge_m3 import DENSE_DIM, VOCAB_SIZE

pytestmark = pytest.mark.slow

TEST_DB_NAME = "lensgraph_test_embeds_repo"
VIDEO_ID = "embeds-test-vid"


def _swap_db(dsn: str, new_db: str) -> str:
    head, _, _ = dsn.rpartition("/")
    return f"{head}/{new_db}"


@pytest.fixture(scope="module")
def test_db():
    admin = resolve_dsn()
    with psycopg.connect(admin, autocommit=True) as c:
        c.execute(f"DROP DATABASE IF EXISTS {TEST_DB_NAME}")
        c.execute(f"CREATE DATABASE {TEST_DB_NAME}")
    test_dsn = _swap_db(admin, TEST_DB_NAME)
    apply(test_dsn)
    yield test_dsn
    with psycopg.connect(admin, autocommit=True) as c:
        c.execute(f"DROP DATABASE IF EXISTS {TEST_DB_NAME}")


@pytest.fixture
def conn(test_db):
    """Per-test connection with a clean slate + seeded talk + one chunk."""
    with psycopg.connect(test_db, autocommit=True) as c:
        c.execute("TRUNCATE talks CASCADE")
        upsert_talk(c, _make_talk())
        yield c


def _make_talk() -> Talk:
    return Talk(
        video_id=VIDEO_ID,
        title="Embeds Test Talk",
        speaker="Speaker",
        url="https://example.com/v",
        duration_sec=600,
        format_tags=["narrative"],
        license="mit",
        captions_source="youtube_auto",
        transcript_path="transcripts/embeds-test.vtt",
        transcript_sha256="a" + "0" * 63,
        accessed_at=datetime(2025, 1, 1, tzinfo=UTC),
    )


def _seed_chunk(conn: psycopg.Connection, start: float = 0.0, end: float = 30.0) -> int:
    [chunk_id] = chunks_repo.upsert(
        conn,
        [
            Chunk(
                video_id=VIDEO_ID,
                start_sec=start,
                end_sec=end,
                text="body",
                chunking_strategy="fixed_window",
                frame_secs=[],
                token_count=1,
            )
        ],
    )
    return chunk_id


def test_dense_upsert_round_trip(conn):
    chunk_id = _seed_chunk(conn)
    vec = np.random.default_rng(0).standard_normal(DENSE_DIM).astype(np.float32)
    embeds.upsert_dense(conn, chunk_id, vec)
    embeds._ensure_registered(conn)
    row = conn.execute(
        "SELECT embedding FROM dense_embeds WHERE chunk_id = %s", (chunk_id,)
    ).fetchone()
    assert row is not None
    fetched = np.asarray(row[0])
    assert fetched.shape == (DENSE_DIM,)
    assert np.allclose(fetched, vec, atol=1e-5)


def test_dense_upsert_is_idempotent(conn):
    chunk_id = _seed_chunk(conn)
    rng = np.random.default_rng(1)
    v1 = rng.standard_normal(DENSE_DIM).astype(np.float32)
    v2 = rng.standard_normal(DENSE_DIM).astype(np.float32)
    embeds.upsert_dense(conn, chunk_id, v1)
    embeds.upsert_dense(conn, chunk_id, v2)
    count = conn.execute(
        "SELECT count(*) FROM dense_embeds WHERE chunk_id = %s", (chunk_id,)
    ).fetchone()[0]
    assert count == 1
    embeds._ensure_registered(conn)
    fetched = np.asarray(
        conn.execute(
            "SELECT embedding FROM dense_embeds WHERE chunk_id = %s", (chunk_id,)
        ).fetchone()[0]
    )
    assert np.allclose(fetched, v2, atol=1e-5)


def test_sparse_upsert_round_trip(conn):
    chunk_id = _seed_chunk(conn)
    # Boundary values: token id 0 (-> sparsevec idx 1) and VOCAB_SIZE-1 (-> idx VOCAB_SIZE).
    sparse = {0: 0.5, 1000: 0.7, VOCAB_SIZE - 1: 0.3}
    embeds.upsert_sparse(conn, chunk_id, sparse)
    count = conn.execute(
        "SELECT count(*) FROM sparse_embeds WHERE chunk_id = %s", (chunk_id,)
    ).fetchone()[0]
    assert count == 1


def test_sparse_token_id_zero_maps_to_index_one(conn):
    chunk_id = _seed_chunk(conn)
    embeds.upsert_sparse(conn, chunk_id, {0: 0.9})
    text = conn.execute(
        "SELECT embedding::text FROM sparse_embeds WHERE chunk_id = %s", (chunk_id,)
    ).fetchone()[0]
    # pgvector sparsevec text repr is `{idx:val,idx:val,...}/dim`.
    assert text.startswith("{1:")


def test_sparse_token_id_max_maps_to_index_dim(conn):
    chunk_id = _seed_chunk(conn)
    embeds.upsert_sparse(conn, chunk_id, {VOCAB_SIZE - 1: 0.4})
    text = conn.execute(
        "SELECT embedding::text FROM sparse_embeds WHERE chunk_id = %s", (chunk_id,)
    ).fetchone()[0]
    # VOCAB_SIZE - 1 == 250001 -> stored at 1-based index 250002 (== sparsevec dim).
    assert f"{{{VOCAB_SIZE}:" in text


def test_token_embeds_replace_inserts_rows(conn):
    chunk_id = _seed_chunk(conn)
    vectors = np.random.default_rng(2).standard_normal((5, DENSE_DIM)).astype(np.float32)
    embeds.replace_token_embeds(conn, chunk_id, vectors)
    rows = conn.execute(
        "SELECT position FROM chunk_token_embeds WHERE chunk_id = %s ORDER BY position",
        (chunk_id,),
    ).fetchall()
    assert [r[0] for r in rows] == [0, 1, 2, 3, 4]


def test_token_embeds_replace_overwrites(conn):
    chunk_id = _seed_chunk(conn)
    rng = np.random.default_rng(3)
    embeds.replace_token_embeds(
        conn, chunk_id, rng.standard_normal((5, DENSE_DIM)).astype(np.float32)
    )
    embeds.replace_token_embeds(
        conn, chunk_id, rng.standard_normal((3, DENSE_DIM)).astype(np.float32)
    )
    count = conn.execute(
        "SELECT count(*) FROM chunk_token_embeds WHERE chunk_id = %s", (chunk_id,)
    ).fetchone()[0]
    assert count == 3
