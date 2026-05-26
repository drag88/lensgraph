from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import psycopg
import pytest
from pgvector.psycopg import register_vector
from PIL import Image, ImageDraw

from db.conn import dsn as resolve_dsn
from db.migrate import apply
from db.repos.talks import Talk
from db.repos.talks import upsert as upsert_talk
from embed.colqwen import POOLED_DIM, encode_image_pooled
from retrieve import visual
from retrieve.types import ChannelResult

pytestmark = pytest.mark.slow

TEST_DB_NAME = "lensgraph_test_visual"


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


@pytest.fixture(scope="module")
def populated_frames(test_db):
    """Generate 4 synthetic slide images, encode pooled, insert into frames.

    Slides are 640x480 white with distinct text per frame. The ColQwen
    model maps each to a 128-d pooled embedding. Storing them via the
    pgvector adapter so the cosine HNSW path is exercised end-to-end.
    """
    slide_texts = [
        "Database architecture diagram with primary key indexes",
        "User authentication flow chart and OAuth handshake",
        "Performance metrics graph showing latency over time",
        "Conclusion summary slide with next steps and references",
    ]
    images = []
    for t in slide_texts:
        img = Image.new("RGB", (640, 480), color="white")
        draw = ImageDraw.Draw(img)
        draw.multiline_text((20, 200), t, fill="black")
        images.append(img)

    pooled = encode_image_pooled(images)
    assert pooled.shape == (len(slide_texts), POOLED_DIM)

    with psycopg.connect(test_db, autocommit=True) as c:
        register_vector(c)
        upsert_talk(
            c,
            Talk(
                video_id="visual-test-vid",
                title="Visual smoke",
                speaker="S",
                url="https://example.com/x",
                duration_sec=120,
                format_tags=["slides_heavy"],
                license="cc-by",
                captions_source="manual_transcript",
                transcript_path="transcripts/visual/test.vtt",
                transcript_sha256="a" + "0" * 63,
                accessed_at=datetime(2025, 1, 1, tzinfo=UTC),
            ),
        )
        for i, _text in enumerate(slide_texts):
            c.execute(
                """
                INSERT INTO frames(video_id, frame_sec, image_path, sha256, pooled_embedding)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    "visual-test-vid",
                    i * 10.0,
                    f"/tmp/visual-test-{i}.png",
                    ("b" * 64),
                    pooled[i],
                ),
            )
    yield {"texts": slide_texts}


def test_pooled_embedding_shape_is_128(populated_frames):
    """Sanity: ColQwen pooled output is (N, 128) — the dim the schema requires."""
    img = Image.new("RGB", (64, 64), color="white")
    pooled = encode_image_pooled([img])
    assert pooled.shape == (1, POOLED_DIM)


def test_visual_retrieve_returns_top_k_frames(test_db, populated_frames):
    with psycopg.connect(test_db, autocommit=True) as conn:
        results = visual.retrieve_frames(conn, "database design", top_k=3)
    assert len(results) == 3
    assert all(r.video_id == "visual-test-vid" for r in results)
    assert [r.rank for r in results] == [1, 2, 3]


def test_visual_retrieve_orders_by_cosine_similarity_non_increasing(test_db, populated_frames):
    with psycopg.connect(test_db, autocommit=True) as conn:
        results = visual.retrieve_frames(conn, "database design", top_k=4)
    scores = [r.score for r in results]
    for prev, cur in zip(scores, scores[1:], strict=False):
        assert cur <= prev, f"scores must be non-increasing by rank, got {scores!r}"


def test_visual_retrieve_carries_required_fields(test_db, populated_frames):
    with psycopg.connect(test_db, autocommit=True) as conn:
        results = visual.retrieve_frames(conn, "database design", top_k=2)
    for r in results:
        assert isinstance(r.frame_id, int) and r.frame_id > 0
        assert r.video_id == "visual-test-vid"
        assert r.frame_sec >= 0
        assert isinstance(r.image_path, str) and r.image_path
        assert isinstance(r.score, float)


# ---------------------------------------------------------------------------
# Stage-2 channel-level `retrieve()` — pooled HNSW prefilter + per-patch MaxSim.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def populated_chunks_and_patches(test_db, populated_frames):
    """Layer chunks + frame_patches onto the populated_frames fixture so
    stage-2 has data to JOIN against.

    Chunks are 30s windows so each frame (at frame_sec = i * 10) falls into
    a distinct chunk's span; this means each of the 4 chunks contains 1
    candidate frame after the prefilter — making the chunk-level scoring
    behave like the frame-level scoring (good for an order-preserving
    sanity check).

    Patch embeddings reuse the pooled vector as a 1-patch frame. The
    actual ColQwen per-patch encoder is the other agent's slice; here we
    only need *something* in frame_patches so stage (c) has matrices to
    multiply.
    """
    slide_texts = [
        "Database architecture diagram with primary key indexes",
        "User authentication flow chart and OAuth handshake",
        "Performance metrics graph showing latency over time",
        "Conclusion summary slide with next steps and references",
    ]
    with psycopg.connect(test_db, autocommit=True) as c:
        register_vector(c)
        frames = c.execute(
            """
            SELECT frame_id, frame_sec, pooled_embedding
            FROM frames
            WHERE video_id = 'visual-test-vid'
            ORDER BY frame_sec
            """
        ).fetchall()

        # 4 chunks: [0, 10), [10, 20), [20, 30), [30, 40). Frames at 0, 10, 20, 30.
        for i, text in enumerate(slide_texts):
            c.execute(
                """
                INSERT INTO chunks (
                    video_id, chunking_strategy, start_sec, end_sec,
                    text, token_count, frame_secs
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (
                    "visual-test-vid",
                    "fixed_window",
                    float(i * 10),
                    float(i * 10 + 10),
                    text,
                    len(text.split()),
                    [float(i * 10)],
                ),
            )

        # One patch per frame, reusing the pooled vector. Real ColQwen emits
        # ~1030 patches per slide; a single-patch frame is enough to exercise
        # the SQL fetch + matmul without inflating the test DB.
        for frame_id, _frame_sec, pooled in frames:
            c.execute(
                """
                INSERT INTO frame_patches (frame_id, patch_index, embedding)
                VALUES (%s, %s, %s)
                ON CONFLICT (frame_id, patch_index) DO NOTHING
                """,
                (int(frame_id), 0, pooled),
            )
    yield


def _patch_query_patches(monkeypatch):
    """Replace BOTH ColQwen text encoders at the retrieve.visual import site
    so stage-1 and stage-2 see a consistent query direction.

    Stage 1 (`encode_text_query`) drives the pooled HNSW prefilter; stage 2
    (`encode_text_query_patches`) drives the per-patch MaxSim. If only the
    patch encoder is monkeypatched, stage 1 uses the real ColQwen text
    encoder for "database architecture", which is free to pick any of the
    4 synthetic slides as its nearest — the slides are not strongly
    distinguishable in the model's text-image alignment, so stage 1 picking
    a different frame than stage 2 expects is a test-fixture flake, not a
    product bug. Both stubs return vectors aligned to frame 0's pooled
    direction so the database-architecture chunk wins deterministically
    at both stages.
    """
    from retrieve import visual as visual_mod

    with psycopg.connect(_swap_db(resolve_dsn(), TEST_DB_NAME), autocommit=True) as c:
        register_vector(c)
        row = c.execute(
            """
            SELECT pooled_embedding
            FROM frames
            WHERE video_id = 'visual-test-vid' AND frame_sec = 0.0
            """
        ).fetchone()
    target = np.asarray(row[0], dtype=np.float32)

    fake_q_patches = np.stack([target, target, target, target]).astype(np.float32)
    monkeypatch.setattr(visual_mod, "encode_text_query_patches", lambda _q: fake_q_patches)
    monkeypatch.setattr(visual_mod, "encode_text_query", lambda _q: target)


def test_visual_retrieve_channel_returns_chunks_not_frames(
    test_db, populated_chunks_and_patches, monkeypatch
):
    """Stage-2 retrieve() returns ChannelResult rows (chunk-level), not
    FrameResult — proving the frame→chunk mapping ran."""
    _patch_query_patches(monkeypatch)
    with psycopg.connect(test_db, autocommit=True) as conn:
        results = visual.retrieve(conn, "database architecture", top_k=2)
    assert len(results) >= 1
    for r in results:
        assert isinstance(r, ChannelResult)
        assert isinstance(r.chunk_id, int) and r.chunk_id > 0
        assert r.video_id == "visual-test-vid"
        assert r.end_sec > r.start_sec
        assert isinstance(r.text, str) and len(r.text) > 0
        assert isinstance(r.score, float)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    scores = [r.score for r in results]
    for prev, cur in zip(scores, scores[1:], strict=False):
        assert cur <= prev


def test_visual_retrieve_channel_maps_frame_to_correct_chunk(
    test_db, populated_chunks_and_patches, monkeypatch
):
    """The top-ranked chunk's span must contain the frame that drove the
    pooled-prefilter score. Catches a frame→chunk JOIN bug where the
    half-open interval is inverted."""
    _patch_query_patches(monkeypatch)
    with psycopg.connect(test_db, autocommit=True) as conn:
        # Sanity: stage-1 picks the database-architecture frame first.
        frames = visual.retrieve_frames(conn, "database architecture", top_k=1)
        top_frame_sec = frames[0].frame_sec
        results = visual.retrieve(conn, "database architecture", top_k=1)
    assert len(results) == 1
    top = results[0]
    assert top.start_sec <= top_frame_sec < top.end_sec


def test_visual_retrieve_channel_empty_when_no_frames(test_db, monkeypatch):
    """A clean DB with no frames returns []. No model load is triggered for
    the empty pooled prefilter."""
    fresh_db = TEST_DB_NAME + "_empty"
    admin = resolve_dsn()
    with psycopg.connect(admin, autocommit=True) as c:
        c.execute(f"DROP DATABASE IF EXISTS {fresh_db}")
        c.execute(f"CREATE DATABASE {fresh_db}")
    try:
        empty_dsn = _swap_db(admin, fresh_db)
        apply(empty_dsn)
        # We do NOT need to patch encode_text_query_patches here because the
        # stage-1 pooled query short-circuits to () before stage-c is reached.
        with psycopg.connect(empty_dsn, autocommit=True) as conn:
            assert visual.retrieve(conn, "anything", top_k=5) == []
    finally:
        with psycopg.connect(admin, autocommit=True) as c:
            c.execute(f"DROP DATABASE IF EXISTS {fresh_db}")
