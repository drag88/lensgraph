from __future__ import annotations

from datetime import UTC, datetime

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


def test_visual_retrieve_orders_by_cosine_similarity_non_increasing(
    test_db, populated_frames
):
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
