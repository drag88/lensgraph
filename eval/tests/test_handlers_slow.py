"""End-to-end handler tests against real Postgres + real BGE-M3 + real ffmpeg.

Slow: requires live Postgres reachable at $POSTGRES_DSN. Per-module
ephemeral DB with migrations + queues pre-created. Per-test TRUNCATE +
purge for isolation.

Prove the handler contract end-to-end:
  - chunk_handler reads a real VTT, persists chunks, fans out embed_text.
  - embed_text_handler loads BGE-M3 (once per module) and writes all three
    embed tables for the correct chunk_id.
  - frames_handler runs real ffmpeg against a synth lavfi video, persists
    frames, writes embed_frames status rows, enqueues ingest_embed_frames.
  - re-running chunk_handler / frames_handler is idempotent at the
    chunks / frames table (the queue is NOT — the embed handlers'
    claim_for_update no-ops on redelivery, verified in test_pgmq_smoke).
  - a handler crash inside process_one rolls back chunks/frames AND
    status AND the queue read.
  - the 1-based sparsevec shift round-trips cleanly for token_id 0 AND
    token_id VOCAB_SIZE - 1 (the boundary tokens).
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import psycopg
import pytest

from chunking.types import Chunk
from db.conn import dsn as resolve_dsn
from db.migrate import apply
from db.repos import chunks as chunks_repo
from db.repos import embeds as embeds_repo
from db.repos import ingest_step_status as iss
from db.repos.talks import Talk
from db.repos.talks import upsert as upsert_talk
from embed.bge_m3 import VOCAB_SIZE
from ingest import frames as frames_mod
from ingest import handlers
from queues import pgmq_client, workers

pytestmark = pytest.mark.slow

TEST_DB_NAME = "lensgraph_test_handlers"


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
    with psycopg.connect(test_dsn, autocommit=True) as c:
        pgmq_client.ensure_queues(c)
    yield test_dsn
    with psycopg.connect(admin, autocommit=True) as c:
        c.execute(f"DROP DATABASE IF EXISTS {TEST_DB_NAME}")


@pytest.fixture
def conn(test_db):
    """Per-test connection with a clean slate. CASCADE drops dense / sparse /
    token rows along with chunks."""
    with psycopg.connect(test_db, autocommit=True) as c:
        c.execute(
            "TRUNCATE chunk_token_embeds, sparse_embeds, dense_embeds, "
            "chunks, ingest_step_status, talks CASCADE"
        )
        for q in pgmq_client.QUEUES:
            pgmq_client.purge(c, q)
        yield c


def _make_talk(video_id: str, transcript_path: Path, duration_sec: int = 30) -> Talk:
    return Talk(
        video_id=video_id,
        title="Handler test",
        speaker="S",
        url="https://example.com/v",
        duration_sec=duration_sec,
        format_tags=["narrative"],
        license="cc-by",
        captions_source="manual_transcript",
        transcript_path=str(transcript_path),
        transcript_sha256="a" + "0" * 63,
        accessed_at=datetime(2025, 1, 1, tzinfo=UTC),
    )


def _make_vtt(tmp_path: Path, name: str = "v.vtt") -> Path:
    """30s VTT with three cues — produces ≥1 chunk under fixed_window's
    default 30s window / 5s overlap."""
    p = tmp_path / name
    p.write_text(
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:10.000\n"
        "transformer architectures use self attention layers\n\n"
        "00:00:10.000 --> 00:00:20.000\n"
        "encoder decoder stacks process tokens in parallel\n\n"
        "00:00:20.000 --> 00:00:30.000\n"
        "positional encodings provide order information for tokens\n",
        encoding="utf-8",
    )
    return p


# -- chunk_handler --------------------------------------------------------


def test_chunk_handler_end_to_end_writes_chunks_and_enqueues_embed_jobs(
    conn, tmp_path
):
    """Seed talk + chunk-pending status + chunk message. Run process_one
    with chunk_handler. Confirm chunks landed, embed_text status rows
    written, ingest_embed_text queue populated with matching entity_ids."""
    vtt = _make_vtt(tmp_path)
    talk = _make_talk("vid-1", vtt)
    upsert_talk(conn, talk)
    iss.upsert(conn, "vid-1", step="chunk", status="pending")
    pgmq_client.send(
        conn, "ingest_chunk", {"video_id": "vid-1", "step": "chunk", "entity_id": 0}
    )

    with conn.transaction():
        processed = workers.process_one(conn, "ingest_chunk", handlers.chunk_handler)
    assert processed is True

    chunk_ids = [
        r[0]
        for r in conn.execute(
            "SELECT chunk_id FROM chunks WHERE video_id = 'vid-1' ORDER BY start_sec"
        ).fetchall()
    ]
    assert len(chunk_ids) >= 1

    embed_status_ids = sorted(
        r[0]
        for r in conn.execute(
            "SELECT entity_id FROM ingest_step_status "
            "WHERE video_id = 'vid-1' AND step = 'embed_text'"
        ).fetchall()
    )
    assert embed_status_ids == sorted(chunk_ids)

    msgs = pgmq_client.read(conn, "ingest_embed_text", vt=30, qty=len(chunk_ids))
    assert len(msgs) == len(chunk_ids)
    msg_entity_ids = sorted(m.message["entity_id"] for m in msgs)
    assert msg_entity_ids == sorted(chunk_ids)
    assert all(m.message["video_id"] == "vid-1" for m in msgs)


def test_embed_text_handler_end_to_end_writes_all_three_embed_tables(
    conn, tmp_path
):
    """Seed a talk + one chunk row + embed_text status. Run embed_text_handler
    via process_one. This is the test that loads BGE-M3 (~80s cold, ~10s
    warm) so it's the slowest in the suite."""
    vtt = _make_vtt(tmp_path)
    upsert_talk(conn, _make_talk("vid-embed", vtt))
    [chunk_id] = chunks_repo.upsert(
        conn,
        [
            Chunk(
                video_id="vid-embed",
                start_sec=0.0,
                end_sec=10.0,
                text="encoder decoder transformer attention model",
                chunking_strategy="fixed_window",
                frame_secs=[],
                token_count=6,
            )
        ],
    )
    iss.upsert(
        conn, "vid-embed", step="embed_text", entity_id=chunk_id, status="pending"
    )
    pgmq_client.send(
        conn,
        "ingest_embed_text",
        {"video_id": "vid-embed", "step": "embed_text", "entity_id": chunk_id},
    )

    with conn.transaction():
        processed = workers.process_one(
            conn, "ingest_embed_text", handlers.embed_text_handler
        )
    assert processed is True

    dense_n = conn.execute(
        "SELECT count(*) FROM dense_embeds WHERE chunk_id = %s", (chunk_id,)
    ).fetchone()[0]
    sparse_n = conn.execute(
        "SELECT count(*) FROM sparse_embeds WHERE chunk_id = %s", (chunk_id,)
    ).fetchone()[0]
    token_n = conn.execute(
        "SELECT count(*) FROM chunk_token_embeds WHERE chunk_id = %s", (chunk_id,)
    ).fetchone()[0]
    assert dense_n == 1
    assert sparse_n == 1
    assert token_n > 0

    positions = [
        r[0]
        for r in conn.execute(
            "SELECT position FROM chunk_token_embeds "
            "WHERE chunk_id = %s ORDER BY position",
            (chunk_id,),
        ).fetchall()
    ]
    assert positions == list(range(token_n))


def test_chunk_handler_retry_does_not_duplicate_chunks(conn, tmp_path):
    """chunk_handler is idempotent at the chunks table — re-running for the
    same payload returns the same chunk_ids without inserting duplicates.
    The queue is NOT idempotent (embed_text msgs accumulate); the embed
    handler's claim_for_update no-ops on already-completed work."""
    vtt = _make_vtt(tmp_path)
    upsert_talk(conn, _make_talk("retry-vid", vtt))
    iss.upsert(conn, "retry-vid", step="chunk", status="pending")
    pgmq_client.send(
        conn, "ingest_chunk", {"video_id": "retry-vid", "step": "chunk", "entity_id": 0}
    )
    with conn.transaction():
        workers.process_one(conn, "ingest_chunk", handlers.chunk_handler)

    first_count = conn.execute(
        "SELECT count(*) FROM chunks WHERE video_id = 'retry-vid'"
    ).fetchone()[0]
    assert first_count >= 1

    # Direct re-invocation (skip process_one's claim path so we exercise the
    # repo's natural-key conflict instead of the worker's idempotency).
    with conn.transaction():
        handlers.chunk_handler(
            conn, {"video_id": "retry-vid", "step": "chunk", "entity_id": 0}
        )

    second_count = conn.execute(
        "SELECT count(*) FROM chunks WHERE video_id = 'retry-vid'"
    ).fetchone()[0]
    assert second_count == first_count


def test_handler_crash_rolls_back_chunks_and_status(conn, tmp_path, monkeypatch):
    """If the handler raises mid-flight, process_one's caller transaction
    rolls back: no chunks row, status row reverted to pre-claim state, and
    the message is back on the visible queue."""
    vtt = _make_vtt(tmp_path)
    upsert_talk(conn, _make_talk("crash-vid", vtt))
    iss.upsert(conn, "crash-vid", step="chunk", status="pending")
    pgmq_client.send(
        conn, "ingest_chunk", {"video_id": "crash-vid", "step": "chunk", "entity_id": 0}
    )

    def boom(c, chunks):
        raise RuntimeError("simulated upsert failure")

    monkeypatch.setattr(handlers.chunks_repo, "upsert", boom)

    with pytest.raises(RuntimeError, match="simulated upsert failure"):
        with conn.transaction():
            workers.process_one(conn, "ingest_chunk", handlers.chunk_handler)

    chunk_n = conn.execute(
        "SELECT count(*) FROM chunks WHERE video_id = 'crash-vid'"
    ).fetchone()[0]
    assert chunk_n == 0

    status_row = conn.execute(
        "SELECT status, attempts FROM ingest_step_status "
        "WHERE video_id = 'crash-vid' AND step = 'chunk'"
    ).fetchone()
    assert status_row == ("pending", 0)

    msgs = pgmq_client.read(conn, "ingest_chunk", vt=30, qty=1)
    assert len(msgs) == 1
    assert msgs[0].message["video_id"] == "crash-vid"


def test_sparsevec_indexing_handles_token_id_0_and_max(conn, tmp_path):
    """Round-trip the boundary token ids: 0 must store at sparsevec index 1,
    VOCAB_SIZE - 1 must store at sparsevec index VOCAB_SIZE. This is the
    regression test for the +1 shift in embeds.upsert_sparse — without it,
    250001 would exceed dim and pgvector would raise."""
    vtt = _make_vtt(tmp_path)
    upsert_talk(conn, _make_talk("sparse-vid", vtt))
    [chunk_id] = chunks_repo.upsert(
        conn,
        [
            Chunk(
                video_id="sparse-vid",
                start_sec=0.0,
                end_sec=5.0,
                text="boundary token test",
                chunking_strategy="fixed_window",
                frame_secs=[],
                token_count=3,
            )
        ],
    )

    embeds_repo.upsert_sparse(conn, chunk_id, {0: 0.9, VOCAB_SIZE - 1: 0.4})

    text = conn.execute(
        "SELECT embedding::text FROM sparse_embeds WHERE chunk_id = %s",
        (chunk_id,),
    ).fetchone()[0]
    assert "1:" in text
    assert f"{VOCAB_SIZE}:" in text


# -- frames_handler --------------------------------------------------------


def _ffmpeg_present() -> bool:
    return shutil.which("ffmpeg") is not None


@pytest.fixture(scope="module")
def synth_video(tmp_path_factory):
    """60s deterministic testsrc video — feeds every frames_handler test below.

    Pinned at module scope so the encode happens once; ~1-2s on this hardware.
    Skipped cleanly if ffmpeg isn't on PATH so CI hosts without the binary
    report 'skipped' instead of failing the suite.
    """
    if not _ffmpeg_present():
        pytest.skip("ffmpeg not on PATH")
    out = tmp_path_factory.mktemp("frames-handler") / "synth_60s.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-fflags",
            "+bitexact",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=60:size=320x240:rate=30",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-flags",
            "+bitexact",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


def _make_frames_talk(video_id: str, *, duration_sec: int = 60) -> Talk:
    return Talk(
        video_id=video_id,
        title="frames handler test",
        speaker="S",
        url="https://example.com/v",
        duration_sec=duration_sec,
        format_tags=["slides_heavy"],
        license="cc-by",
        captions_source="manual_transcript",
        transcript_path=f"transcripts/ai_engineering_v0/{video_id}.vtt",
        transcript_sha256="a" + "0" * 63,
        accessed_at=datetime(2025, 1, 1, tzinfo=UTC),
    )


def test_frames_handler_end_to_end_writes_frames_status_and_enqueues_embed_jobs(
    conn, tmp_path, monkeypatch, synth_video
):
    """Seed talk + frame_sample status + ingest_frames msg. Run process_one.
    Confirm: frames table populated, embed_frames status rows per frame,
    ingest_embed_frames queue carries matching payloads. ffmpeg runs for real."""
    upsert_talk(conn, _make_frames_talk("fr-vid", duration_sec=60))
    iss.upsert(conn, "fr-vid", step="frame_sample", status="pending")
    pgmq_client.send(
        conn,
        "ingest_frames",
        {"video_id": "fr-vid", "step": "frame_sample", "entity_id": 0},
    )

    monkeypatch.setattr(
        handlers.frames_mod,
        "default_video_path_for_talk",
        lambda t: synth_video,
    )
    # Pin frame on-disk output under tmp_path so the test doesn't write
    # under the repo's frames/ dir.
    orig_sample = frames_mod.sample

    def hermetic_sample(video_path, **kwargs):
        kwargs.setdefault("out_dir", tmp_path / kwargs["video_id"])
        return orig_sample(video_path, **kwargs)

    monkeypatch.setattr(handlers.frames_mod, "sample", hermetic_sample)

    with conn.transaction():
        processed = workers.process_one(conn, "ingest_frames", handlers.frames_handler)
    assert processed is True

    frame_rows = conn.execute(
        "SELECT frame_id, frame_sec, sha256 FROM frames "
        "WHERE video_id = 'fr-vid' ORDER BY frame_sec"
    ).fetchall()
    assert len(frame_rows) == 6
    assert [r[1] for r in frame_rows] == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    assert all(len(r[2]) == 64 for r in frame_rows)
    frame_ids = [r[0] for r in frame_rows]

    embed_status_ids = sorted(
        r[0]
        for r in conn.execute(
            "SELECT entity_id FROM ingest_step_status "
            "WHERE video_id = 'fr-vid' AND step = 'embed_frames'"
        ).fetchall()
    )
    assert embed_status_ids == sorted(frame_ids)

    msgs = pgmq_client.read(conn, "ingest_embed_frames", vt=30, qty=len(frame_ids))
    assert len(msgs) == len(frame_ids)
    assert sorted(m.message["entity_id"] for m in msgs) == sorted(frame_ids)
    assert all(m.message["video_id"] == "fr-vid" for m in msgs)
    assert all(m.message["step"] == "embed_frames" for m in msgs)

    # pooled_embedding stays NULL — slice 2 owns that column.
    nulls = conn.execute(
        "SELECT count(*) FROM frames "
        "WHERE video_id = 'fr-vid' AND pooled_embedding IS NULL"
    ).fetchone()[0]
    assert nulls == 6


def test_frames_handler_retry_does_not_duplicate_frames(
    conn, tmp_path, monkeypatch, synth_video
):
    """Re-running frames_handler for the same payload returns the same frame_ids
    without inserting duplicates. Queue is NOT idempotent — the embed_frames
    handler's claim_for_update no-ops on its own redelivery (verified in
    test_pgmq_smoke for chunk_handler's mirror)."""
    upsert_talk(conn, _make_frames_talk("retry-fr", duration_sec=30))
    iss.upsert(conn, "retry-fr", step="frame_sample", status="pending")

    monkeypatch.setattr(
        handlers.frames_mod,
        "default_video_path_for_talk",
        lambda t: synth_video,
    )
    orig_sample = frames_mod.sample

    def hermetic_sample(video_path, **kwargs):
        kwargs.setdefault("out_dir", tmp_path / kwargs["video_id"])
        return orig_sample(video_path, **kwargs)

    monkeypatch.setattr(handlers.frames_mod, "sample", hermetic_sample)

    with conn.transaction():
        handlers.frames_handler(
            conn,
            {"video_id": "retry-fr", "step": "frame_sample", "entity_id": 0},
        )

    first_ids = [
        r[0]
        for r in conn.execute(
            "SELECT frame_id FROM frames WHERE video_id = 'retry-fr' "
            "ORDER BY frame_sec"
        ).fetchall()
    ]
    assert len(first_ids) >= 1

    with conn.transaction():
        handlers.frames_handler(
            conn,
            {"video_id": "retry-fr", "step": "frame_sample", "entity_id": 0},
        )

    second_ids = [
        r[0]
        for r in conn.execute(
            "SELECT frame_id FROM frames WHERE video_id = 'retry-fr' "
            "ORDER BY frame_sec"
        ).fetchall()
    ]
    assert second_ids == first_ids


def test_frames_handler_crash_rolls_back_frames_status_and_message(
    conn, tmp_path, monkeypatch, synth_video
):
    """If frames_repo.upsert raises, process_one's caller transaction rolls
    back: no frames row, status reverted to pending/0 attempts, the message
    redelivers."""
    upsert_talk(conn, _make_frames_talk("crash-fr", duration_sec=30))
    iss.upsert(conn, "crash-fr", step="frame_sample", status="pending")
    pgmq_client.send(
        conn,
        "ingest_frames",
        {"video_id": "crash-fr", "step": "frame_sample", "entity_id": 0},
    )

    monkeypatch.setattr(
        handlers.frames_mod,
        "default_video_path_for_talk",
        lambda t: synth_video,
    )
    orig_sample = frames_mod.sample

    def hermetic_sample(video_path, **kwargs):
        kwargs.setdefault("out_dir", tmp_path / kwargs["video_id"])
        return orig_sample(video_path, **kwargs)

    monkeypatch.setattr(handlers.frames_mod, "sample", hermetic_sample)

    def boom(c, samples):
        raise RuntimeError("simulated frames upsert failure")

    monkeypatch.setattr(handlers.frames_repo, "upsert", boom)

    with pytest.raises(RuntimeError, match="simulated frames upsert failure"):
        with conn.transaction():
            workers.process_one(conn, "ingest_frames", handlers.frames_handler)

    n = conn.execute(
        "SELECT count(*) FROM frames WHERE video_id = 'crash-fr'"
    ).fetchone()[0]
    assert n == 0

    status_row = conn.execute(
        "SELECT status, attempts FROM ingest_step_status "
        "WHERE video_id = 'crash-fr' AND step = 'frame_sample'"
    ).fetchone()
    assert status_row == ("pending", 0)

    msgs = pgmq_client.read(conn, "ingest_frames", vt=30, qty=1)
    assert len(msgs) == 1
    assert msgs[0].message["video_id"] == "crash-fr"
