"""Fast handler tests — no DB, no model load, no ffmpeg.

The handler module imports submodules (talks_repo, chunks_repo, embeds_repo,
frames_repo, frames_mod, iss, pgmq_client, bge_m3) by attribute, so
monkeypatching the attribute on those modules from inside
`ingest.handlers`'s namespace reaches the same object the handler will
look up. Concretely:

    import ingest.handlers as h
    monkeypatch.setattr(h.chunks_repo, "upsert", fake)

works because `h.chunks_repo` is the live `db.repos.chunks` module — the
patch hits the module-level function the handler reads at call time.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

import ingest.handlers as h
from db.repos.talks import Talk
from embed.bge_m3 import DENSE_DIM
from ingest.frames import FrameSample


def _stub_talk(transcript_path: Path) -> Talk:
    return Talk(
        video_id="v",
        title="Stub",
        speaker="S",
        url="https://example.com/v",
        duration_sec=60,
        format_tags=["narrative"],
        license="cc-by",
        captions_source="manual_transcript",
        transcript_path=str(transcript_path),
        transcript_sha256="a" + "0" * 63,
        accessed_at=datetime(2025, 1, 1, tzinfo=UTC),
    )


def _stub_vtt(tmp_path: Path) -> Path:
    """Two short cues — enough for fixed_window to emit at least one chunk
    but the chunker output is monkeypatched away anyway."""
    p = tmp_path / "v.vtt"
    p.write_text(
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:05.000\nhello world\n\n"
        "00:00:05.000 --> 00:00:10.000\ngoodbye world\n",
        encoding="utf-8",
    )
    return p


def test_chunk_handler_invokes_chunker_persists_and_fans_out(tmp_path, monkeypatch):
    """chunk_handler runs the chunker, persists via chunks_repo.upsert, then
    upserts an embed_text status row per chunk and enqueues a matching
    payload batch. The status row must land BEFORE the message so the embed
    handler's claim_for_update finds its row on the first delivery."""
    transcript = _stub_vtt(tmp_path)
    talk = _stub_talk(transcript)

    monkeypatch.setattr(h.talks_repo, "get", lambda c, vid: talk)
    monkeypatch.setattr(h.chunks_repo, "upsert", lambda c, chunks: [101, 102, 103])

    iss_calls: list[tuple[str, str, int, str]] = []

    def fake_iss_upsert(conn, video_id, *, step, entity_id=0, status="pending"):
        iss_calls.append((video_id, step, entity_id, status))

    monkeypatch.setattr(h.iss, "upsert", fake_iss_upsert)

    send_calls: list[tuple[str, list[dict]]] = []

    def fake_send_batch(conn, queue, payloads):
        send_calls.append((queue, list(payloads)))
        return [9001, 9002, 9003]

    monkeypatch.setattr(h.pgmq_client, "send_batch", fake_send_batch)

    h.chunk_handler(
        conn=None,  # fake — handler never calls conn directly under monkeypatch
        payload={"video_id": "v", "step": "chunk", "entity_id": 0},
    )

    assert len(iss_calls) == 3
    assert [c[2] for c in iss_calls] == [101, 102, 103]
    expected = [("v", "embed_text", cid, "pending") for cid in (101, 102, 103)]
    assert iss_calls == expected

    assert len(send_calls) == 1
    queue, payloads = send_calls[0]
    assert queue == "ingest_embed_text"
    assert payloads == [
        {"video_id": "v", "step": "embed_text", "entity_id": 101},
        {"video_id": "v", "step": "embed_text", "entity_id": 102},
        {"video_id": "v", "step": "embed_text", "entity_id": 103},
    ]


def test_embed_text_handler_writes_three_embeds_for_chunk_id(monkeypatch):
    """embed_text_handler loads one chunk's text, encodes it once, and writes
    all three embed channels for the same chunk_id."""

    monkeypatch.setattr(
        h.chunks_repo,
        "get_text",
        lambda c, chunk_id, video_id: "the cat sat on the mat",
    )

    fake_dense = np.zeros((1, DENSE_DIM), dtype=np.float32)
    fake_sparse = [{1: 0.5}]
    fake_multi = [np.zeros((4, DENSE_DIM), dtype=np.float32)]
    monkeypatch.setattr(
        h.bge_m3,
        "encode",
        lambda sentences: h.bge_m3.BgeM3Output(
            dense=fake_dense, sparse=fake_sparse, multi=fake_multi
        ),
    )

    dense_calls: list[tuple[int, np.ndarray]] = []
    sparse_calls: list[tuple[int, dict[int, float]]] = []
    multi_calls: list[tuple[int, np.ndarray]] = []

    monkeypatch.setattr(
        h.embeds_repo,
        "upsert_dense",
        lambda c, cid, v: dense_calls.append((cid, v)),
    )
    monkeypatch.setattr(
        h.embeds_repo,
        "upsert_sparse",
        lambda c, cid, s: sparse_calls.append((cid, s)),
    )
    monkeypatch.setattr(
        h.embeds_repo,
        "replace_token_embeds",
        lambda c, cid, v: multi_calls.append((cid, v)),
    )

    h.embed_text_handler(
        conn=None,
        payload={"video_id": "v", "step": "embed_text", "entity_id": 42},
    )

    assert len(dense_calls) == 1 and dense_calls[0][0] == 42
    assert dense_calls[0][1].shape == (DENSE_DIM,)
    assert sparse_calls == [(42, {1: 0.5})]
    assert len(multi_calls) == 1 and multi_calls[0][0] == 42
    assert multi_calls[0][1].shape == (4, DENSE_DIM)


def test_chunk_handler_raises_on_missing_talks_row(monkeypatch):
    """Contract: a missing talks row is a hard failure. process_one wraps
    the handler in the caller's transaction — raising rolls back the dequeue
    and PGMQ redelivers when (eventually) the talks row exists."""
    monkeypatch.setattr(h.talks_repo, "get", lambda c, vid: None)

    with pytest.raises(ValueError, match="no talks row"):
        h.chunk_handler(
            conn=None,
            payload={"video_id": "missing-vid", "step": "chunk", "entity_id": 0},
        )


# -- frames_handler ------------------------------------------------------


def _stub_talk_for_frames(
    *, source_video_id: str | None = None, duration_sec: int = 120
) -> Talk:
    return Talk(
        video_id="vid-fr",
        title="frames stub",
        speaker="S",
        url="https://example.com/v",
        duration_sec=duration_sec,
        format_tags=["slides_heavy"],
        license="cc-by",
        captions_source="manual_transcript",
        transcript_path="transcripts/ai_engineering_v0/vid-fr.vtt",
        transcript_sha256="a" + "0" * 63,
        accessed_at=datetime(2025, 1, 1, tzinfo=UTC),
        source_video_id=source_video_id,
        source_start_sec=300.0 if source_video_id else None,
        source_end_sec=420.0 if source_video_id else None,
    )


class _FakePath:
    """Minimal Path stand-in — only `exists()` is consulted by the handler."""

    def __init__(self, path_str: str, *, exists: bool = True):
        self._s = path_str
        self._exists = exists

    def exists(self) -> bool:
        return self._exists

    def __str__(self) -> str:
        return self._s


def test_frames_handler_samples_persists_and_fans_out_embed_frames(
    monkeypatch,
):
    """Handler: sample → upsert frames → write embed_frames status rows →
    enqueue ingest_embed_frames messages. Status row + message order must
    match the chunk_handler convention (status first, then send_batch) so
    the slice-2 embed_frames_handler's claim_for_update finds its row."""
    talk = _stub_talk_for_frames(duration_sec=60)
    monkeypatch.setattr(h.talks_repo, "get", lambda c, vid: talk)
    monkeypatch.setattr(
        h.frames_mod,
        "default_video_path_for_talk",
        lambda t: _FakePath("/fake/vid-fr.mp4", exists=True),
    )

    sample_args: dict = {}

    def fake_sample(video_path, *, video_id, every_sec, start_sec, duration_sec, out_dir=None):
        sample_args["video_path"] = video_path
        sample_args["video_id"] = video_id
        sample_args["every_sec"] = every_sec
        sample_args["start_sec"] = start_sec
        sample_args["duration_sec"] = duration_sec
        return [
            FrameSample(
                video_id=video_id,
                frame_sec=float(i) * every_sec,
                image_path=f"/fake/frames/{i:06d}.png",
                sha256=f"{i:064d}",
            )
            for i in range(3)
        ]

    monkeypatch.setattr(h.frames_mod, "sample", fake_sample)
    monkeypatch.setattr(h.frames_repo, "upsert", lambda c, samples: [501, 502, 503])

    iss_calls: list[tuple[str, str, int, str]] = []

    def fake_iss_upsert(conn, video_id, *, step, entity_id=0, status="pending"):
        iss_calls.append((video_id, step, entity_id, status))

    monkeypatch.setattr(h.iss, "upsert", fake_iss_upsert)

    send_calls: list[tuple[str, list[dict]]] = []

    def fake_send_batch(conn, queue, payloads):
        send_calls.append((queue, list(payloads)))
        return [9101, 9102, 9103]

    monkeypatch.setattr(h.pgmq_client, "send_batch", fake_send_batch)

    h.frames_handler(
        conn=None,
        payload={"video_id": "vid-fr", "step": "frame_sample", "entity_id": 0},
    )

    # Regular talk → sample the full [0, duration_sec) window.
    assert sample_args["start_sec"] == 0.0
    assert sample_args["duration_sec"] == 60.0
    assert sample_args["every_sec"] == 10.0
    assert sample_args["video_id"] == "vid-fr"

    assert [c[2] for c in iss_calls] == [501, 502, 503]
    assert iss_calls == [
        ("vid-fr", "embed_frames", fid, "pending") for fid in (501, 502, 503)
    ]

    assert len(send_calls) == 1
    queue, payloads = send_calls[0]
    assert queue == "ingest_embed_frames"
    assert payloads == [
        {"video_id": "vid-fr", "step": "embed_frames", "entity_id": fid}
        for fid in (501, 502, 503)
    ]


def test_frames_handler_chapter_slice_passes_source_window_to_sampler(
    monkeypatch,
):
    """For a chapter slice, sample() must be called with start_sec=source_start_sec
    and duration_sec=source_end_sec - source_start_sec — the *parent* video's
    coordinates, not the talk's own."""
    talk = _stub_talk_for_frames(source_video_id="PARENT_ID", duration_sec=120)
    monkeypatch.setattr(h.talks_repo, "get", lambda c, vid: talk)
    monkeypatch.setattr(
        h.frames_mod,
        "default_video_path_for_talk",
        lambda t: _FakePath("/fake/PARENT_ID.mp4", exists=True),
    )

    sample_args: dict = {}

    def fake_sample(video_path, *, video_id, every_sec, start_sec, duration_sec, out_dir=None):
        sample_args["start_sec"] = start_sec
        sample_args["duration_sec"] = duration_sec
        return []

    monkeypatch.setattr(h.frames_mod, "sample", fake_sample)
    monkeypatch.setattr(
        h.frames_repo,
        "upsert",
        lambda c, samples: pytest.fail("upsert called for empty samples"),
    )

    h.frames_handler(
        conn=None,
        payload={"video_id": "vid-fr", "step": "frame_sample", "entity_id": 0},
    )

    assert sample_args["start_sec"] == 300.0
    assert sample_args["duration_sec"] == 120.0


def test_frames_handler_raises_on_missing_talks_row(monkeypatch):
    monkeypatch.setattr(h.talks_repo, "get", lambda c, vid: None)
    with pytest.raises(ValueError, match="no talks row"):
        h.frames_handler(
            conn=None,
            payload={"video_id": "missing", "step": "frame_sample", "entity_id": 0},
        )


def test_frames_handler_raises_on_missing_local_video(monkeypatch):
    """Missing .mp4 must be a hard failure — process_one rolls back and the
    operator stages the file before the next ingest run."""
    talk = _stub_talk_for_frames()
    monkeypatch.setattr(h.talks_repo, "get", lambda c, vid: talk)
    monkeypatch.setattr(
        h.frames_mod,
        "default_video_path_for_talk",
        lambda t: _FakePath("/nope/missing.mp4", exists=False),
    )
    with pytest.raises(FileNotFoundError, match="local video not found"):
        h.frames_handler(
            conn=None,
            payload={"video_id": "vid-fr", "step": "frame_sample", "entity_id": 0},
        )


def test_frames_handler_empty_samples_short_circuits(monkeypatch):
    """An empty sample list (e.g. duration too short to fit one window) must
    skip the upsert / iss / send_batch path entirely so we don't enqueue a
    zero-frame ingest stage."""
    talk = _stub_talk_for_frames(duration_sec=60)
    monkeypatch.setattr(h.talks_repo, "get", lambda c, vid: talk)
    monkeypatch.setattr(
        h.frames_mod,
        "default_video_path_for_talk",
        lambda t: _FakePath("/fake/vid-fr.mp4", exists=True),
    )
    monkeypatch.setattr(
        h.frames_mod,
        "sample",
        lambda *a, **kw: [],
    )

    def boom(*a, **kw):  # noqa: ARG001
        raise AssertionError("must not be called when samples are empty")

    monkeypatch.setattr(h.frames_repo, "upsert", boom)
    monkeypatch.setattr(h.iss, "upsert", boom)
    monkeypatch.setattr(h.pgmq_client, "send_batch", boom)

    h.frames_handler(
        conn=None,
        payload={"video_id": "vid-fr", "step": "frame_sample", "entity_id": 0},
    )
