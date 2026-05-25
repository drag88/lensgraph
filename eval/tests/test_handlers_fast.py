"""Fast handler tests — no DB, no model load.

The handler module imports submodules (talks_repo, chunks_repo, embeds_repo,
iss, pgmq_client, bge_m3) by attribute, so monkeypatching the attribute on
those modules from inside `ingest.handlers`'s namespace reaches the same
object the handler will look up. Concretely:

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
