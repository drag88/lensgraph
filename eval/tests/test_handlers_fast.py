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


def _stub_talk_for_frames(*, source_video_id: str | None = None, duration_sec: int = 120) -> Talk:
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
    assert iss_calls == [("vid-fr", "embed_frames", fid, "pending") for fid in (501, 502, 503)]

    assert len(send_calls) == 1
    queue, payloads = send_calls[0]
    assert queue == "ingest_embed_frames"
    assert payloads == [
        {"video_id": "vid-fr", "step": "embed_frames", "entity_id": fid} for fid in (501, 502, 503)
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


# -- embed_frames_handler -------------------------------------------------


class _FakeImage:
    """Minimal PIL.Image stand-in for the with-block lifecycle. Only
    ``load`` and the context-manager protocol are exercised."""

    def load(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _stub_patches_3d(p: int = 4, dim: int = 128):
    """Return a deterministic (1, p, dim) array — emulates
    encode_image_patches([img]) without loading ColQwen."""
    return np.stack(
        [np.stack([np.full((dim,), 0.1 + i * 0.01, dtype=np.float32) for i in range(p)])]
    )


def test_embed_frames_handler_loads_image_encodes_and_writes_pooled_and_patches(
    monkeypatch, tmp_path
):
    """Happy path: get frame row → open image → encode patches → write
    pooled + patches. Pooled vector must be derivable from the patches
    via pool_patches so HNSW and MaxSim see the same content."""
    img_path = tmp_path / "f.png"
    img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)

    monkeypatch.setattr(
        h.frames_repo,
        "get",
        lambda conn, frame_id, video_id: (str(img_path), 10.0),
    )

    # Capture-only fake for PIL.Image.open — the handler's `with` block
    # exercises __enter__/__exit__/load. We don't want to actually decode.
    from PIL import Image

    opened: dict = {}

    def fake_open(path):
        opened["path"] = path
        return _FakeImage()

    monkeypatch.setattr(Image, "open", fake_open)

    patches_3d = _stub_patches_3d(p=4)
    encode_calls: list = []

    def fake_encode_image_patches(images):
        encode_calls.append(images)
        return patches_3d

    pool_calls: list = []

    def fake_pool_patches(arr):
        pool_calls.append(arr)
        return arr.mean(axis=0).astype(np.float32)

    monkeypatch.setattr(h.colqwen, "encode_image_patches", fake_encode_image_patches)
    monkeypatch.setattr(h.colqwen, "pool_patches", fake_pool_patches)

    update_calls: list = []
    replace_calls: list = []
    monkeypatch.setattr(
        h.frames_repo,
        "update_pooled",
        lambda c, fid, vec: update_calls.append((fid, vec)),
    )
    monkeypatch.setattr(
        h.frames_repo,
        "replace_patches",
        lambda c, fid, patches: replace_calls.append((fid, patches)),
    )

    h.embed_frames_handler(
        conn=None,
        payload={"video_id": "vid-e", "step": "embed_frames", "entity_id": 77},
    )

    assert opened["path"] == img_path
    assert len(encode_calls) == 1 and len(encode_calls[0]) == 1  # one image
    assert len(pool_calls) == 1
    np.testing.assert_allclose(pool_calls[0], patches_3d[0])

    assert len(update_calls) == 1
    assert update_calls[0][0] == 77
    assert update_calls[0][1].shape == (128,)
    assert update_calls[0][1].dtype == np.float32

    assert len(replace_calls) == 1
    assert replace_calls[0][0] == 77
    np.testing.assert_allclose(replace_calls[0][1], patches_3d[0])


def test_embed_frames_handler_raises_on_missing_frame_row(monkeypatch):
    """Contract: a missing frame row is a hard failure (cross-video lookup
    mismatch, or row deleted out from under the worker). process_one rolls
    back and the message redelivers."""
    monkeypatch.setattr(h.frames_repo, "get", lambda c, frame_id, video_id: None)

    with pytest.raises(ValueError, match="no frames row"):
        h.embed_frames_handler(
            conn=None,
            payload={"video_id": "missing-vid", "step": "embed_frames", "entity_id": 999},
        )


def test_embed_frames_handler_raises_on_missing_image_file(monkeypatch, tmp_path):
    """Missing image file on disk is a hard failure — operator must
    re-stage the source video (frames sampling regenerates the PNGs)."""
    monkeypatch.setattr(
        h.frames_repo,
        "get",
        lambda c, frame_id, video_id: (str(tmp_path / "nope.png"), 0.0),
    )

    with pytest.raises(FileNotFoundError, match="frame image not found"):
        h.embed_frames_handler(
            conn=None,
            payload={"video_id": "vid-e", "step": "embed_frames", "entity_id": 1},
        )


def _wire_embed_frames_image(monkeypatch, tmp_path, *, frame_id=88):
    """Shared setup for the NaN/inf skip tests: a real-enough PNG path, a
    capture-only PIL.Image.open, and frame_id resolution. Returns nothing —
    callers monkeypatch encode_image_patches / pool_patches themselves."""
    img_path = tmp_path / "f.png"
    img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
    monkeypatch.setattr(
        h.frames_repo,
        "get",
        lambda conn, frame_id, video_id: (str(img_path), 10.0),
    )
    from PIL import Image

    monkeypatch.setattr(Image, "open", lambda path: _FakeImage())


def _capture_writes(monkeypatch):
    """Record calls to the three frames_repo write paths the skip must guard.
    Returns (clear_calls, update_calls, replace_calls)."""
    clear_calls: list = []
    update_calls: list = []
    replace_calls: list = []
    monkeypatch.setattr(h.frames_repo, "clear_embeddings", lambda c, fid: clear_calls.append(fid))
    monkeypatch.setattr(
        h.frames_repo, "update_pooled", lambda c, fid, vec: update_calls.append(fid)
    )
    monkeypatch.setattr(
        h.frames_repo, "replace_patches", lambda c, fid, p: replace_calls.append(fid)
    )
    return clear_calls, update_calls, replace_calls


def test_embed_frames_handler_nonfinite_patches_clear_and_skip(monkeypatch, tmp_path):
    """A re-embed that yields NaN/inf patches must NOT write pooled/patches,
    and must clear any stale vectors a prior good embedding left on the row.
    Otherwise the frame keeps serving a stale pooled vector while the report
    claims it carries none."""
    _wire_embed_frames_image(monkeypatch, tmp_path)
    nan_patches = np.full((1, 4, 128), np.nan, dtype=np.float32)
    monkeypatch.setattr(h.colqwen, "encode_image_patches", lambda imgs: nan_patches)

    pool_called: list = []
    monkeypatch.setattr(h.colqwen, "pool_patches", lambda a: pool_called.append(a))

    clear_calls, update_calls, replace_calls = _capture_writes(monkeypatch)

    h.embed_frames_handler(
        conn=None,
        payload={"video_id": "vid-e", "step": "embed_frames", "entity_id": 88},
    )

    assert clear_calls == [88]  # stale rows cleared
    assert update_calls == [] and replace_calls == []  # no new rows written
    assert pool_called == []  # short-circuits before pooling


def test_embed_frames_handler_nonfinite_pooled_clear_and_skip(monkeypatch, tmp_path):
    """Patches are finite but the pooled vector comes out non-finite — same
    contract: clear stale rows, write nothing new."""
    _wire_embed_frames_image(monkeypatch, tmp_path)
    finite_patches = _stub_patches_3d(p=4)
    monkeypatch.setattr(h.colqwen, "encode_image_patches", lambda imgs: finite_patches)
    monkeypatch.setattr(
        h.colqwen, "pool_patches", lambda a: np.full((128,), np.inf, dtype=np.float32)
    )

    clear_calls, update_calls, replace_calls = _capture_writes(monkeypatch)

    h.embed_frames_handler(
        conn=None,
        payload={"video_id": "vid-e", "step": "embed_frames", "entity_id": 88},
    )

    assert clear_calls == [88]
    assert update_calls == [] and replace_calls == []


def test_embed_frames_handler_finite_path_does_not_clear(monkeypatch, tmp_path):
    """The happy path must not call clear_embeddings — it writes fresh rows
    via update_pooled + replace_patches, which already overwrite cleanly."""
    _wire_embed_frames_image(monkeypatch, tmp_path)
    finite_patches = _stub_patches_3d(p=4)
    monkeypatch.setattr(h.colqwen, "encode_image_patches", lambda imgs: finite_patches)
    monkeypatch.setattr(h.colqwen, "pool_patches", lambda a: a.mean(axis=0).astype(np.float32))

    clear_calls, update_calls, replace_calls = _capture_writes(monkeypatch)

    h.embed_frames_handler(
        conn=None,
        payload={"video_id": "vid-e", "step": "embed_frames", "entity_id": 88},
    )

    assert clear_calls == []
    assert update_calls == [88] and replace_calls == [88]
