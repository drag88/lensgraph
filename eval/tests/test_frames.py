"""Frame-sampling unit + repo tests against real ffmpeg + real Postgres.

Slow: ffmpeg subprocess for the synth lavfi video (per-module fixture) and
a live Postgres instance for the repo idempotency assertions. Skipped
cleanly if ffmpeg isn't on PATH (CI hosts without the binary should not
fail; they should report 'skipped').

Coverage:
  - sample() yields N frames for a known cadence and stable sha256 across
    re-runs (PNG + bitexact ffmpeg flags make this byte-identical).
  - sample() chapter-slice path returns frame_sec aligned to the talk's
    zero, not the parent video's absolute time.
  - default_video_path_for_talk picks source_video_id for chapter slices
    and video_id otherwise, under videos/<corpus>/.
  - frames_repo.upsert returns ids aligned to input order, is idempotent
    on (video_id, frame_sec), and never touches pooled_embedding (slice 2
    owns that column).
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import psycopg
import pytest

from db.conn import dsn as resolve_dsn
from db.migrate import apply
from db.repos import frames as frames_repo
from db.repos.talks import Talk
from db.repos.talks import upsert as upsert_talk
from ingest.frames import (
    FfmpegNotAvailableError,
    FrameSample,
    default_video_path_for_talk,
    sample,
)

pytestmark = pytest.mark.slow

TEST_DB_NAME = "lensgraph_test_frames"


def _swap_db(dsn: str, new_db: str) -> str:
    head, _, _ = dsn.rpartition("/")
    return f"{head}/{new_db}"


def _ffmpeg_present() -> bool:
    return shutil.which("ffmpeg") is not None


@pytest.fixture(scope="module")
def synth_video(tmp_path_factory):
    """60s deterministic testsrc video — used by every sampler test below.

    `testsrc` produces a time-varying pattern (counters + colour bars), so
    frames at different positions have distinct content and the chapter-
    slice test can prove start_sec actually changed the input window.
    libx264 ultrafast + bitexact flags make the encode reproducible.
    """
    if not _ffmpeg_present():
        pytest.skip("ffmpeg not on PATH")
    out = tmp_path_factory.mktemp("synth") / "synth_60s.mp4"
    cmd = [
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
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


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
    with psycopg.connect(test_db, autocommit=True) as c:
        c.execute("TRUNCATE frame_patches, frames, talks CASCADE")
        yield c


def _make_talk(video_id: str, *, source_video_id: str | None = None) -> Talk:
    return Talk(
        video_id=video_id,
        title="Frames test",
        speaker="S",
        url="https://example.com/v",
        duration_sec=60,
        format_tags=["slides_heavy"],
        license="cc-by",
        captions_source="manual_transcript",
        transcript_path=f"transcripts/ai_engineering_v0/{video_id}.vtt",
        transcript_sha256="a" + "0" * 63,
        accessed_at=datetime(2025, 1, 1, tzinfo=UTC),
        source_video_id=source_video_id,
        source_start_sec=60.0 if source_video_id else None,
        source_end_sec=120.0 if source_video_id else None,
    )


# -- sample() unit tests --------------------------------------------------


def test_sample_yields_six_frames_for_60s_at_10s_cadence(synth_video, tmp_path):
    out_dir = tmp_path / "ten"
    samples = sample(
        synth_video,
        video_id="vid-ten",
        every_sec=10.0,
        duration_sec=60.0,
        out_dir=out_dir,
    )
    assert [s.frame_sec for s in samples] == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    assert all(s.video_id == "vid-ten" for s in samples)
    assert all(Path(s.image_path).exists() for s in samples)
    assert all(len(s.sha256) == 64 for s in samples)
    # testsrc content varies across time, so each frame's sha256 should be
    # distinct — a regression where every frame is "frame 0" would surface here.
    assert len({s.sha256 for s in samples}) == len(samples)


def test_sample_sha256_stable_across_reruns(synth_video, tmp_path):
    out_dir = tmp_path / "stable"
    first = sample(
        synth_video,
        video_id="vid-stab",
        every_sec=10.0,
        duration_sec=60.0,
        out_dir=out_dir,
    )
    second = sample(
        synth_video,
        video_id="vid-stab",
        every_sec=10.0,
        duration_sec=60.0,
        out_dir=out_dir,
    )
    assert [s.sha256 for s in first] == [s.sha256 for s in second]


def test_sample_chapter_slice_frame_sec_relative_to_talk(synth_video, tmp_path):
    """Sampling [30s, 60s) of the parent returns frame_sec=0,10,20 — aligned
    to the talk's zero. Content differs from start_sec=0 sampling (testsrc
    is time-varying), proving start_sec actually shifted the input window."""
    head = sample(
        synth_video,
        video_id="head",
        every_sec=10.0,
        start_sec=0.0,
        duration_sec=30.0,
        out_dir=tmp_path / "head",
    )
    tail = sample(
        synth_video,
        video_id="tail",
        every_sec=10.0,
        start_sec=30.0,
        duration_sec=30.0,
        out_dir=tmp_path / "tail",
    )
    assert [s.frame_sec for s in tail] == [0.0, 10.0, 20.0]
    # frame_sec is identical (relative), but the content (sha256) differs
    # because we sampled a later window of the time-varying source.
    assert [s.frame_sec for s in head] == [s.frame_sec for s in tail]
    assert [s.sha256 for s in head] != [s.sha256 for s in tail]


def test_sample_raises_when_video_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="video not found"):
        sample(
            tmp_path / "does-not-exist.mp4",
            video_id="v",
            every_sec=10.0,
            duration_sec=60.0,
        )


def test_sample_raises_when_every_sec_non_positive(synth_video, tmp_path):
    with pytest.raises(ValueError, match="every_sec must be > 0"):
        sample(
            synth_video,
            video_id="v",
            every_sec=0.0,
            duration_sec=60.0,
            out_dir=tmp_path / "x",
        )


def test_sample_raises_when_ffmpeg_missing(synth_video, tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _name: None)
    with pytest.raises(FfmpegNotAvailableError, match="ffmpeg not on PATH"):
        sample(
            synth_video,
            video_id="v",
            every_sec=10.0,
            duration_sec=60.0,
            out_dir=tmp_path / "x",
        )


def test_sample_purges_stale_frames_in_target_dir(synth_video, tmp_path):
    """A leftover frame_*.png from an earlier longer sample must not appear
    in the new sha256 list — otherwise audits would mis-attribute frames."""
    out_dir = tmp_path / "purge"
    out_dir.mkdir()
    stale = out_dir / "frame_999999.png"
    stale.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    samples = sample(
        synth_video,
        video_id="purge-vid",
        every_sec=10.0,
        duration_sec=30.0,
        out_dir=out_dir,
    )
    written_paths = {Path(s.image_path) for s in samples}
    assert stale not in written_paths
    assert not stale.exists()


# -- default_video_path_for_talk --------------------------------------------


def test_default_video_path_uses_video_id_for_regular_talk():
    talk = _make_talk("Z123")
    p = default_video_path_for_talk(talk)
    assert p.name == "Z123.mp4"
    assert "videos/ai_engineering_v0" in p.as_posix()


def test_default_video_path_uses_source_video_id_for_chapter_slice():
    talk = _make_talk("aie_sg_day2", source_video_id="m12vGjfbNlo")
    p = default_video_path_for_talk(talk)
    assert p.name == "m12vGjfbNlo.mp4"
    assert "videos/ai_engineering_v0" in p.as_posix()


# -- frames_repo.upsert idempotency -----------------------------------------


def _stub_samples(video_id: str, n: int = 3) -> list[FrameSample]:
    return [
        FrameSample(
            video_id=video_id,
            frame_sec=float(i * 10),
            image_path=f"/tmp/{video_id}/frame_{i:06d}.png",
            sha256=f"{i:064d}",
        )
        for i in range(n)
    ]


def test_upsert_returns_aligned_frame_ids(conn):
    upsert_talk(conn, _make_talk("up-1"))
    ids = frames_repo.upsert(conn, _stub_samples("up-1", n=3))
    assert len(ids) == 3
    assert all(isinstance(i, int) and i > 0 for i in ids)
    # Confirm rows landed and the returned ids round-trip.
    rows = conn.execute(
        "SELECT frame_id, frame_sec FROM frames "
        "WHERE video_id = 'up-1' ORDER BY frame_sec"
    ).fetchall()
    assert [r[0] for r in rows] == ids
    assert [r[1] for r in rows] == [0.0, 10.0, 20.0]


def test_upsert_is_idempotent_on_video_id_frame_sec(conn):
    upsert_talk(conn, _make_talk("idem"))
    samples = _stub_samples("idem", n=3)
    first = frames_repo.upsert(conn, samples)
    second = frames_repo.upsert(conn, samples)
    assert first == second
    n = conn.execute("SELECT count(*) FROM frames WHERE video_id = 'idem'").fetchone()[0]
    assert n == 3


def test_upsert_refreshes_image_path_and_sha_but_preserves_pooled_embedding(
    conn,
):
    """Slice 2 writes pooled_embedding asynchronously. A re-run of slice 1's
    upsert with a renamed file path must NOT zero out the embedding —
    otherwise downstream retrieval silently loses the visual channel."""
    upsert_talk(conn, _make_talk("preserve"))
    [fid] = frames_repo.upsert(
        conn,
        [
            FrameSample(
                video_id="preserve",
                frame_sec=0.0,
                image_path="/tmp/preserve/old.png",
                sha256="a" * 64,
            )
        ],
    )
    # Simulate slice 2 populating pooled_embedding out-of-band.
    conn.execute(
        "UPDATE frames SET pooled_embedding = "
        "  ARRAY(SELECT 0.1::real FROM generate_series(1, 128))::vector "
        "WHERE frame_id = %s",
        (fid,),
    )

    # Slice 1 re-runs (e.g. operator re-staged the video).
    frames_repo.upsert(
        conn,
        [
            FrameSample(
                video_id="preserve",
                frame_sec=0.0,
                image_path="/tmp/preserve/new.png",
                sha256="b" * 64,
            )
        ],
    )

    row = conn.execute(
        "SELECT image_path, sha256, pooled_embedding IS NULL "
        "FROM frames WHERE frame_id = %s",
        (fid,),
    ).fetchone()
    assert row[0] == "/tmp/preserve/new.png"
    assert row[1] == "b" * 64
    assert row[2] is False  # pooled_embedding survived the re-upsert


def test_upsert_empty_short_circuits(conn):
    assert frames_repo.upsert(conn, []) == []
