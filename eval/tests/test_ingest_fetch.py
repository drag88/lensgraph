"""Ingest fetch slice — LocalFsFetcher + reconciler.

Slow: requires live Postgres reachable at $POSTGRES_DSN. Per-module
ephemeral DB with migrations + queues pre-created. Per-test TRUNCATE +
purge for isolation. No network / yt-dlp.

The contract proven here:
  - LocalFsFetcher resolves <root>/<video_id>.vtt; missing file raises.
  - fetch_reconcile, given a Talk + Fetcher, writes the talks row with
    fetcher-derived provenance, sets ingest_step_status rows, enqueues
    chunk, and enqueues asr / frames only when their skip-conditions fail.
  - All DB writes + queue sends commit atomically with the caller's
    transaction — a fetcher raise (e.g., FileNotFoundError) rolls back
    everything.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import psycopg
import pytest

from db.conn import dsn as resolve_dsn
from db.migrate import apply
from db.repos import talks as talks_repo
from db.repos.talks import Talk
from ingest import pipeline
from ingest.fetch import LocalFsFetcher, paths_from_talks_yaml
from ingest.quality import probe
from queues import pgmq_client

pytestmark = pytest.mark.slow

TEST_DB_NAME = "lensgraph_test_ingest_fetch"


def _swap_db(dsn: str, new_db: str) -> str:
    head, _, _ = dsn.rpartition("/")
    return f"{head}/{new_db}"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
    with psycopg.connect(test_db, autocommit=True) as c:
        c.execute("TRUNCATE ingest_step_status, talks CASCADE")
        for q in pgmq_client.QUEUES:
            pgmq_client.purge(c, q)
        yield c


@pytest.fixture(scope="module")
def transcript_dir(tmp_path_factory):
    """Pre-stage two VTTs: one passes quality (good), one fails (bad)."""
    d = tmp_path_factory.mktemp("transcripts")

    # 30s talk with ~150 words → 300 wpm, no non-speech markers → passes.
    good = "WEBVTT\n\n00:00:00.000 --> 00:00:30.000\n"
    good += " ".join(["lorem ipsum dolor sit amet consectetur"] * 25) + "\n"
    (d / "good-vid.vtt").write_text(good, encoding="utf-8")

    # 60s talk with one [Music] cue and almost no words → fails on both
    # word-rate AND non-speech-ratio.
    bad = "WEBVTT\n\n00:00:00.000 --> 00:01:00.000\n[Music]\n"
    (d / "bad-vid.vtt").write_text(bad, encoding="utf-8")

    return d


def _make_talk(
    video_id: str,
    *,
    format_tags: list[str] | None = None,
    duration_sec: int = 30,
    title: str = "Test",
) -> Talk:
    return Talk(
        video_id=video_id,
        title=title,
        speaker="Speaker",
        url="https://example.com/v",
        duration_sec=duration_sec,
        format_tags=format_tags if format_tags is not None else ["narrative"],
        license="cc-by",
        captions_source="youtube_auto",  # pre-fetch placeholder; reconciler overwrites
        transcript_path="placeholder",  # reconciler overwrites
        transcript_sha256="a" + "0" * 63,  # reconciler overwrites
        accessed_at=datetime(2025, 1, 1, tzinfo=UTC),
    )


# -- LocalFsFetcher -------------------------------------------------------


def test_local_fetcher_returns_path_sha_and_source(transcript_dir):
    fetcher = LocalFsFetcher(root=transcript_dir, captions_source="youtube_auto")
    result = fetcher.fetch("good-vid")
    assert result.transcript_path == transcript_dir / "good-vid.vtt"
    expected_sha = _sha256_text((transcript_dir / "good-vid.vtt").read_text())
    assert result.transcript_sha256 == expected_sha
    assert result.captions_source == "youtube_auto"


def test_local_fetcher_missing_file_raises(transcript_dir):
    fetcher = LocalFsFetcher(root=transcript_dir)
    with pytest.raises(FileNotFoundError):
        fetcher.fetch("does-not-exist")


# -- reconciler: happy path (the canonical slice assertion) ---------------


def test_local_fetcher_writes_talks_row_and_enqueues_chunk(conn, transcript_dir):
    """Per design §8 step 7 RED: LocalFsFetcher reads pre-staged VTT,
    reconciler writes talks row, enqueues chunk. Plus: asr + frames both
    skipped (quality passes, no visual format_tags)."""
    talk = _make_talk("good-vid", format_tags=["narrative"], duration_sec=30)
    fetcher = LocalFsFetcher(root=transcript_dir, captions_source="youtube_auto")

    with conn.transaction():
        result = pipeline.fetch_reconcile(conn, talk, fetcher)

    # talks row reflects fetcher-derived provenance + ingested_at stamped.
    fetched = talks_repo.get(conn, "good-vid")
    assert fetched is not None
    assert fetched.transcript_path == str(transcript_dir / "good-vid.vtt")
    assert fetched.captions_source == "youtube_auto"
    assert fetched.transcript_sha256 == _sha256_text(
        (transcript_dir / "good-vid.vtt").read_text()
    )
    assert fetched.ingested_at is not None

    # Status rows match the design's step shape.
    statuses = dict(
        conn.execute(
            "SELECT step, status FROM ingest_step_status WHERE video_id='good-vid'"
        ).fetchall()
    )
    assert statuses == {
        "fetch": "completed",
        "asr": "skipped",
        "frame_sample": "skipped",
        "chunk": "pending",
    }

    # chunk queue holds one message; asr + frames queues are empty.
    chunk_msgs = pgmq_client.read(conn, "ingest_chunk", vt=30, qty=1)
    assert len(chunk_msgs) == 1
    assert chunk_msgs[0].message == {"video_id": "good-vid", "step": "chunk"}
    assert chunk_msgs[0].msg_id == result.chunk_msg_id

    assert pgmq_client.read(conn, "ingest_asr", vt=1, qty=1) == []
    assert pgmq_client.read(conn, "ingest_frames", vt=1, qty=1) == []


# -- asr skip vs enqueue --------------------------------------------------


def test_skip_asr_when_caption_quality_passes(conn, transcript_dir):
    talk = _make_talk("good-vid", format_tags=["narrative"], duration_sec=30)
    fetcher = LocalFsFetcher(root=transcript_dir)
    with conn.transaction():
        result = pipeline.fetch_reconcile(conn, talk, fetcher)
    assert result.asr_skipped is True
    assert (
        conn.execute(
            "SELECT status FROM ingest_step_status "
            "WHERE video_id='good-vid' AND step='asr'"
        ).fetchone()[0]
        == "skipped"
    )
    assert pgmq_client.read(conn, "ingest_asr", vt=1, qty=1) == []


def test_enqueue_asr_when_caption_quality_fails(conn, transcript_dir):
    talk = _make_talk("bad-vid", format_tags=["narrative"], duration_sec=60)
    fetcher = LocalFsFetcher(root=transcript_dir)
    with conn.transaction():
        result = pipeline.fetch_reconcile(conn, talk, fetcher)
    assert result.asr_skipped is False
    assert (
        conn.execute(
            "SELECT status FROM ingest_step_status "
            "WHERE video_id='bad-vid' AND step='asr'"
        ).fetchone()[0]
        == "pending"
    )
    msgs = pgmq_client.read(conn, "ingest_asr", vt=30, qty=1)
    assert len(msgs) == 1
    assert msgs[0].message == {"video_id": "bad-vid", "step": "asr"}


# -- frames skip vs enqueue ----------------------------------------------


def test_skip_frames_when_format_tags_text_only(conn, transcript_dir):
    """format_tags lacking any of slides_heavy / code_heavy / whiteboard /
    live_demo / diagram => no visual sampling needed."""
    talk = _make_talk(
        "good-vid",
        format_tags=["narrative", "conversational", "panel"],
        duration_sec=30,
    )
    fetcher = LocalFsFetcher(root=transcript_dir)
    with conn.transaction():
        result = pipeline.fetch_reconcile(conn, talk, fetcher)
    assert result.frames_skipped is True
    assert (
        conn.execute(
            "SELECT status FROM ingest_step_status "
            "WHERE video_id='good-vid' AND step='frame_sample'"
        ).fetchone()[0]
        == "skipped"
    )
    assert pgmq_client.read(conn, "ingest_frames", vt=1, qty=1) == []


def test_enqueue_frames_when_format_tags_have_visual_content(conn, transcript_dir):
    talk = _make_talk(
        "good-vid", format_tags=["slides_heavy", "narrative"], duration_sec=30
    )
    fetcher = LocalFsFetcher(root=transcript_dir)
    with conn.transaction():
        result = pipeline.fetch_reconcile(conn, talk, fetcher)
    assert result.frames_skipped is False
    assert (
        conn.execute(
            "SELECT status FROM ingest_step_status "
            "WHERE video_id='good-vid' AND step='frame_sample'"
        ).fetchone()[0]
        == "pending"
    )
    msgs = pgmq_client.read(conn, "ingest_frames", vt=30, qty=1)
    assert len(msgs) == 1
    assert msgs[0].message == {"video_id": "good-vid", "step": "frame_sample"}


# -- transactional atomicity ----------------------------------------------


def test_fetcher_failure_rolls_back_all_writes(conn, transcript_dir):
    """FileNotFoundError from the fetcher must roll the entire reconcile —
    no talks row, no status rows, no queued messages."""
    talk = _make_talk("missing-vid", format_tags=["narrative"], duration_sec=30)
    fetcher = LocalFsFetcher(root=transcript_dir)

    with pytest.raises(FileNotFoundError):
        with conn.transaction():
            pipeline.fetch_reconcile(conn, talk, fetcher)

    assert talks_repo.get(conn, "missing-vid") is None
    rows = conn.execute(
        "SELECT step FROM ingest_step_status WHERE video_id='missing-vid'"
    ).fetchall()
    assert rows == []
    assert pgmq_client.read(conn, "ingest_chunk", vt=1, qty=1) == []
    assert pgmq_client.read(conn, "ingest_asr", vt=1, qty=1) == []
    assert pgmq_client.read(conn, "ingest_frames", vt=1, qty=1) == []


def test_re_running_reconcile_overwrites_talks_row(conn, transcript_dir):
    """Re-ingest: a second fetch_reconcile updates the talks row in place
    (ON CONFLICT). Queue sends are NOT idempotent — they accumulate, and
    the consuming worker's claim_for_update no-ops on a redelivery
    (verified in test_pgmq_smoke.py)."""
    talk = _make_talk("good-vid", format_tags=["narrative"], duration_sec=30)
    fetcher = LocalFsFetcher(root=transcript_dir)

    with conn.transaction():
        pipeline.fetch_reconcile(conn, talk, fetcher)

    revised = _make_talk(
        "good-vid", format_tags=["narrative"], duration_sec=30, title="Revised"
    )
    with conn.transaction():
        pipeline.fetch_reconcile(conn, revised, fetcher)

    fetched = talks_repo.get(conn, "good-vid")
    assert fetched is not None
    assert fetched.title == "Revised"


# -- quality probe regression ---------------------------------------------


def test_quality_fails_on_timestamp_only_vtt(tmp_path):
    """A VTT with only a header + timestamp line and zero cue text must
    fail quality. Counting timestamp digits as 'words' is the old bug."""
    vtt = "WEBVTT\n\n00:00:00.000 --> 00:01:00.000\n"
    path = tmp_path / "timestamp-only.vtt"
    path.write_text(vtt, encoding="utf-8")
    result = probe(path, duration_sec=60.0)
    assert result.passes is False
    assert result.details["cue_line_count"] == 0
    assert result.details["word_count"] == 0


def test_quality_fails_on_sparse_cues(tmp_path):
    """60s VTT with two single-word cues = real WPM 2. The fix must
    ignore timestamp lines so wpm reflects spoken content only."""
    vtt = (
        "WEBVTT\n"
        "Kind: captions\n"
        "Language: en\n"
        "\n"
        "00:00:00.000 --> 00:00:30.000 align:start position:0%\n"
        "hello\n"
        "\n"
        "00:00:30.000 --> 00:01:00.000 align:start position:0%\n"
        "world\n"
    )
    path = tmp_path / "sparse.vtt"
    path.write_text(vtt, encoding="utf-8")
    result = probe(path, duration_sec=60.0)
    assert result.passes is False
    assert result.details["cue_line_count"] == 2
    assert result.details["word_count"] == 2
    assert result.details["wpm"] == pytest.approx(2.0)


def test_quality_fails_on_placeholder_heavy_cues(tmp_path):
    """Two cue lines, both non-speech markers => non_speech_ratio = 1.0
    over cue lines (header/timestamp lines must NOT dilute it)."""
    vtt = (
        "WEBVTT\n"
        "\n"
        "00:00:00.000 --> 00:00:30.000\n"
        "[Music]\n"
        "\n"
        "00:00:30.000 --> 00:01:00.000\n"
        "[Applause]\n"
    )
    path = tmp_path / "placeholders.vtt"
    path.write_text(vtt, encoding="utf-8")
    result = probe(path, duration_sec=60.0)
    assert result.passes is False
    assert result.details["cue_line_count"] == 2
    assert result.details["non_speech_lines"] == 2
    assert result.details["non_speech_ratio"] == pytest.approx(1.0)


# -- talks.yaml path resolution -------------------------------------------


def test_local_fetcher_resolves_explicit_paths_dict(tmp_path):
    """paths={video_id: Path} overrides the <root>/<video_id>.vtt convention.
    Real corpora use this mode — talks.yaml.transcript_path values are NOT
    of the form <root>/<video_id>.vtt (they carry a `.en.vtt` suffix and
    live under a corpus-specific subdir)."""
    vtt_path = tmp_path / "v-1.en.vtt"
    vtt_path.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:05.000\nhello world\n",
        encoding="utf-8",
    )
    fetcher = LocalFsFetcher(
        paths={"v-1": vtt_path}, captions_source="youtube_auto"
    )
    result = fetcher.fetch("v-1")
    assert result.transcript_path == vtt_path
    assert result.captions_source == "youtube_auto"
    assert result.transcript_sha256 == _sha256_text(vtt_path.read_text())


def test_local_fetcher_paths_dict_beats_root_when_both_set(tmp_path):
    """paths takes precedence over root for matching video_ids; root remains
    the fallback for ids not in paths. Keeps fixture-style tests working
    while real ingest can pass an explicit map."""
    specific_dir = tmp_path / "specific"
    fallback_dir = tmp_path / "fallback"
    specific_dir.mkdir()
    fallback_dir.mkdir()

    specific_path = specific_dir / "weirdly-named.vtt"
    specific_path.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:05.000\nspecific cue\n",
        encoding="utf-8",
    )
    fallback_path = fallback_dir / "v-2.vtt"
    fallback_path.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:05.000\nfallback cue\n",
        encoding="utf-8",
    )

    fetcher = LocalFsFetcher(
        paths={"v-1": specific_path},
        root=fallback_dir,
    )

    hit = fetcher.fetch("v-1")
    assert hit.transcript_path == specific_path

    miss = fetcher.fetch("v-2")
    assert miss.transcript_path == fallback_path


def test_paths_from_talks_yaml_returns_absolute_paths_for_ai_engineering_v0():
    """Integration: verify the helper reads the real corpus and returns
    absolute paths that match the staged transcripts on disk."""
    corpus_dir = Path("eval/corpora/ai_engineering_v0")
    paths = paths_from_talks_yaml(corpus_dir)

    assert "W_CYk2ogcDI" in paths
    target = paths["W_CYk2ogcDI"]
    assert target.is_absolute()
    assert str(target).endswith(
        "transcripts/ai_engineering_v0/W_CYk2ogcDI.en.vtt"
    )
    assert target.exists()


# -- positional root regression --------------


def test_local_fetcher_positional_root_resolves_video_id_vtt(tmp_path):
    """LocalFsFetcher(tmp_path).fetch(video_id) must resolve to
    <tmp_path>/<video_id>.vtt. Regression for the dataclass field order
    bug where the positional Path was binding to `paths` (dict-typed)."""
    (tmp_path / "abc-123.vtt").write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:05.000\nhello world\n",
        encoding="utf-8",
    )
    fetcher = LocalFsFetcher(tmp_path)  # POSITIONAL — must bind to root
    result = fetcher.fetch("abc-123")
    assert result.transcript_path == tmp_path / "abc-123.vtt"
