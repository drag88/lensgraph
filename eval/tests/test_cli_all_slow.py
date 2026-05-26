"""cli_all exit-code tests.

Slow: requires live Postgres reachable at $POSTGRES_DSN. cli_all is a
one-shot driver; the contract these tests guard is that fetch failures
must not be silently swallowed — main() must return 1 so CI / make see
the failure.
"""

from __future__ import annotations

import psycopg
import pytest

from db.conn import dsn as resolve_dsn
from db.migrate import apply
from queues import pgmq_client

pytestmark = pytest.mark.slow

TEST_DB_NAME = "lensgraph_test_cli_all"


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
def clean_db(test_db, monkeypatch):
    with psycopg.connect(test_db, autocommit=True) as c:
        c.execute(
            "TRUNCATE chunk_token_embeds, sparse_embeds, dense_embeds, "
            "chunks, ingest_step_status, talks CASCADE"
        )
        for q in pgmq_client.QUEUES:
            pgmq_client.purge(c, q)
    monkeypatch.setenv("POSTGRES_DSN", test_db)
    yield test_db


def test_cli_all_returns_nonzero_when_a_talk_transcript_missing(clean_db, tmp_path, monkeypatch):
    """cli_all keeps going on per-talk fetch failures (the existing
    report-and-continue try/except) but main() must return 1 so callers
    see the failure. Builds a tmp corpus pointing at a non-existent
    transcript so the LocalFsFetcher raises during fetch_reconcile."""
    corpora_root = tmp_path / "eval" / "corpora" / "fake_corpus"
    corpora_root.mkdir(parents=True)
    sha = "a" + "0" * 63
    talks_yaml = corpora_root / "talks.yaml"
    talks_yaml.write_text(
        "- video_id: ghost-vid\n"
        '  title: "Will Never Fetch"\n'
        '  speaker: "N/A"\n'
        '  url: "https://example.com/x"\n'
        "  duration_sec: 60\n"
        "  format_tags: [narrative]\n"
        "  license: cc-by\n"
        "  captions_source: manual_transcript\n"
        '  transcript_path: "transcripts/does_not_exist.vtt"\n'
        f'  transcript_sha256: "{sha}"\n'
        '  accessed_at: "2025-01-01T00:00:00Z"\n',
        encoding="utf-8",
    )

    from ingest import cli_all

    monkeypatch.setattr(cli_all, "REPO_ROOT", tmp_path)
    exit_code = cli_all.main(["--corpus", "fake_corpus"])
    assert exit_code == 1


def test_cli_all_returns_nonzero_when_readiness_incomplete(clean_db, tmp_path, monkeypatch):
    """Fetch succeeds for a real on-disk transcript, but no chunks land
    (we don't run the chunk handler), so readiness is incomplete and
    main() must still return 1."""
    # Real VTT on disk so fetch_reconcile succeeds.
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    vtt = transcripts / "ok.vtt"
    vtt.write_text(
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:10.000\n"
        "first cue body text\n\n"
        "00:00:10.000 --> 00:00:20.000\n"
        "second cue body text\n",
        encoding="utf-8",
    )
    import hashlib

    sha = hashlib.sha256(vtt.read_bytes()).hexdigest()

    corpora_root = tmp_path / "eval" / "corpora" / "real_corpus"
    corpora_root.mkdir(parents=True)
    talks_yaml = corpora_root / "talks.yaml"
    talks_yaml.write_text(
        "- video_id: real-vid\n"
        '  title: "Real Vid"\n'
        '  speaker: "S"\n'
        '  url: "https://example.com/y"\n'
        "  duration_sec: 20\n"
        "  format_tags: [narrative]\n"
        "  license: cc-by\n"
        "  captions_source: manual_transcript\n"
        f'  transcript_path: "transcripts/ok.vtt"\n'
        f'  transcript_sha256: "{sha}"\n'
        '  accessed_at: "2025-01-01T00:00:00Z"\n',
        encoding="utf-8",
    )
    # dev_gold so readiness.collect_video_ids picks up the video_id.
    (corpora_root / "dev_gold.jsonl").write_text(
        '{"id": "x", "video_id": "real-vid"}\n', encoding="utf-8"
    )

    from ingest import cli_all

    monkeypatch.setattr(cli_all, "REPO_ROOT", tmp_path)
    # Stub the drain so chunks never land — readiness will be incomplete.
    monkeypatch.setattr(cli_all, "_drain", lambda c, q, h: 0)
    exit_code = cli_all.main(["--corpus", "real_corpus"])
    assert exit_code == 1


def test_cli_all_main_returns_int_type(clean_db, tmp_path, monkeypatch):
    """Sanity: main returns int, not bool/None, so sys.exit gets a real code."""
    corpora_root = tmp_path / "eval" / "corpora" / "empty_corpus"
    corpora_root.mkdir(parents=True)
    (corpora_root / "talks.yaml").write_text("[]\n", encoding="utf-8")

    from ingest import cli_all

    monkeypatch.setattr(cli_all, "REPO_ROOT", tmp_path)
    rc = cli_all.main(["--corpus", "empty_corpus"])
    assert isinstance(rc, int)
    assert rc == 0


# -- frames-drain gate ----------------------------------------------------


def _stage_vtt(transcripts_dir, name: str) -> tuple[str, str]:
    """Write a tiny VTT under `transcripts_dir/name` and return
    (relative_repo_path, sha256). Mirrors the helper used in
    test_cli_all_returns_nonzero_when_readiness_incomplete."""
    import hashlib

    transcripts_dir.mkdir(parents=True, exist_ok=True)
    vtt = transcripts_dir / name
    vtt.write_text(
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:10.000\n"
        "first cue body text\n\n"
        "00:00:10.000 --> 00:00:20.000\n"
        "second cue body text\n",
        encoding="utf-8",
    )
    return f"transcripts/{name}", hashlib.sha256(vtt.read_bytes()).hexdigest()


def test_cli_all_skips_frames_drain_when_no_visual_talks_in_corpus(
    clean_db, tmp_path, monkeypatch, capsys
):
    """A corpus with only `narrative` format_tags has zero visual talks →
    zero required .mp4s → frames drain is skipped cleanly with exit 0.
    The text-only ingest path must not be blocked by the visual gate."""
    rel, sha = _stage_vtt(tmp_path / "transcripts", "novisual.vtt")
    corpora_root = tmp_path / "eval" / "corpora" / "novisual_corpus"
    corpora_root.mkdir(parents=True)
    (corpora_root / "talks.yaml").write_text(
        "- video_id: nv-vid\n"
        '  title: "Narrative Only"\n'
        '  speaker: "S"\n'
        '  url: "https://example.com/n"\n'
        "  duration_sec: 20\n"
        "  format_tags: [narrative]\n"
        "  license: cc-by\n"
        "  captions_source: manual_transcript\n"
        f'  transcript_path: "{rel}"\n'
        f'  transcript_sha256: "{sha}"\n'
        '  accessed_at: "2025-01-01T00:00:00Z"\n',
        encoding="utf-8",
    )

    from ingest import cli_all

    monkeypatch.setattr(cli_all, "REPO_ROOT", tmp_path)
    # Stub drain so we don't need to load BGE-M3 for this gate test.
    monkeypatch.setattr(cli_all, "_drain", lambda c, q, h: 0)

    exit_code = cli_all.main(["--corpus", "novisual_corpus"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "skipping ingest_frames drain" in out
    assert "no visual talks" in out


def test_cli_all_skips_frames_drain_when_zero_required_videos_staged(
    clean_db, tmp_path, monkeypatch, capsys
):
    """A corpus with a slides_heavy talk but NO .mp4 staged anywhere
    (videos/ dir doesn't even exist) → skip cleanly with exit 0. Operators
    stage videos out of band; the text-only path must not be blocked."""
    rel, sha = _stage_vtt(tmp_path / "transcripts", "slides.vtt")
    corpora_root = tmp_path / "eval" / "corpora" / "slides_corpus"
    corpora_root.mkdir(parents=True)
    (corpora_root / "talks.yaml").write_text(
        "- video_id: sh-vid\n"
        '  title: "Slides Heavy No Video"\n'
        '  speaker: "S"\n'
        '  url: "https://example.com/s"\n'
        "  duration_sec: 20\n"
        "  format_tags: [slides_heavy]\n"
        "  license: cc-by\n"
        "  captions_source: manual_transcript\n"
        f'  transcript_path: "{rel}"\n'
        f'  transcript_sha256: "{sha}"\n'
        '  accessed_at: "2025-01-01T00:00:00Z"\n',
        encoding="utf-8",
    )

    from ingest import cli_all

    monkeypatch.setattr(cli_all, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cli_all, "_drain", lambda c, q, h: 0)

    assert not (tmp_path / "videos").exists()  # invariant: no videos/ dir
    exit_code = cli_all.main(["--corpus", "slides_corpus"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "skipping ingest_frames drain" in out
    assert "no local videos staged" in out


def test_cli_all_returns_nonzero_when_some_required_videos_missing(
    clean_db, tmp_path, monkeypatch, capsys
):
    """Two slides_heavy talks, one .mp4 staged. The drain MUST NOT run
    (partial drain still infinite-loops on the missing one), AND main()
    must exit nonzero so CI surfaces the gap."""
    rel_a, sha_a = _stage_vtt(tmp_path / "transcripts", "a.vtt")
    rel_b, sha_b = _stage_vtt(tmp_path / "transcripts", "b.vtt")
    corpora_root = tmp_path / "eval" / "corpora" / "partial_corpus"
    corpora_root.mkdir(parents=True)
    (corpora_root / "talks.yaml").write_text(
        "- video_id: vid-a\n"
        '  title: "A"\n'
        '  speaker: "S"\n'
        '  url: "https://example.com/a"\n'
        "  duration_sec: 20\n"
        "  format_tags: [slides_heavy]\n"
        "  license: cc-by\n"
        "  captions_source: manual_transcript\n"
        f'  transcript_path: "{rel_a}"\n'
        f'  transcript_sha256: "{sha_a}"\n'
        '  accessed_at: "2025-01-01T00:00:00Z"\n'
        "- video_id: vid-b\n"
        '  title: "B"\n'
        '  speaker: "S"\n'
        '  url: "https://example.com/b"\n'
        "  duration_sec: 20\n"
        "  format_tags: [slides_heavy]\n"
        "  license: cc-by\n"
        "  captions_source: manual_transcript\n"
        f'  transcript_path: "{rel_b}"\n'
        f'  transcript_sha256: "{sha_b}"\n'
        '  accessed_at: "2025-01-01T00:00:00Z"\n',
        encoding="utf-8",
    )

    # Stage ONLY vid-a.mp4; vid-b.mp4 is intentionally missing.
    videos_dir = tmp_path / "videos" / "partial_corpus"
    videos_dir.mkdir(parents=True)
    (videos_dir / "vid-a.mp4").touch()

    from ingest import cli_all

    monkeypatch.setattr(cli_all, "REPO_ROOT", tmp_path)
    # Stub drain to assert it is NEVER called for ingest_frames in this case
    # (partial drain would retry-loop forever on the missing video).
    drained_queues: list[str] = []

    def fake_drain(conn, queue, handler):
        drained_queues.append(queue)
        return 0

    monkeypatch.setattr(cli_all, "_drain", fake_drain)

    exit_code = cli_all.main(["--corpus", "partial_corpus"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "ingest_frames" not in drained_queues  # drain skipped, as designed
    assert "1 of 2 required videos missing" in out
    assert "vid-b.mp4" in out


def test_cli_all_drains_ingest_frames_when_all_required_videos_staged(
    clean_db, tmp_path, monkeypatch, capsys
):
    """All required .mp4 files staged → _drain runs once for ingest_frames
    AND main exits 0 (text readiness stubbed complete via readiness.for_corpus).

    This is the positive-path regression for the required-videos guard: the
    other tests confirm the gate REJECTS partial/empty staging; this one
    confirms the gate ADMITS full staging without forcing exit 1."""
    from ingest.readiness import VideoReadiness

    rel, sha = _stage_vtt(tmp_path / "transcripts", "full.vtt")
    corpora_root = tmp_path / "eval" / "corpora" / "full_corpus"
    corpora_root.mkdir(parents=True)
    (corpora_root / "talks.yaml").write_text(
        "- video_id: vid-c\n"
        '  title: "Complete Slides Talk"\n'
        '  speaker: "S"\n'
        '  url: "https://example.com/c"\n'
        "  duration_sec: 20\n"
        "  format_tags: [slides_heavy]\n"
        "  license: cc-by\n"
        "  captions_source: manual_transcript\n"
        f'  transcript_path: "{rel}"\n'
        f'  transcript_sha256: "{sha}"\n'
        '  accessed_at: "2025-01-01T00:00:00Z"\n',
        encoding="utf-8",
    )

    # Stage the single required .mp4 — convention is
    # videos/<corpus>/<source_video_id or video_id>.mp4.
    videos_dir = tmp_path / "videos" / "full_corpus"
    videos_dir.mkdir(parents=True)
    (videos_dir / "vid-c.mp4").touch()

    from ingest import cli_all

    monkeypatch.setattr(cli_all, "REPO_ROOT", tmp_path)

    # Track each _drain invocation by queue so we can assert ingest_frames
    # drained exactly once.
    drained_queues: list[str] = []

    def fake_drain(conn, queue, handler):
        drained_queues.append(queue)
        return 0

    monkeypatch.setattr(cli_all, "_drain", fake_drain)

    # Stub readiness so text-side reports complete and we test ONLY the
    # frames-drain gate / exit-code interaction.
    stub_report = VideoReadiness(
        video_id="vid-c",
        talks_present=True,
        chunks_n=1,
        dense_n=1,
        sparse_n=1,
        tokens_n=1,
        frame_sample_status="completed",
        frames_n=2,
    )
    monkeypatch.setattr(
        cli_all.readiness, "for_corpus", lambda conn, corpus_dir: [stub_report]
    )

    exit_code = cli_all.main(["--corpus", "full_corpus"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert drained_queues.count("ingest_frames") == 1
    # Drain order is chunk → embed_text → frames per cli_all.main; assert
    # the full sequence so a future reorder surfaces here.
    assert drained_queues == ["ingest_chunk", "ingest_embed_text", "ingest_frames"]
    assert "drained 0 from ingest_frames" in out
    # The skip-with-warning messages must NOT appear when all required
    # videos are staged.
    assert "skipping ingest_frames drain" not in out
