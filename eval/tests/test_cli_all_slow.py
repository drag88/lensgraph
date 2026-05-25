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
