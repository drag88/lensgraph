"""Unit tests for scripts/fetch_video.py — pure command building, no
network. The actual yt-dlp call is exercised only via the smoke-style
``--video-id`` dry-run path (skipped if the binary is absent)."""

from __future__ import annotations

import shutil

import pytest

from scripts.fetch_video import (
    SAFETY_PAD_SEC,
    YT_DLP_FORMAT,
    build_commands,
)


def _talks() -> list[dict]:
    return [
        {
            "video_id": "fullvid",
            "url": "https://www.youtube.com/watch?v=fullvid",
            "duration_sec": 600,
        },
        {
            "video_id": "ch-a",
            "source_video_id": "parent1",
            "source_start_sec": 100,
            "source_end_sec": 300,
            "url": "https://www.youtube.com/watch?v=parent1&t=100s",
            "duration_sec": 200,
        },
        {
            "video_id": "ch-b",
            "source_video_id": "parent1",
            "source_start_sec": 500,
            "source_end_sec": 900,
            "url": "https://www.youtube.com/watch?v=parent1&t=500s",
            "duration_sec": 400,
        },
        {
            "video_id": "lone-chapter",
            "source_video_id": "parent2",
            "source_start_sec": 50,
            "source_end_sec": 1200,
            "url": "https://www.youtube.com/watch?v=parent2",
            "duration_sec": 1150,
        },
    ]


def test_full_video_yields_plain_yt_dlp(tmp_path, monkeypatch):
    from scripts import fetch_video

    monkeypatch.setattr(fetch_video, "REPO_ROOT", tmp_path)
    cmds = build_commands("c", _talks(), only_video_id="fullvid")
    assert len(cmds) == 1
    target, argv, reason = cmds[0]
    assert target.name == "fullvid.mp4"
    assert "--download-sections" not in argv
    assert YT_DLP_FORMAT in argv
    assert "no chapter slice" in reason


def test_chapter_group_uses_union_end_plus_safety_pad(tmp_path, monkeypatch):
    """Two chapters share parent1 with source_end_sec 300 and 900.
    The single command must download `*0-<max(end)+SAFETY_PAD_SEC>`."""
    from scripts import fetch_video

    monkeypatch.setattr(fetch_video, "REPO_ROOT", tmp_path)
    cmds = build_commands("c", _talks(), only_video_id="ch-a")
    assert len(cmds) == 1, "ch-a and ch-b share parent1 → ONE command"
    target, argv, reason = cmds[0]
    assert target.name == "parent1.mp4"
    section_idx = argv.index("--download-sections")
    section_spec = argv[section_idx + 1]
    assert section_spec == f"*0-{900 + SAFETY_PAD_SEC}"
    assert "ch-a" in reason and "ch-b" in reason


def test_lone_chapter_still_pads(tmp_path, monkeypatch):
    from scripts import fetch_video

    monkeypatch.setattr(fetch_video, "REPO_ROOT", tmp_path)
    cmds = build_commands("c", _talks(), only_video_id="lone-chapter")
    assert len(cmds) == 1
    _, argv, _ = cmds[0]
    assert f"*0-{1200 + SAFETY_PAD_SEC}" in argv


def test_url_query_stripped(tmp_path, monkeypatch):
    """The `&t=` fragment must be dropped — yt-dlp ignores it and a
    bare URL makes the printed command identical for sibling chapters."""
    from scripts import fetch_video

    monkeypatch.setattr(fetch_video, "REPO_ROOT", tmp_path)
    cmds = build_commands("c", _talks(), only_video_id="ch-a")
    _, argv, _ = cmds[0]
    url = argv[-1]
    assert url == "https://www.youtube.com/watch?v=parent1".replace("?v=", "?v=").rstrip()
    # Cheaper assertion: nothing after the `v` query disambiguator survived.
    assert "t=" not in url
    assert "&" not in url


def test_corpus_all_returns_one_per_source(tmp_path, monkeypatch):
    """Three physical files cover four talks: fullvid, parent1
    (ch-a + ch-b merged), parent2 (lone-chapter alone)."""
    from scripts import fetch_video

    monkeypatch.setattr(fetch_video, "REPO_ROOT", tmp_path)
    cmds = build_commands("c", _talks())
    names = sorted(c[0].name for c in cmds)
    assert names == ["fullvid.mp4", "parent1.mp4", "parent2.mp4"]


def test_skip_when_file_exists_without_force(tmp_path, monkeypatch):
    from scripts import fetch_video

    monkeypatch.setattr(fetch_video, "REPO_ROOT", tmp_path)
    # Pre-create the target so build_commands sees an existing file.
    target = tmp_path / "videos" / "c" / "fullvid.mp4"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing")
    cmds = build_commands("c", _talks(), only_video_id="fullvid")
    assert cmds[0][1] == [], "no argv when target exists and force is False"
    assert "skip" in cmds[0][2]


def test_force_overrides_existing(tmp_path, monkeypatch):
    from scripts import fetch_video

    monkeypatch.setattr(fetch_video, "REPO_ROOT", tmp_path)
    target = tmp_path / "videos" / "c" / "fullvid.mp4"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing")
    cmds = build_commands("c", _talks(), only_video_id="fullvid", force=True)
    assert cmds[0][1], "force=True must produce a yt-dlp argv even if file exists"


def test_unknown_video_id_raises(tmp_path, monkeypatch):
    from scripts import fetch_video

    monkeypatch.setattr(fetch_video, "REPO_ROOT", tmp_path)
    with pytest.raises(KeyError, match="not found"):
        build_commands("c", _talks(), only_video_id="nope")


def test_yt_dlp_is_available_on_path():
    """ADR 005 names yt-dlp as the canonical fetcher. If it's missing,
    every downstream visual eval workflow breaks silently — this test
    fails loudly so the operator pins the dependency."""
    assert shutil.which("yt-dlp") is not None, (
        "yt-dlp not on PATH; install per ADR 005 (e.g. `uv tool install yt-dlp` "
        "or `pipx install yt-dlp`)"
    )
