"""Tests for eval/curation/clip_vtt.py.

The clipper rebases the OUTER cue timestamps and strips YouTube's inline
word-level karaoke tags. Stripping is the fix for the bug where chapter-sliced
transcripts kept inline tags at original livestream offsets (e.g. cue header
00:00:00.479 but payload <00:08:36.560><c>...</c>), leaving the file
internally inconsistent.
"""

from __future__ import annotations

import textwrap

from eval.curation.clip_vtt import clip_vtt, strip_inline_tags


def _src(*lines: str) -> str:
    return textwrap.dedent("\n".join(lines)).lstrip("\n")


def test_header_preserved():
    src = _src(
        "WEBVTT",
        "Kind: captions",
        "",
        "00:08:36.000 --> 00:08:39.000",
        "hello world",
        "",
    )
    out = clip_vtt(src, 8 * 60 + 36, 8 * 60 + 40)
    assert out.startswith("WEBVTT\nKind: captions")


def test_cue_timestamps_rebased_to_zero():
    src = _src(
        "WEBVTT",
        "",
        "00:08:36.000 --> 00:08:39.000",
        "hello world",
        "",
    )
    out = clip_vtt(src, 516, 540)  # 8:36..9:00
    assert "00:00:00.000 --> 00:00:03.000" in out
    assert "00:08:36" not in out  # original offsets must be gone


def test_inline_word_timestamps_stripped():
    src = _src(
        "WEBVTT",
        "",
        "00:08:36.000 --> 00:08:39.000",
        "Good<00:08:36.560><c> morning</c><00:08:36.880><c> everyone.</c>",
        "",
    )
    out = clip_vtt(src, 516, 540)
    # No livestream offsets and no <c> tags anywhere in the output.
    assert "<00:08:" not in out
    assert "<c>" not in out and "</c>" not in out
    assert "Good morning everyone." in out


def test_inline_cue_class_with_color_stripped():
    src = _src(
        "WEBVTT",
        "",
        "00:00:10.000 --> 00:00:12.000",
        "Hi<00:00:10.500><c.colorE5E5E5> there</c>",
        "",
    )
    out = clip_vtt(src, 10, 13)
    assert "<c" not in out and "</c>" not in out


def test_cues_outside_range_dropped():
    src = _src(
        "WEBVTT",
        "",
        "00:00:00.000 --> 00:00:05.000",
        "before",
        "",
        "00:00:30.000 --> 00:00:33.000",
        "after",
        "",
        "00:00:10.000 --> 00:00:14.000",
        "inside",
        "",
    )
    out = clip_vtt(src, 10, 20)
    assert "inside" in out
    assert "before" not in out
    assert "after" not in out


def test_cue_partially_overlapping_range_is_clipped():
    src = _src(
        "WEBVTT",
        "",
        "00:00:08.000 --> 00:00:12.000",
        "spanning",
        "",
    )
    out = clip_vtt(src, 10, 20)
    # Cue extends from 8s to 12s; clipped to start of range (10s).
    assert "00:00:00.000 --> 00:00:02.000" in out


def test_strip_inline_tags_helper_is_idempotent():
    payload = "Good<00:08:36.560><c> morning</c>"
    once = strip_inline_tags(payload)
    twice = strip_inline_tags(once)
    assert once == twice == "Good morning"
