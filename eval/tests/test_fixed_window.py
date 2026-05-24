"""Tests for chunking.fixed_window — pure functions, no DB / no network."""

from __future__ import annotations

import textwrap

import pytest

from chunking import get_chunker
from chunking.fixed_window import chunk
from chunking.types import Chunk, Frame


def _vtt_60s() -> str:
    """Six cues spanning 60 seconds.

    Cues 0,1,3 are 10s, non-overlapping at 0,10,30. Cue 2 (22-30) is short
    so window 1 (which spans cues 2,3,4) stays within width=30. Cue 4
    (40-52) bridges into window 2's region; cue 5 (51-60) bridges back.
    The shape is intentional: it produces three chunks whose temporal
    spans strictly overlap consecutively while staying inside window_sec.
    """
    return textwrap.dedent(
        """\
        WEBVTT
        Kind: captions
        Language: en

        00:00:00.000 --> 00:00:10.000
        alpha bravo charlie delta echo

        00:00:10.000 --> 00:00:20.000
        foxtrot golf hotel india juliet

        00:00:22.000 --> 00:00:30.000
        kilo lima mike november oscar

        00:00:30.000 --> 00:00:40.000
        papa quebec romeo sierra tango

        00:00:40.000 --> 00:00:52.000
        uniform victor whiskey xray yankee

        00:00:51.000 --> 00:01:00.000
        zulu alpha2 bravo2 charlie2 delta2
        """
    )


def test_chunk_count_for_60s_vtt_with_default_window():
    chunks = chunk(_vtt_60s(), [], video_id="vid1")
    assert len(chunks) == 3
    starts = [c.start_sec for c in chunks]
    # Stride is 25; first chunk starts at first cue (0), then ~25, then ~50.
    assert starts[0] == pytest.approx(0.0)
    assert starts[1] == pytest.approx(22.0)
    assert starts[2] == pytest.approx(51.0)


def test_chunk_widths_are_within_window_sec():
    chunks = chunk(_vtt_60s(), [], video_id="vid1")
    for c in chunks:
        assert c.end_sec - c.start_sec <= 30.0 + 1e-9


def test_chunks_overlap_at_boundaries():
    chunks = chunk(_vtt_60s(), [], video_id="vid1")
    for i in range(len(chunks) - 1):
        assert chunks[i].end_sec > chunks[i + 1].start_sec, (
            f"chunk[{i}].end_sec={chunks[i].end_sec} should overlap "
            f"chunk[{i + 1}].start_sec={chunks[i + 1].start_sec}"
        )


def test_chunk_spans_are_monotonically_increasing():
    chunks = chunk(_vtt_60s(), [], video_id="vid1")
    for i in range(len(chunks) - 1):
        assert chunks[i].start_sec < chunks[i + 1].start_sec
        assert chunks[i].end_sec <= chunks[i + 1].end_sec


def test_sentence_boundary_snap_prefers_period_endings():
    vtt = textwrap.dedent(
        """\
        WEBVTT

        00:00:00.000 --> 00:00:10.000
        First sentence ends here. Second sentence begins

        00:00:10.000 --> 00:00:20.000
        and continues without finishing
        """
    )
    chunks = chunk(vtt, [], video_id="vid1", window_sec=30.0, overlap_sec=5.0)
    assert len(chunks) == 1
    text = chunks[0].text
    # Joined text ends mid-sentence ("...finishing"); snap should trim back
    # to the period after "First sentence ends here."
    assert text.rstrip().endswith(".")
    assert text == "First sentence ends here."


def test_frame_secs_population_assigns_frames_to_containing_chunk():
    frames = [Frame(5.0), Frame(35.0), Frame(55.0)]
    chunks = chunk(_vtt_60s(), frames, video_id="vid1")
    assert len(chunks) == 3
    # chunk 0 spans [0, 30) → contains 5.0 only
    assert chunks[0].frame_secs == [5.0]
    # chunk 1 spans [22, 52) → contains 35.0 only
    assert chunks[1].frame_secs == [35.0]
    # chunk 2 spans [51, 60) → contains 55.0 only
    assert chunks[2].frame_secs == [55.0]


def test_max_tokens_truncates_oversized_chunk():
    vtt = textwrap.dedent(
        """\
        WEBVTT

        00:00:00.000 --> 00:00:10.000
        one two three four five six seven eight nine ten eleven twelve
        """
    )
    chunks = chunk(vtt, [], video_id="vid1", max_tokens=10)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.token_count <= 10
    # Truncation keeps a strict prefix of the original word stream.
    original_words = "one two three four five six seven eight nine ten eleven twelve".split()
    truncated_words = c.text.split()
    assert truncated_words == original_words[: len(truncated_words)]


def test_overlap_must_be_less_than_window():
    with pytest.raises(ValueError):
        chunk(_vtt_60s(), [], video_id="vid1", window_sec=30.0, overlap_sec=30.0)
    with pytest.raises(ValueError):
        chunk(_vtt_60s(), [], video_id="vid1", window_sec=30.0, overlap_sec=35.0)


def test_get_chunker_returns_callable():
    fn = get_chunker("fixed_window")
    assert callable(fn)
    result = fn(_vtt_60s(), [], video_id="vid1")
    assert all(isinstance(c, Chunk) for c in result)
    with pytest.raises(KeyError):
        get_chunker("does_not_exist")
