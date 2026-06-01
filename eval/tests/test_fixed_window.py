"""Tests for chunking.fixed_window — pure functions, no DB / no network."""

from __future__ import annotations

import textwrap

import pytest

from chunking import get_chunker
from chunking.fixed_window import STRATEGY_NAME, _dedup_rolling, _parse_vtt, chunk
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
    fn = get_chunker(STRATEGY_NAME)
    assert callable(fn)
    result = fn(_vtt_60s(), [], video_id="vid1")
    assert all(isinstance(c, Chunk) for c in result)
    assert all(c.chunking_strategy == STRATEGY_NAME for c in result)
    with pytest.raises(KeyError):
        get_chunker("does_not_exist")


# -- sentence-snap bound -------------------------------------------------


def test_sentence_snap_is_bounded_for_far_boundary():
    """RED for the bug where _snap_to_sentence trimmed back to ANY sentence
    boundary regardless of distance. With a short opener cue + a long
    monologue cue, the unbounded snap would drop the entire monologue (27+
    words) while leaving end_sec at the monologue's end_sec — text and span
    misaligned. After the fix, the bounded snap leaves the text alone when
    the trim distance exceeds the bound; alignment preserved.
    """
    vtt = textwrap.dedent(
        """\
        WEBVTT

        00:00:00.000 --> 00:00:05.000
        Brief opener.

        00:00:05.000 --> 00:00:25.000
        Then a very long extended monologue continues without any further sentence terminators just rambling on and on covering many words but never reaching a period mark
        """
    )
    chunks = chunk(vtt, [], video_id="vid1", window_sec=30.0, overlap_sec=5.0)
    assert len(chunks) == 1
    c = chunks[0]
    # The monologue text MUST be present. Without the bound, snap would
    # trim back to the period after "Brief opener.", dropping ~27 words.
    assert "monologue" in c.text
    assert "rambling" in c.text
    # Timestamp/text alignment: end_sec matches the last cue's end_sec
    # (25.0), not the position of the early sentence boundary.
    assert c.start_sec == pytest.approx(0.0)
    assert c.end_sec == pytest.approx(25.0)


# -- rolling-caption dedup -----------------------------------------------
# YouTube auto-captions repeat each line as residue in the next cue(s),
# triplicating text and bleeding earlier words into later spans. The
# parser must dedup the residue WITHOUT moving any cue's (start, end).


def _vtt_rolling_tagged() -> str:
    """Roll-up with inline <c> word tags (W_CYk2ogcDI shape). Cue 2 is a
    10ms flicker repeating cue 1's settled text; cue 3 carries cue 1's
    tail as residue then the new line."""
    return textwrap.dedent(
        """\
        WEBVTT
        Kind: captions
        Language: en

        00:00:15.200 --> 00:00:16.710 align:start position:0%

        Thanks<00:00:15.519><c> for</c><00:00:15.679><c> coming.</c><00:00:16.000><c> Thanks</c><00:00:16.480><c> me</c>

        00:00:16.710 --> 00:00:16.720 align:start position:0%
        Thanks for coming. Thanks me


        00:00:16.720 --> 00:00:20.950 align:start position:0%
        Thanks for coming. Thanks me
        here.<00:00:17.039><c> Um</c><00:00:17.520><c> I'm</c><00:00:18.240><c> Tongima.</c>
        """
    )


def _vtt_rolling_tagless() -> str:
    """Roll-up with NO word tags and &gt;&gt; speaker markers (arize shape).
    Residue must be caught by a verbatim prev-tail compare, not by tag
    presence."""
    return textwrap.dedent(
        """\
        WEBVTT
        Kind: captions
        Language: en

        00:00:00.479 --> 00:00:01.750 align:start position:0%

        &gt;&gt; Good morning everyone. Thanks so much

        00:00:01.750 --> 00:00:01.760 align:start position:0%
        &gt;&gt; Good morning everyone. Thanks so much

        00:00:01.760 --> 00:00:03.750 align:start position:0%
        &gt;&gt; Good morning everyone. Thanks so much
        for spending your morning with me. It's
        """
    )


def test_rolling_dedup_removes_residue_tagged():
    cues = _dedup_rolling(_parse_vtt(_vtt_rolling_tagged()))
    texts = [t for _, _, t in cues]
    assert texts == [
        "Thanks for coming. Thanks me",
        "",  # flicker collapses to empty, cue KEPT
        "here. Um I'm Tongima.",  # residue line dropped, new line kept
    ]
    # The settled phrase appears exactly once across the whole stream.
    assert " ".join(texts).count("Thanks for coming. Thanks me") == 1
    # No head-bleed: the 16.720 span must NOT carry the earlier phrase.
    assert "Thanks for coming" not in cues[2][2]


def test_rolling_dedup_removes_residue_tagless():
    cues = _dedup_rolling(_parse_vtt(_vtt_rolling_tagless()))
    texts = [t for _, _, t in cues]
    assert texts == [
        "Good morning everyone. Thanks so much",
        "",
        "for spending your morning with me. It's",
    ]
    # >> speaker marker and &gt; entity are gone everywhere.
    joined = " ".join(texts)
    assert "&gt;" not in joined
    assert ">>" not in joined


def test_rolling_dedup_preserves_cue_timestamps_and_count():
    parsed = _parse_vtt(_vtt_rolling_tagged())
    deduped = _dedup_rolling(parsed)
    # Boundary-preserving: same number of cues, identical (start, end).
    assert len(deduped) == len(parsed)
    assert [(s, e) for s, e, _ in deduped] == [(s, e) for s, e, _ in parsed]


def test_rolling_dedup_html_unescape_and_speaker_strip():
    chunks = chunk(_vtt_rolling_tagless(), [], video_id="vid1")
    for c in chunks:
        assert "&gt;" not in c.text
        assert not c.text.lstrip().startswith(">>")


def test_chunk_text_aligns_to_span_no_head_bleed():
    # Full chunk() over the tagged fixture: the phrase "Thanks for coming"
    # must appear at most once (at its real 15.2s origin), never triplicated
    # nor bled into a later span.
    chunks = chunk(_vtt_rolling_tagged(), [], video_id="vid1")
    for c in chunks:
        assert c.text.count("Thanks for coming") <= 1
    # The clean fixtures with distinct cue text are unaffected (dedup no-op).
    clean = chunk(_vtt_60s(), [], video_id="vid1")
    assert len(clean) == 3
