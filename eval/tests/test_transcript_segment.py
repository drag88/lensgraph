"""Tests for chunking.transcript_segment — pure functions, no DB / no network.

transcript_segment chunks on SENTENCE boundaries instead of fixed time
windows, so chunk edges land at completed thoughts (the fix for fixed_window's
mid-sentence cuts surfaced by the 2026-06-01 boundary audit). It accumulates
cues until the chunk both clears a target duration AND ends on terminal
punctuation, with a max_tokens safety cut for runaway sentences.
"""

from __future__ import annotations

import textwrap

import pytest

from chunking import get_chunker
from chunking.transcript_segment import STRATEGY_NAME, chunk
from chunking.types import Chunk, Frame


def _vtt_multi_sentence() -> str:
    """~65s of speech where sentences end cleanly at cue boundaries, so a
    30s target yields two sentence-aligned chunks: [0,32] and [32,65]."""
    return textwrap.dedent(
        """\
        WEBVTT
        Kind: captions
        Language: en

        00:00:00.000 --> 00:00:10.000
        Alpha one two three. Beta four five six.

        00:00:10.000 --> 00:00:20.000
        Gamma seven eight nine ten eleven twelve.

        00:00:20.000 --> 00:00:32.000
        Delta thirteen fourteen fifteen sixteen seventeen eighteen.

        00:00:32.000 --> 00:00:42.000
        Epsilon nineteen twenty. Zeta twenty-one twenty-two.

        00:00:42.000 --> 00:00:55.000
        Eta twenty-three twenty-four twenty-five twenty-six.

        00:00:55.000 --> 00:01:05.000
        Theta twenty-eight twenty-nine thirty.
        """
    )


def test_chunks_end_on_sentence_terminal_punctuation():
    chunks = chunk(_vtt_multi_sentence(), [], video_id="vid1", target_sec=30.0)
    assert len(chunks) == 2
    for c in chunks:
        assert c.text.rstrip()[-1] in ".!?", f"chunk does not end on a sentence: {c.text!r}"


def test_chunks_start_at_a_sentence_start():
    chunks = chunk(_vtt_multi_sentence(), [], video_id="vid1", target_sec=30.0)
    # The second chunk must begin a new sentence (capitalized first word),
    # not continue the previous one mid-thought.
    assert chunks[1].text[:1].isupper()
    assert chunks[1].text.startswith("Epsilon")


def test_boundaries_are_sentence_aligned_and_non_overlapping():
    chunks = chunk(_vtt_multi_sentence(), [], video_id="vid1", target_sec=30.0)
    assert (chunks[0].start_sec, chunks[0].end_sec) == pytest.approx((0.0, 32.0))
    assert (chunks[1].start_sec, chunks[1].end_sec) == pytest.approx((32.0, 65.0))
    # Sentence-aligned => no overlap (unlike fixed_window's 5s overlap).
    for i in range(len(chunks) - 1):
        assert chunks[i + 1].start_sec >= chunks[i].end_sec


def test_does_not_cut_before_target_even_at_a_sentence_end():
    # cue0 ends a sentence at 10s, well before the 30s target — must NOT flush.
    chunks = chunk(_vtt_multi_sentence(), [], video_id="vid1", target_sec=30.0)
    assert chunks[0].end_sec > 10.0


def test_max_tokens_force_cut_for_runaway_sentence():
    vtt = textwrap.dedent(
        """\
        WEBVTT

        00:00:00.000 --> 00:00:40.000
        one two three four five six seven eight nine ten eleven twelve thirteen
        """
    )
    chunks = chunk(vtt, [], video_id="vid1", target_sec=30.0, max_tokens=8)
    assert len(chunks) >= 1
    assert all(c.token_count <= 8 for c in chunks)
    # force-cut keeps a strict prefix of the word stream
    assert chunks[0].text.split()[:3] == ["one", "two", "three"]


def test_handles_rolling_caption_vtt_without_duplication():
    vtt = textwrap.dedent(
        """\
        WEBVTT
        Kind: captions
        Language: en

        00:00:00.000 --> 00:00:02.000

        Hello<00:00:00.500><c> there.</c><00:00:01.000><c> This</c>

        00:00:02.000 --> 00:00:02.010
        Hello there. This

        00:00:02.010 --> 00:00:35.000
        Hello there. This
        is a complete thought that ends cleanly.
        """
    )
    chunks = chunk(vtt, [], video_id="vid1", target_sec=10.0)
    full = " ".join(c.text for c in chunks)
    assert full.count("Hello there.") == 1  # roll-up residue deduped
    assert "&gt;" not in full


def test_frame_secs_assigned_to_containing_chunk():
    frames = [Frame(5.0), Frame(50.0)]
    chunks = chunk(_vtt_multi_sentence(), frames, video_id="vid1", target_sec=30.0)
    assert chunks[0].frame_secs == [5.0]  # in [0, 32)
    assert chunks[1].frame_secs == [50.0]  # in [32, 65)


def test_registered_and_returns_chunks():
    fn = get_chunker(STRATEGY_NAME)
    assert callable(fn)
    out = fn(_vtt_multi_sentence(), [], video_id="vid1")
    assert all(isinstance(c, Chunk) for c in out)
    assert all(c.chunking_strategy == STRATEGY_NAME for c in out)
