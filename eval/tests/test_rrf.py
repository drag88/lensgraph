"""Closed-form synthetic tests for retrieve.rrf.fuse.

No DB, no model — just constructed ChannelResult objects with known
ranks and arithmetic against the RRF formula.
"""

from __future__ import annotations

import math

from retrieve.rrf import RRF_K, fuse
from retrieve.types import ChannelResult


def _cr(chunk_id: int, rank: int, *, score: float = 0.0) -> ChannelResult:
    return ChannelResult(
        chunk_id=chunk_id,
        video_id=f"v-{chunk_id}",
        start_sec=0.0,
        end_sec=1.0,
        text=f"chunk {chunk_id}",
        score=score,
        rank=rank,
    )


def test_empty_input_returns_empty():
    assert fuse({}) == []


def test_empty_per_channel_lists_returns_empty():
    assert fuse({"bm25": [], "dense": []}) == []


def test_single_channel_preserves_order():
    fused = fuse(
        {"bm25": [_cr(1, 1), _cr(2, 2), _cr(3, 3)]},
        k=60,
        top_k=8,
    )
    assert [r.chunk_id for r in fused] == [1, 2, 3]
    assert [r.rank for r in fused] == [1, 2, 3]
    assert math.isclose(fused[0].score, 1 / 61)
    assert math.isclose(fused[1].score, 1 / 62)
    assert math.isclose(fused[2].score, 1 / 63)
    assert fused[0].score > fused[1].score > fused[2].score


def test_two_channels_known_score():
    fused = fuse(
        {
            "bm25": [_cr(1, 1), _cr(2, 2)],
            "dense": [_cr(2, 1), _cr(1, 2)],
        },
        k=60,
    )
    expected = 1 / 61 + 1 / 62
    assert fused[0].chunk_id == 1
    assert fused[1].chunk_id == 2
    assert math.isclose(fused[0].score, expected)
    assert math.isclose(fused[1].score, expected)


def test_chunk_in_two_channels_beats_chunk_in_one():
    fused = fuse(
        {
            "bm25": [_cr(2, 1), _cr(1, 5)],
            "dense": [_cr(1, 5)],
        },
        k=60,
    )
    assert fused[0].chunk_id == 1
    assert math.isclose(fused[0].score, 2 * (1 / 65))
    assert fused[1].chunk_id == 2
    assert math.isclose(fused[1].score, 1 / 61)


def test_missing_channel_handled_correctly():
    fused = fuse(
        {
            "bm25": [_cr(1, 1), _cr(2, 2)],
            "dense": [_cr(1, 3)],
        },
        k=60,
    )
    assert fused[0].chunk_id == 1
    assert fused[1].chunk_id == 2
    assert math.isclose(fused[0].score, 1 / 61 + 1 / 63)
    assert math.isclose(fused[1].score, 1 / 62)


def test_top_k_truncates_output():
    channel = [_cr(i, i) for i in range(1, 11)]
    fused = fuse({"bm25": channel}, top_k=3)
    assert len(fused) == 3
    assert [r.chunk_id for r in fused] == [1, 2, 3]


def test_fused_ranks_are_1_indexed():
    fused = fuse(
        {"bm25": [_cr(10, 1), _cr(20, 2), _cr(30, 3), _cr(40, 4)]},
        top_k=4,
    )
    assert [r.rank for r in fused] == [1, 2, 3, 4]


def test_k_parameter_affects_curve_shape():
    channels = {"bm25": [_cr(1, 1), _cr(2, 10)]}
    sharp = fuse(channels, k=1)
    flat = fuse(channels, k=1000)

    sharp_gap = sharp[0].score - sharp[1].score
    flat_gap = flat[0].score - flat[1].score

    assert sharp_gap > flat_gap
    assert math.isclose(sharp[0].score, 1 / 2)
    assert math.isclose(sharp[1].score, 1 / 11)
    assert math.isclose(flat[0].score, 1 / 1001)
    assert math.isclose(flat[1].score, 1 / 1010)


def test_metadata_preserved_from_first_seen_channel():
    bm25 = [
        ChannelResult(
            chunk_id=42,
            video_id="vid-42",
            start_sec=10.5,
            end_sec=20.75,
            text="hello world",
            score=0.9,
            rank=1,
        )
    ]
    dense = [
        ChannelResult(
            chunk_id=42,
            video_id="vid-42",
            start_sec=10.5,
            end_sec=20.75,
            text="hello world",
            score=0.8,
            rank=2,
        )
    ]
    fused = fuse({"bm25": bm25, "dense": dense})
    assert len(fused) == 1
    assert fused[0].chunk_id == 42
    assert fused[0].video_id == "vid-42"
    assert fused[0].start_sec == 10.5
    assert fused[0].end_sec == 20.75
    assert fused[0].text == "hello world"
    assert math.isclose(fused[0].score, 1 / 61 + 1 / 62)
    assert fused[0].rank == 1


def test_rrf_k_default_constant():
    assert RRF_K == 60
