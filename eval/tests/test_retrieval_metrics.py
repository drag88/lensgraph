"""Fast tests for GoldSpanContained@k and IoU@1 (pure functions, no DB)."""

from __future__ import annotations

import pytest

from eval.runners.measure_embeddings import (
    GoldQuery,
    channel_recall,
    gold_span_contained_at_k,
    iou_at_1,
)
from retrieve.types import ChannelResult


def _r(start, end, *, video="V", rank=1, cid=1):
    return ChannelResult(
        chunk_id=cid, video_id=video, start_sec=start, end_sec=end, text="t", score=1.0, rank=rank
    )


_GOLD = GoldQuery(example_id="ex1", question="q", video_id="V", start_sec=100.0, end_sec=120.0)


# -- GoldSpanContained@k --------------------------------------------------


def test_contained_true_when_a_chunk_fully_wraps_the_span():
    results = [_r(90.0, 130.0)]  # 90..130 contains 100..120
    assert gold_span_contained_at_k(results, _GOLD, k=5) is True


def test_contained_false_when_chunk_too_small():
    # a 10s chunk cannot contain a 20s span (over-segmentation penalty)
    results = [_r(100.0, 110.0), _r(110.0, 120.0)]
    assert gold_span_contained_at_k(results, _GOLD, k=5) is False


def test_contained_respects_k_and_video():
    # containing chunk is at rank 6 -> excluded at k=5
    results = [_r(0.0, 1.0, rank=i) for i in range(1, 6)] + [_r(90.0, 130.0, rank=6)]
    assert gold_span_contained_at_k(results, _GOLD, k=5) is False
    assert gold_span_contained_at_k(results, _GOLD, k=6) is True
    # wrong video never contains
    assert gold_span_contained_at_k([_r(90.0, 130.0, video="OTHER")], _GOLD, k=5) is False


# -- IoU@1 ----------------------------------------------------------------


def test_iou_exact_match_is_one():
    assert iou_at_1([_r(100.0, 120.0)], _GOLD) == pytest.approx(1.0)


def test_iou_penalizes_oversized_chunk():
    # 100s chunk [60,160] around a 20s span -> inter=20, union=100 -> 0.2
    assert iou_at_1([_r(60.0, 160.0)], _GOLD) == pytest.approx(0.2)


def test_iou_centered_double_width_is_half():
    # [90,130] (40s) vs [100,120] (20s): inter=20, union=40 -> 0.5
    assert iou_at_1([_r(90.0, 130.0)], _GOLD) == pytest.approx(0.5)


def test_iou_uses_rank_1_only_and_zero_when_empty():
    # rank-1 is a poor match even though rank-2 is exact
    results = [_r(0.0, 5.0, rank=1), _r(100.0, 120.0, rank=2)]
    assert iou_at_1(results, _GOLD) == pytest.approx(0.0)
    assert iou_at_1([], _GOLD) == 0.0
    assert iou_at_1([_r(100.0, 120.0, video="OTHER")], _GOLD) == 0.0


# -- channel_recall aggregation -------------------------------------------


def test_channel_recall_aggregates_all_three():
    examples = [
        GoldQuery("e1", "q1", "V", 100.0, 120.0),
        GoldQuery("e2", "q2", "V", 200.0, 220.0),
    ]

    def fn(conn, q):
        if q == "q1":
            return [_r(90.0, 130.0)]  # contains; iou 0.5
        return [_r(0.0, 5.0)]  # misses entirely; iou 0

    out = channel_recall(None, examples, fn, k=5)
    assert out["tr_at_k"] == pytest.approx(0.5)  # 1 of 2 overlaps
    assert out["gold_span_contained_at_k"] == pytest.approx(0.5)
    assert out["iou_at_1"] == pytest.approx(0.25)  # (0.5 + 0.0)/2
