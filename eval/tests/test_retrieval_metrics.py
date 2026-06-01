"""Fast tests for GoldSpanContained@k and IoU@1 (pure functions, no DB)."""

from __future__ import annotations

import pytest

from eval.runners.measure_embeddings import (
    GoldQuery,
    channel_recall,
    gold_span_contained_at_k,
    iou_at_1,
    region_iou_at_k,
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


# -- Region-IoU@k ---------------------------------------------------------
#
# Region-scale localization: IoU of the UNION of the top-k chunks vs the gold
# region. Unlike IoU@1 (rank-1 only) it rewards tiling a region with several
# small chunks; unlike Contained@k it does not require a single wrapping chunk.
# This is the selector for the "clips are answer regions" methodology.


def test_region_iou_single_chunk_exact_match_is_one():
    assert region_iou_at_k([_r(100.0, 120.0)], _GOLD, k=5) == pytest.approx(1.0)


def test_region_iou_small_chunks_tiling_the_region_score_one():
    # two adjacent 10s chunks tile [100,120] exactly. Contained@k would be False
    # (no single chunk wraps it) and IoU@1 only 0.5 (sees rank-1 [100,110]).
    results = [_r(100.0, 110.0, rank=1), _r(110.0, 120.0, rank=2)]
    assert region_iou_at_k(results, _GOLD, k=5) == pytest.approx(1.0)


def test_region_iou_penalizes_oversized_union():
    # one 100s chunk around the 20s region -> inter 20, union 100 -> 0.2
    assert region_iou_at_k([_r(60.0, 160.0)], _GOLD, k=5) == pytest.approx(0.2)


def test_region_iou_penalizes_overhang_from_extra_chunks():
    # union [90,130]=40s vs 20s region: inter 20, union 40 -> 0.5
    results = [_r(90.0, 115.0, rank=1), _r(115.0, 130.0, rank=2)]
    assert region_iou_at_k(results, _GOLD, k=5) == pytest.approx(0.5)


def test_region_iou_merges_overlapping_chunks_without_double_count():
    # overlapping [100,115] + [110,120] merge to [100,120] -> exact -> 1.0
    results = [_r(100.0, 115.0, rank=1), _r(110.0, 120.0, rank=2)]
    assert region_iou_at_k(results, _GOLD, k=5) == pytest.approx(1.0)


def test_region_iou_respects_k_and_video_and_empty():
    # tiling chunks past k are excluded
    results = [_r(100.0, 110.0, rank=1)] + [_r(110.0, 120.0, rank=2)]
    assert region_iou_at_k(results, _GOLD, k=1) == pytest.approx(0.5)  # only [100,110]
    assert region_iou_at_k(results, _GOLD, k=2) == pytest.approx(1.0)
    # wrong-video chunks are filtered out of the union
    assert region_iou_at_k([_r(100.0, 120.0, video="OTHER")], _GOLD, k=5) == 0.0
    assert region_iou_at_k([], _GOLD, k=5) == 0.0


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
    assert out["region_iou_at_k"] == pytest.approx(0.25)  # (0.5 + 0.0)/2, single chunks
