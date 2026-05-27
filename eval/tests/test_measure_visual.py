"""Unit tests for visual eval metric primitives.

These exercise the metric logic with synthetic ``FrameResult`` /
``ChannelResult`` inputs — no DB, no model load. They are FAST and run
under ``make test``. The integration test that actually goes through
``retrieve.visual`` against a populated DB lives in
``test_run_visual_eval.py`` and is slow + auto-skipped when frames are
not ingested.
"""

from __future__ import annotations

import pytest

from eval.runners.measure_visual import (
    FRAME_SAMPLE_EVERY_SEC,
    VisualGoldQuery,
    chunk_passes_at_k,
    frame_passes_at_k,
    load_visual_gold,
    measure_visual,
    visual_chunk_tr_at_k,
    visual_frame_recall_at_k,
    visual_lift_at_k,
)
from retrieve.types import ChannelResult, FusedResult
from retrieve.visual import FrameResult


def _frame(*, frame_id=1, video_id="v1", frame_sec=170.0, rank=1) -> FrameResult:
    return FrameResult(
        frame_id=frame_id,
        video_id=video_id,
        frame_sec=frame_sec,
        image_path=f"/tmp/{frame_id}.png",
        score=1.0 / rank,
        rank=rank,
    )


def _chunk(*, chunk_id=1, video_id="v1", start=160.0, end=180.0, rank=1) -> ChannelResult:
    return ChannelResult(
        chunk_id=chunk_id,
        video_id=video_id,
        start_sec=start,
        end_sec=end,
        text="t",
        score=1.0 / rank,
        rank=rank,
    )


def _gold(start=170.0, end=180.0, video="v1", ex_id="ex-1") -> VisualGoldQuery:
    # Unique question per example so per-example stubs can route off it.
    return VisualGoldQuery(
        example_id=ex_id,
        question=f"q-{ex_id}",
        video_id=video,
        start_sec=start,
        end_sec=end,
    )


# ---- frame_passes_at_k ---------------------------------------------------


def test_frame_pass_when_frame_inside_span():
    g = _gold(170, 180)
    frames = [_frame(frame_sec=175)]
    assert frame_passes_at_k(frames, g, k=5)


def test_frame_pass_within_default_tolerance():
    """A frame at 165 should pass for a gold span [170, 180] under the
    default 10s tolerance (frame sample cadence)."""
    g = _gold(170, 180)
    frames = [_frame(frame_sec=165)]
    assert frame_passes_at_k(frames, g, k=5)
    assert FRAME_SAMPLE_EVERY_SEC == 10.0  # guard against silent change


def test_frame_fail_when_video_mismatch():
    g = _gold(170, 180, video="v1")
    frames = [_frame(frame_sec=175, video_id="v2")]
    assert not frame_passes_at_k(frames, g, k=5)


def test_frame_fail_when_outside_tolerance():
    g = _gold(170, 180)
    frames = [_frame(frame_sec=200)]
    assert not frame_passes_at_k(frames, g, k=5)


def test_frame_pass_only_within_top_k():
    g = _gold(170, 180)
    # Matching frame is at rank 6 — top-5 should miss it.
    frames = [_frame(frame_sec=400, rank=i) for i in range(1, 6)] + [
        _frame(frame_sec=175, rank=6)
    ]
    assert not frame_passes_at_k(frames, g, k=5)
    assert frame_passes_at_k(frames, g, k=6)


# ---- chunk_passes_at_k ---------------------------------------------------


def test_chunk_pass_open_interval_overlap():
    """Same open-interval convention as text TR@5: ``end > gold.start AND
    start < gold.end``. A chunk that exactly touches the boundary does NOT
    overlap because the inequality is strict."""
    g = _gold(170, 180)
    assert chunk_passes_at_k([_chunk(start=165, end=175)], g, k=5)
    assert chunk_passes_at_k([_chunk(start=175, end=185)], g, k=5)
    assert chunk_passes_at_k([_chunk(start=160, end=200)], g, k=5)  # super-set
    assert chunk_passes_at_k([_chunk(start=172, end=178)], g, k=5)  # sub-set
    # Strict boundary cases — touching but not overlapping.
    assert not chunk_passes_at_k([_chunk(start=180, end=190)], g, k=5)
    assert not chunk_passes_at_k([_chunk(start=160, end=170)], g, k=5)


def test_chunk_fail_when_video_mismatch():
    g = _gold(170, 180, video="v1")
    assert not chunk_passes_at_k([_chunk(start=165, end=175, video_id="v2")], g, k=5)


def test_chunk_works_with_fused_result_too():
    """visual_chunk_tr_at_k accepts the channel-level retrieve() output
    OR a fused list — both shapes carry video_id + start/end."""
    g = _gold(170, 180)
    fused = FusedResult(
        chunk_id=1,
        video_id="v1",
        start_sec=165,
        end_sec=175,
        text="t",
        score=0.5,
        rank=1,
        channel_ranks={"visual": 1},
    )
    assert chunk_passes_at_k([fused], g, k=5)


# ---- aggregate runners (injectable channel functions) --------------------


def test_visual_frame_recall_aggregates():
    examples = [_gold(170, 180, ex_id="a"), _gold(300, 320, ex_id="b")]

    def stub_frames(_conn, q):
        # Example a's question routes to a passing frame; example b's misses.
        return [_frame(frame_sec=175)] if q == examples[0].question else [_frame(frame_sec=500)]

    score, passes, top_ks = visual_frame_recall_at_k(
        conn=None, examples=examples, k=5, frame_fn=stub_frames
    )
    assert score == 0.5
    assert passes == [True, False]
    assert len(top_ks) == 2 and len(top_ks[0]) == 1


def test_visual_chunk_tr_aggregates():
    examples = [_gold(170, 180, ex_id="a"), _gold(300, 320, ex_id="b")]

    def stub_chunks(_conn, q):
        return [_chunk(start=165, end=175)] if q == examples[0].question else [
            _chunk(start=400, end=410)
        ]

    score, passes, _top_ks = visual_chunk_tr_at_k(
        conn=None, examples=examples, k=5, chunk_fn=stub_chunks
    )
    assert score == 0.5
    assert passes == [True, False]


# ---- measure_visual end-to-end (monkeypatched DB layer) ------------------


def test_measure_visual_empty_examples_returns_zeros():
    summary, detail = measure_visual(conn=None, examples=[], include_lift=False)
    assert summary["n_examples"] == 0
    assert summary["visual_frame_recall_at_k"] == 0.0
    assert summary["visual_chunk_tr_at_k"] == 0.0
    assert detail == {}


def test_measure_visual_shape_matches_runner_contract(monkeypatch):
    """measure_visual must return the same (summary, per_example_detail)
    tuple shape that scripts/run_visual_eval.py consumes — same
    contract as measure_all_channels in bakeoff #1."""
    examples = [_gold(170, 180, ex_id="a")]

    from eval.runners import measure_visual as mod
    from retrieve import visual as visual_mod

    monkeypatch.setattr(
        visual_mod, "retrieve_frames", lambda _c, _q, top_k: [_frame(frame_sec=175)]
    )
    monkeypatch.setattr(
        visual_mod, "retrieve", lambda _c, _q, top_k: [_chunk(start=165, end=175)]
    )

    summary, detail = mod.measure_visual(conn=None, examples=examples, include_lift=False)
    assert summary["n_examples"] == 1
    assert summary["visual_frame_recall_at_k"] == 1.0
    assert summary["visual_chunk_tr_at_k"] == 1.0
    assert set(summary["per_example"]["a"]) == {"frame_pass", "chunk_pass"}
    assert set(detail["a"]) == {
        "gold_span",
        "frame_pass_at_k",
        "chunk_pass_at_k",
        "top_k_frames",
        "top_k_chunks",
    }


# ---- visual_lift_at_k -----------------------------------------------------


def test_visual_lift_empty_examples():
    out = visual_lift_at_k(conn=None, examples=[], k=5)
    assert out["with_visual_pass_rate"] == 0.0
    assert out["lift_pp"] == 0.0
    assert out["per_example"] == {}


def test_visual_lift_computes_per_example_and_aggregate(monkeypatch):
    """Lift = (with_visual_pass_rate - without_visual_pass_rate) * 100.
    Patch both internal rrf helpers to control pass/fail per example."""
    examples = [
        _gold(170, 180, ex_id="a"),
        _gold(300, 320, ex_id="b"),
        _gold(500, 520, ex_id="c"),
    ]

    from eval.runners import measure_visual as mod

    # 'a': visual helps (with passes, without fails).
    # 'b': both pass (visual neither helps nor hurts).
    # 'c': both fail.
    with_visual_map = {"a": True, "b": True, "c": False}
    without_visual_map = {"a": False, "b": True, "c": False}

    def stub_with(_c, q, *, k):
        ex = next(e for e in examples if e.question == q)
        # Build a fake fused result that overlaps the gold iff with_visual_map[ex] is True.
        if with_visual_map[ex.example_id]:
            return [_chunk(start=ex.start_sec + 1, end=ex.end_sec - 1)]
        return []

    def stub_without(_c, q, *, k):
        ex = next(e for e in examples if e.question == q)
        if without_visual_map[ex.example_id]:
            return [_chunk(start=ex.start_sec + 1, end=ex.end_sec - 1)]
        return []

    monkeypatch.setattr(mod, "_rrf_with_visual", stub_with)
    monkeypatch.setattr(mod, "_rrf_text_only", stub_without)

    out = visual_lift_at_k(conn=None, examples=examples, k=5)
    assert out["with_visual_pass_rate"] == pytest.approx(2 / 3)
    assert out["without_visual_pass_rate"] == pytest.approx(1 / 3)
    assert out["lift_pp"] == pytest.approx(33.33, abs=0.01)
    assert out["per_example"]["a"] == {"with_visual": True, "without_visual": False}
    assert out["per_example"]["b"] == {"with_visual": True, "without_visual": True}


# ---- load_visual_gold ----------------------------------------------------


def test_load_visual_gold_handles_empty_file(tmp_path):
    p = tmp_path / "visual_gold.jsonl"
    p.write_text("")
    assert load_visual_gold(p) == []


def test_load_visual_gold_skips_non_single_clip(tmp_path):
    """synthesis and negative rows are skipped — visual eval scores
    single_clip only (synthesis would need per-span video gold and is
    out of scope for the v0 visual gate)."""
    import json as _json

    p = tmp_path / "visual_gold.jsonl"
    rows = [
        {
            "id": "vis-1",
            "question": "?" * 16,
            "video_id": "v1",
            "split": "dev",
            "question_type": "single_clip",
            "gold_spans": [{"start_sec": 100, "end_sec": 120}],
            "modality": ["slide"],
            "difficulty": "easy",
            "curator": "tester",
            "curated_at": "2026-05-27T00:00:00Z",
            "verified": True,
        },
        {
            "id": "syn-1",
            "question": "?" * 16,
            "split": "dev",
            "question_type": "synthesis",
            "gold_spans": [
                {"start_sec": 100, "end_sec": 120, "video_id": "v1"},
                {"start_sec": 200, "end_sec": 220, "video_id": "v2"},
            ],
            "modality": ["slide"],
            "difficulty": "easy",
            "curator": "tester",
            "curated_at": "2026-05-27T00:00:00Z",
            "verified": True,
        },
    ]
    p.write_text("\n".join(_json.dumps(r) for r in rows))
    loaded = load_visual_gold(p)
    assert [q.example_id for q in loaded] == ["vis-1"]
