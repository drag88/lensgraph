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
    VisualEvidenceItem,
    VisualGoldQuery,
    answer_term_hit_at_k,
    answer_term_matches_at_k,
    chunk_passes_at_k,
    chunk_text_contains_evidence,
    frame_passes_at_k,
    load_visual_evidence,
    load_visual_gold,
    measure_visual,
    normalize_text,
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


def _chunk(
    *, chunk_id=1, video_id="v1", start=160.0, end=180.0, rank=1, text="t"
) -> ChannelResult:
    return ChannelResult(
        chunk_id=chunk_id,
        video_id=video_id,
        start_sec=start,
        end_sec=end,
        text=text,
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
    # `answer_term_hit` is None when evidence_by_example is not supplied — the
    # row is not evaluable, not "failed". Tests for the populated case
    # live further down (see test_answer_term_hit_at_k_*).
    assert set(summary["per_example"]["a"]) == {"frame_pass", "chunk_pass", "answer_term_hit"}
    assert summary["per_example"]["a"]["answer_term_hit"] is None
    assert set(detail["a"]) == {
        "gold_span",
        "frame_pass_at_k",
        "chunk_pass_at_k",
        "answer_term_hit_at_k",
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
    # `lift_pp` is the back-compat alias for `lift_chunk_tr_pp`; both
    # should agree byte-for-byte for the bakeoff-#1 atomicity contract.
    assert out["lift_pp"] == pytest.approx(33.33, abs=0.01)
    assert out["lift_chunk_tr_pp"] == out["lift_pp"]
    # With no evidence_by_example, the answer-hit tier is absent (None).
    assert out["lift_answer_term_pp"] is None
    assert out["answer_term_n_evaluable"] == 0
    assert out["per_example"]["a"] == {
        "with_visual": True,
        "without_visual": False,
        "answer_term_with_visual": None,
        "answer_term_without_visual": None,
    }
    assert out["per_example"]["b"] == {
        "with_visual": True,
        "without_visual": True,
        "answer_term_with_visual": None,
        "answer_term_without_visual": None,
    }


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


def test_load_visual_gold_rejects_transcript_only_modality(tmp_path):
    """A visual_gold entry must include at least one of {slide,
    screen_code, diagram, whiteboard}. A transcript-only entry is a
    measurement-mode mistake — it belongs in dev_gold."""
    import json as _json

    p = tmp_path / "visual_gold.jsonl"
    p.write_text(
        _json.dumps(
            {
                "id": "bad-vis",
                "question": "?" * 16,
                "video_id": "v1",
                "split": "dev",
                "question_type": "single_clip",
                "gold_spans": [{"start_sec": 100, "end_sec": 120}],
                "modality": ["transcript"],  # NO visual tag
                "difficulty": "easy",
                "curator": "tester",
                "curated_at": "2026-05-27T00:00:00Z",
                "verified": True,
            }
        )
    )
    with pytest.raises(ValueError, match="lacks any of"):
        load_visual_gold(p)


def test_load_visual_gold_accepts_visual_plus_transcript(tmp_path):
    """A slide-bearing example may also tag transcript — the rule is
    'at least one visual tag', not 'visual only'."""
    import json as _json

    p = tmp_path / "visual_gold.jsonl"
    p.write_text(
        _json.dumps(
            {
                "id": "vis-and-text",
                "question": "?" * 16,
                "video_id": "v1",
                "split": "dev",
                "question_type": "single_clip",
                "gold_spans": [{"start_sec": 100, "end_sec": 120}],
                "modality": ["slide", "transcript"],
                "difficulty": "easy",
                "curator": "tester",
                "curated_at": "2026-05-27T00:00:00Z",
                "verified": True,
            }
        )
    )
    loaded = load_visual_gold(p)
    assert [q.example_id for q in loaded] == ["vis-and-text"]


# ---- resolve_visual_candidate_id -----------------------------------------


def test_resolve_visual_candidate_id_happy_path():
    """The yaml in the repo today has exactly one local candidate
    (colqwen2.5) under candidates.visual_retrieval.options."""
    import yaml as _yaml

    from eval.runners.run_visual_eval import _CONFIG_PATH, resolve_visual_candidate_id

    cfg = _yaml.safe_load(_CONFIG_PATH.read_text())
    assert resolve_visual_candidate_id(cfg) == "colqwen2.5"


def test_resolve_visual_candidate_id_zero_local_raises():
    from eval.runners.run_visual_eval import resolve_visual_candidate_id

    cfg = {
        "candidates": {
            "visual_retrieval": {
                "options": [
                    {"id": "hosted-only", "provider": "google", "family": "x"},
                ]
            }
        }
    }
    with pytest.raises(RuntimeError, match="Expected exactly 1 local"):
        resolve_visual_candidate_id(cfg)


def test_resolve_visual_candidate_id_two_local_raises():
    from eval.runners.run_visual_eval import resolve_visual_candidate_id

    cfg = {
        "candidates": {
            "visual_retrieval": {
                "options": [
                    {"id": "colqwen-a", "provider": "local", "family": "colpali"},
                    {"id": "colqwen-b", "provider": "local", "family": "colpali"},
                ]
            }
        }
    }
    with pytest.raises(RuntimeError, match="Expected exactly 1 local"):
        resolve_visual_candidate_id(cfg)


def test_scripts_shim_delegates_to_canonical_main():
    """The scripts/run_visual_eval.py back-compat shim must import
    main() from eval.runners.run_visual_eval, NOT define its own."""
    import scripts.run_visual_eval as shim
    from eval.runners.run_visual_eval import main as canonical_main

    assert shim.main is canonical_main, (
        "shim main must be the canonical eval.runners.run_visual_eval.main "
        "(no duplicated logic)"
    )


# ---- validator: visual_gold modality enforcement -------------------------


def test_validator_rejects_visual_gold_without_visual_modality(tmp_path, monkeypatch, capsys):
    """validate_corpora must reject a visual_gold.jsonl entry that has
    no visual modality tag — symmetric with load_visual_gold's runtime
    guard so the rule fires at commit time AND at runtime."""
    import json as _json

    import yaml as _yaml

    from eval import validate as v

    corpus = tmp_path / "corpora" / "test_visual_modality"
    corpus.mkdir(parents=True)
    transcript = tmp_path / "transcripts" / "v1.vtt"
    transcript.parent.mkdir(parents=True)
    transcript.write_text("WEBVTT\n\n00:00.000 --> 00:01.000\nfake\n")
    import hashlib as _hashlib

    sha = _hashlib.sha256(transcript.read_bytes()).hexdigest()
    (corpus / "talks.yaml").write_text(
        _yaml.safe_dump(
            [
                {
                    "video_id": "v1",
                    "title": "t",
                    "speaker": "s",
                    "url": "https://example.com/v1",
                    "duration_sec": 60,
                    "format_tags": ["slides_heavy"],
                    "license": "youtube_standard",
                    "captions_source": "youtube_auto",
                    "transcript_path": str(transcript.relative_to(tmp_path)),
                    "transcript_sha256": sha,
                    "accessed_at": "2026-05-27T00:00:00Z",
                }
            ]
        )
    )
    bad_visual = {
        "id": "bad-vis",
        "question": "?" * 16,
        "video_id": "v1",
        "split": "dev",
        "question_type": "single_clip",
        "gold_spans": [{"start_sec": 100, "end_sec": 120}],
        "modality": ["transcript"],  # no visual tag
        "difficulty": "easy",
        "curator": "tester",
        "curated_at": "2026-05-27T00:00:00Z",
        "verified": True,
    }
    (corpus / "visual_gold.jsonl").write_text(_json.dumps(bad_visual))

    monkeypatch.setattr(v, "CORPORA_DIR", tmp_path / "corpora")
    monkeypatch.setattr(v, "PROJECT_ROOT", tmp_path)
    # The model_candidates check looks at v.CONFIG_DIR for the yaml; the
    # real project yaml passes, so leave it alone.

    rc = v.validate_corpora(strict=False)
    out = capsys.readouterr().out
    assert rc != 0, f"validator should fail; output was:\n{out}"
    assert "visual_gold entry" in out and "lacks any of" in out, out


# ---- normalize_text ------------------------------------------------------


def test_normalize_text_lowercase_collapse_ws():
    assert (
        normalize_text("  Retrieval  Quality  (NDCG@10)  ", "lowercase_collapse_ws")
        == "retrieval quality (ndcg@10)"
    )


def test_normalize_text_exact_preserves_case_and_ws():
    assert normalize_text("is_resumable=True", "exact") == "is_resumable=True"
    # Whitespace inside is preserved too — `exact` is byte-for-byte.
    assert normalize_text("  hello   World  ", "exact") == "  hello   World  "


def test_normalize_text_unknown_mode_raises_valueerror():
    with pytest.raises(ValueError, match="unknown normalize mode"):
        normalize_text("anything", "title_case")


# ---- chunk_text_contains_evidence ----------------------------------------


def test_chunk_text_contains_evidence_single_item_disjunction():
    """One item with multiple required_any — chunk passes if ANY term hits."""
    item = VisualEvidenceItem(
        visual_element="plot x-axis",
        required_any=("price per million tokens", "tokens-per-second"),
    )
    # Only the first term appears → still passes (disjunction).
    assert chunk_text_contains_evidence(
        "the slide shows Price per Million Tokens on the x-axis", [item]
    )
    # Neither term appears → fails.
    assert not chunk_text_contains_evidence(
        "the speaker discusses retrieval at length", [item]
    )


def test_chunk_text_contains_evidence_multi_item_conjunction():
    """Two items — chunk passes only if BOTH items have a term that hits.
    (Across items is conjunction; within an item is disjunction.)"""
    items = [
        VisualEvidenceItem(visual_element="y-axis label", required_any=("NDCG@10",)),
        VisualEvidenceItem(visual_element="x-axis label", required_any=("price per million tokens",)),
    ]
    # Both axis labels mentioned → passes.
    assert chunk_text_contains_evidence(
        "ndcg@10 on the y-axis and price per million tokens on the x-axis",
        items,
    )
    # Only one axis mentioned → fails (conjunction).
    assert not chunk_text_contains_evidence(
        "ndcg@10 on the y-axis but no axis label given for x", items
    )


def test_chunk_text_contains_evidence_exact_normalize_case_sensitive():
    """`exact` mode is byte-for-byte — case matters."""
    item = VisualEvidenceItem(
        visual_element="code identifier",
        required_any=("is_resumable=True",),
        normalize="exact",
    )
    assert chunk_text_contains_evidence("we set is_resumable=True in the config", [item])
    # Different case → fails under `exact`.
    assert not chunk_text_contains_evidence(
        "we set IS_RESUMABLE=true in the config", [item]
    )


# ---- answer_term_matches_at_k -----------------------------------------------------


def test_answer_term_matches_at_k_requires_span_overlap_AND_terms():  # noqa: N802 — AND signals logical conjunction
    """The headline bug fix: a topically-similar chunk on the WRONG span
    that happens to contain the term must NOT pass answer_term_matches_at_k, even
    though it would pass the span-only chunk_passes_at_k.

    Two competing chunks:
      * On-span (overlaps gold) but no answer-bearing text.
      * Off-span (no overlap) but contains the required term.
    Neither should pass — the metric is an AND of the two gates."""
    g = _gold(170, 180)
    evidence = [VisualEvidenceItem(visual_element="y-axis", required_any=("NDCG@10",))]
    on_span_no_terms = _chunk(start=175, end=178, text="discussion about retrieval", rank=1)
    off_span_with_terms = _chunk(start=400, end=410, text="ndcg@10 is the metric", rank=2)
    assert not answer_term_matches_at_k([on_span_no_terms, off_span_with_terms], g, evidence, k=5)


def test_answer_term_matches_at_k_passes_when_both_gates_pass():
    g = _gold(170, 180)
    evidence = [VisualEvidenceItem(visual_element="y-axis", required_any=("NDCG@10",))]
    chunk = _chunk(start=175, end=178, text="the y-axis shows ndcg@10 quality", rank=1)
    assert answer_term_matches_at_k([chunk], g, evidence, k=5)


def test_answer_term_matches_at_k_empty_evidence_returns_false():
    """Undefined-by-design — callers must gate this at the aggregator
    level. Returning False is the safe default so a stray empty-evidence
    row never silently inflates the numerator."""
    g = _gold(170, 180)
    chunk = _chunk(start=175, end=178, text="anything", rank=1)
    assert not answer_term_matches_at_k([chunk], g, [], k=5)


def test_answer_term_matches_distinguishes_topical_chunk_from_answer_bearing_chunk():
    """The headline regression test. Both chunks overlap [2010, 2030] on
    the same video; one is topical only ("Shubam explains resume tracks
    tools"), the other contains the curator-required answer-bearing
    string ("resumability_config=ResumabilityConfig(is_resumable=True)").

    * chunk_passes_at_k → both pass (they both overlap the gold span).
    * answer_term_matches_at_k → only the answer-bearing chunk passes."""
    g = _gold(2010, 2030, video="nXafozNIk3c", ex_id="resumability")
    topical = _chunk(
        video_id="nXafozNIk3c",
        start=2010,
        end=2020,
        text="Shubam explains that resume tracks tools that already ran",
        rank=1,
    )
    answer_bearing = _chunk(
        chunk_id=2,
        video_id="nXafozNIk3c",
        start=2020,
        end=2030,
        text=(
            "the code wires resumability_config=ResumabilityConfig(is_resumable=True)"
        ),
        rank=2,
    )
    # Sanity: both pass the span-only metric.
    assert chunk_passes_at_k([topical], g, k=5)
    assert chunk_passes_at_k([answer_bearing], g, k=5)
    # The new metric distinguishes them.
    evidence = [
        VisualEvidenceItem(
            visual_element="code identifier",
            required_any=("is_resumable=True", "ResumabilityConfig"),
        )
    ]
    assert not answer_term_matches_at_k([topical], g, evidence, k=5)
    assert answer_term_matches_at_k([answer_bearing], g, evidence, k=5)


# ---- answer_term_hit_at_k (aggregator) ---------------------------------


def test_answer_term_hit_at_k_skips_rows_without_evidence_in_denominator():
    """A row with no evidence contributes None to per_example_passes and
    is not counted in the numerator OR the denominator. The aggregate
    score therefore reflects ONLY rows with curator-supplied evidence."""
    examples = [
        _gold(170, 180, ex_id="a"),  # evidence present, will pass
        _gold(300, 320, ex_id="b"),  # NO evidence — skipped
        _gold(500, 520, ex_id="c"),  # evidence present, will fail
    ]
    evidence_by_example: dict[str, list[VisualEvidenceItem]] = {
        "a": [VisualEvidenceItem(visual_element="x", required_any=("hit",))],
        "c": [VisualEvidenceItem(visual_element="x", required_any=("missing-term",))],
    }

    def stub(_c, q):
        ex = next(e for e in examples if e.question == q)
        return [
            _chunk(
                start=ex.start_sec + 1,
                end=ex.end_sec - 1,
                text="this chunk contains the HIT term",
            )
        ]

    score, passes, top_ks = answer_term_hit_at_k(
        conn=None,
        examples=examples,
        evidence_by_example=evidence_by_example,
        k=5,
        chunk_fn=stub,
    )
    # Denominator = 2 (rows a and c). Numerator = 1 (only row a hit).
    assert score == 0.5
    assert passes == [True, None, False]
    assert len(top_ks) == 3
    # Even the skipped row gets its top-k captured for the methodology MDX.
    assert len(top_ks[1]) == 1


def test_answer_term_hit_at_k_all_rows_lack_evidence_returns_zero():
    """When no rows are evaluable the headline score collapses to 0.0
    but per-example slots are all None (so a caller can tell "no signal"
    apart from "every row failed")."""
    examples = [_gold(170, 180, ex_id="a"), _gold(300, 320, ex_id="b")]

    def stub(_c, q):
        return [_chunk(start=171, end=179, text="anything")]

    score, passes, top_ks = answer_term_hit_at_k(
        conn=None,
        examples=examples,
        evidence_by_example={},
        k=5,
        chunk_fn=stub,
    )
    assert score == 0.0
    assert passes == [None, None]
    assert len(top_ks) == 2


# ---- load_visual_evidence ------------------------------------------------


def test_load_visual_evidence_parses_committed_visual_gold():
    """The committed visual_gold.jsonl carries `visual_evidence` on the
    10 `visual-required-*` rows that drive the v2 metric. Curator-side
    work lands the field; the loader must surface exactly those rows.

    If the curator task has not finished yet, this test will fail with
    a clear count mismatch (10 expected, fewer present) — that's the
    intended tripwire, not a flake."""
    from eval.runners.measure_visual import _VISUAL_GOLD_PATH

    out = load_visual_evidence(_VISUAL_GOLD_PATH)
    assert len(out) == 10, (
        "expected 10 visual-required rows with visual_evidence; got "
        f"{len(out)} ({sorted(out)})"
    )
    # Every loaded key should begin with the visual-required prefix per
    # the curator playbook.
    assert all(k.startswith("visual-required-") for k in out), sorted(out)


def test_load_visual_evidence_empty_required_any_raises_valueerror(tmp_path):
    """A typo that produces required_any=[] must fail loudly — the
    metric would otherwise vacuously pass every chunk."""
    import json as _json

    p = tmp_path / "visual_gold.jsonl"
    p.write_text(
        _json.dumps(
            {
                "id": "vis-bad",
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
                "visual_evidence": [
                    {"visual_element": "plot legend", "required_any": []}
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="empty required_any"):
        load_visual_evidence(p)
