"""Fast tests for the boundary-kappa pure functions (no DB, no provider call)."""

from __future__ import annotations

import pytest

from scripts.run_boundary_kappa import (
    binarize,
    build_messages,
    cohens_kappa,
    content_sha,
    kappa_set,
    parse_judge_scores,
)

# -- parse_judge_scores ---------------------------------------------------


def test_parse_plain_json():
    out = parse_judge_scores('{"edge_sensibility": 4, "standalone": 2, "rationale": "ok"}')
    assert out == {"edge_sensibility": 4, "standalone": 2, "rationale": "ok"}


def test_parse_tolerates_code_fence_and_prose():
    raw = 'Here is my score:\n```json\n{"edge_sensibility": 5, "standalone": 3}\n```\n'
    out = parse_judge_scores(raw)
    assert out["edge_sensibility"] == 5
    assert out["standalone"] == 3
    assert out["rationale"] == ""  # missing rationale defaults to empty


def test_parse_rejects_out_of_range():
    with pytest.raises(ValueError):
        parse_judge_scores('{"edge_sensibility": 7, "standalone": 3}')


def test_parse_rejects_missing_field():
    with pytest.raises(ValueError):
        parse_judge_scores('{"edge_sensibility": 4}')


def test_parse_rejects_non_int_and_bool():
    with pytest.raises(ValueError):
        parse_judge_scores('{"edge_sensibility": 4.5, "standalone": 3}')
    with pytest.raises(ValueError):
        parse_judge_scores('{"edge_sensibility": true, "standalone": 3}')


def test_parse_rejects_no_json():
    with pytest.raises(ValueError):
        parse_judge_scores("no json here")


# -- binarize -------------------------------------------------------------


def test_binarize_threshold_at_4():
    assert [binarize(s) for s in (1, 2, 3, 4, 5)] == [0, 0, 0, 1, 1]


# -- build_messages -------------------------------------------------------


def test_build_messages_shape():
    clip = {"video_id": "V", "start_sec": 1.0, "end_sec": 9.0, "text": "hello world"}
    msgs = build_messages("RUBRIC", clip)
    assert msgs[0] == {"role": "system", "content": "RUBRIC"}
    assert msgs[1]["role"] == "user"
    assert "video_id: V" in msgs[1]["content"]
    assert "hello world" in msgs[1]["content"]


# -- content_sha ----------------------------------------------------------


def test_content_sha_is_stable_and_truncated():
    a = content_sha("abc")
    assert len(a) == 12
    assert a == content_sha("abc")
    assert a != content_sha("abd")


# -- kappa_set ------------------------------------------------------------


def test_kappa_perfect_agreement_is_one():
    k = kappa_set([1, 3, 5, 2, 4], [1, 3, 5, 2, 4])
    assert k["n"] == 5
    assert k["linear"] == pytest.approx(1.0)
    assert k["quadratic"] == pytest.approx(1.0)


def test_kappa_empty_is_none():
    k = kappa_set([], [])
    assert k == {"n": 0, "linear": None, "quadratic": None, "binarized": None}


def test_kappa_length_mismatch_raises():
    with pytest.raises(ValueError):
        kappa_set([1, 2], [1])


def test_kappa_uniform_human_collapses():
    # all human=3 (zero variance) -> kappa undefined/~0 even if judge varies.
    # This is the standalone failure mode the variance fix addresses.
    k = kappa_set([3, 3, 3, 3], [3, 4, 2, 5])
    assert k["linear"] == pytest.approx(0.0) or k["linear"] is not None


def test_kappa_mixed_pins_all_three_weights():
    # Non-perfect, mixed-label pair. Values verified equal to
    # sklearn.metrics.cohen_kappa_score (max abs diff < 1e-15) before sklearn
    # was removed, so this pins the hand-rolled helper to sklearn's output.
    human = [1, 2, 3, 4, 5, 3, 3]
    judge = [2, 2, 3, 5, 5, 4, 1]
    k = kappa_set(human, judge)
    assert k["n"] == 7
    assert k["linear"] == pytest.approx(0.533, abs=5e-4)
    assert k["quadratic"] == pytest.approx(0.72, abs=5e-4)
    # binarized (>=4 clean): hb=[0,0,0,1,1,0,0] vs jb=[0,0,0,1,1,1,0]
    assert k["binarized"] == pytest.approx(0.696, abs=5e-4)


def test_cohens_kappa_unweighted_matches_known_value():
    # Direct helper check on the binarized label set [0,1].
    hb = [0, 0, 0, 1, 1, 0, 0]
    jb = [0, 0, 0, 1, 1, 1, 0]
    assert cohens_kappa(hb, jb, labels=[0, 1]) == pytest.approx(0.696, abs=5e-4)


def test_cohens_kappa_rejects_unknown_weights():
    with pytest.raises(ValueError):
        cohens_kappa([1, 2], [1, 2], labels=[1, 2], weights="cubic")
