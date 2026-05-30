# Design: VisualAnswerGrounding@k audit payload

Status: Proposed (design only — not implemented this session).
Date: 2026-05-29.
Replaces: the bare `bool` returned by `visual_answer_grounding_at_k` and written
to `eval_results.metrics.visual_answer_grounding_at_k`.

Symbols below were verified against the repo this session:
`run_visual_eval._write_per_example_results` (line 117, with the `TODO(v4)` at
127), `eval_runs.insert_result` (already `Jsonb`-wraps `metrics`),
`measure_visual.frame_ocr_matches_at_k` (389), the `OcrFn` seam (386), and
`frame_ocr_text` (361).

## Why this exists

The 2026-05-28 headline run scored `VisualAnswerGrounding@5 = 0/10`. That result
is defensible without an audit payload precisely because nothing claimed a pass
(integrity_check.md §B, "PASS trivially"). The moment a future run produces even
one `vag = true` row, the boolean is no longer enough: §B's spot-check (which
frame, which OCR text, which term hit) has nothing to read. This payload makes
every positive row self-defending — you can reconstruct exactly which on-disk PNG
was OCR'd and which curator term matched, without re-running the pipeline.

## Decision: option (a) — payload lives in `eval_results.metrics.visual_answer_grounding`

Choose **(a)**: a JSONB sub-object at `metrics.visual_answer_grounding`, replacing
today's scalar `metrics.visual_answer_grounding_at_k`. Reject **(b)**
(`system_output.ocr_audit` alongside `top_k_frames`).

1. **The audit is a metric, not a system output.** `system_output.top_k_frames`
   is what the *retriever* returned — the raw ranked list, identical regardless of
   which gold row it is scored against. The OCR evaluation, the in-window filter,
   the term match, and `passed` are all *judgments the metric makes about* that
   output against curator evidence. Mixing the judgment into `system_output` blurs
   the retriever/scorer boundary the other three metrics (`frame_pass_at_k`,
   `chunk_pass_at_k`, `answer_term_hit_at_k`) already respect by living in
   `metrics`.
2. **§B's audit query targets `metrics`.** It is written as
   `rows where metrics.visual_answer_grounding_at_k = true`. Keeping the payload
   under `metrics.visual_answer_grounding.passed` keeps that query one field from
   its current form and co-locates the headline scalar and its evidence.
3. **No duplication of frame data.** `evaluated_frame_ids` is a *subset* of the
   frame IDs already in `system_output.top_k_frames`; the paths and excerpts key
   off those IDs. Putting the audit in `system_output` invites two parallel frame
   lists in one blob. Under (a) the audit references the in-window subset by ID
   and the reader joins back to `top_k_frames` for rank/score.

## Payload shape

`metrics.visual_answer_grounding` (JSONB object; replaces the scalar key):

```json
{
  "passed": false,
  "evaluated_frame_ids": [4821, 4830],
  "evaluated_image_paths": ["frames/vid123/0480.png", "frames/vid123/0490.png"],
  "ocr_excerpts": {"4821": "Attention is all you...", "4830": "Q K V softmax..."},
  "matched_term": null,
  "failure_reason": "ocr_no_terms_matched",
  "judge_kind": "ocr"
}
```

- `passed: bool` — mirrors the old scalar. `null` is never written here; a
  non-evaluable row writes the whole sub-object as `null`, matching how
  `metrics.answer_term_hit_at_k` already carries `None`.
- `evaluated_frame_ids: [int]` — the in-window (`gold_span ± tolerance_sec`, right
  `video_id`) top-k frame IDs the judge actually ran against. Empty `[]` when no
  frame passed the window gate.
- `evaluated_image_paths: [str]` — on-disk PNG paths, index-aligned to
  `evaluated_frame_ids`.
- `ocr_excerpts: {str(frame_id): str}` — first 500 chars of OCR per evaluated
  frame. JSONB keys are strings; the reader casts. OCR-path field.
- `matched_term: str | null` — the specific `required_any` term that hit, or
  `null` when `passed=false`.
- `failure_reason`: one of `"no_in_window_frame"` (window gate emptied the
  candidate set — the actual 0/10 cause today), `"ocr_no_terms_matched"` (frames
  evaluated, no term hit), `"row_not_evaluable"` (no curator evidence), or `null`
  when `passed=true`.
- `judge_kind: "ocr" | "vlm"` — discriminator for the VLM-judge path. Not in §B's
  minimum list; added so the two paths never silently overwrite each other's
  evidence field.

## Function signature change

`visual_answer_grounding_at_k` today returns
`tuple[float, list[bool | None], list[list[dict]]]`. Change the per-example
element from `bool | None` to `VisualGroundingAudit | None`:

```python
@dataclass(frozen=True)
class VisualGroundingAudit:
    passed: bool
    evaluated_frame_ids: tuple[int, ...]
    evaluated_image_paths: tuple[str, ...]
    ocr_excerpts: dict[int, str]          # int keys in-process; stringified at JSONB write
    matched_term: str | None
    failure_reason: str | None            # four-value enum, or None on pass
    judge_kind: str = "ocr"

def visual_answer_grounding_at_k(
    conn, examples, evidence_by_example, *,
    k=5, tolerance_sec=FRAME_SAMPLE_EVERY_SEC, frame_fn=None, ocr_fn=None,
) -> tuple[float, list[VisualGroundingAudit | None], list[list[dict]]]:
```

`score` (float) and the third element (`top_k_frames` dicts) are unchanged. The
aggregator loop swaps `frame_ocr_matches_at_k` (returns `bool`) for a new
`frame_ocr_grounding_at_k` that returns the audit; the two share the
window/normalization logic, and the new one additionally records
`evaluated_frame_ids`, captures excerpts, and reports
`matched_term`/`failure_reason`. The `int(hit)` numerator becomes
`int(audit.passed)`; the `None` (no-evidence) branch is unchanged.

`measure_visual`'s `summary` assembly: keep `summary["visual_answer_grounding_at_k"]`
(the float) and `n_evaluable`/`n_skipped` as-is. In `per_example_detail`, replace
the `visual_answer_grounding_at_k: vg` bool with
`visual_answer_grounding: <audit-dict-or-None>` produced by a small
`_grounding_to_dict(audit)` (stringifies `ocr_excerpts` keys, tuples → lists).
Keep the headline `per_example[...]["visual_answer_grounding"]` boolean
(`audit.passed if audit else None`) so the existing summary contract and any bool
consumer keep working.

## Where the payload is assembled and written

Assembled in `visual_answer_grounding_at_k` (the dataclass), serialized in
`measure_visual` (`_grounding_to_dict`), persisted in
`run_visual_eval._write_per_example_results`. The `TODO(v4)` block there changes
from writing a scalar to:

```python
metrics = {
    "frame_pass_at_k": detail["frame_pass_at_k"],
    "chunk_pass_at_k": detail["chunk_pass_at_k"],
    "answer_term_hit_at_k": detail.get("answer_term_hit_at_k"),
    "visual_answer_grounding": detail.get("visual_answer_grounding"),  # dict | None
}
```

`eval_runs.insert_result` already wraps `metrics` in `Jsonb(...)` — no change. Drop
the old scalar key `visual_answer_grounding_at_k` from the per-row `metrics` (no
back-compat shim, per project rules; the run-level float stays in
`eval_runs.summary`).

## Schema / fixture impact: none required

`eval/schemas/` governs **corpus shapes** (`talk`, `gold_example`,
`boundary_audit`, `model_candidates`) — curated inputs. It does *not* govern
`eval_results.metrics` or `system_output`; those are free-form JSONB validated
only by the DB column type and `db/repos/eval_runs.py`. No `valid_*/invalid_*`
fixture pair is needed because no JSON Schema conditional changes. The correct
guard is a **pytest case**: extend `eval/tests/test_run_visual_eval.py` (and the
measure-visual tests) to assert a forced `passed=true` row carries a non-empty
`evaluated_frame_ids`, a non-null `matched_term`, and `failure_reason=null`, and
that the three documented failure paths each set the right enum value. This is the
regression test §B's gap demands.

## VLM-judge compatibility (future research-matrix option)

The frame-side metric will eventually offer a second judge: instead of Tesseract
OCR + substring match, send each in-window frame PNG to a VLM and ask whether the
curator's claim is visually supported. The payload absorbs this without a shape
change:

- `judge_kind` flips to `"vlm"`. `evaluated_frame_ids` / `evaluated_image_paths`
  mean the same thing.
- `ocr_excerpts` is the OCR-path evidence field; the VLM path writes its analogue
  under `visual_judge_outputs: {str(frame_id): str}` (the model's verdict per
  frame). Exactly one of the two evidence maps is populated per row, keyed by
  `judge_kind`. Both are `{frame_id: str}` so §B's spot-check reads "the evidence
  for this frame" identically regardless of judge.
- `matched_term` stays meaningful for VLM if it returns which claimed term it
  grounded; otherwise `null` with `passed=true` and a populated
  `visual_judge_outputs` entry is acceptable — `matched_term` is OCR-substring
  specific by nature.
- `failure_reason` enum is judge-agnostic: `no_in_window_frame` (before either
  judge runs), the OCR/VLM "evaluated but no support" case
  (`ocr_no_terms_matched` for OCR; add a sibling `vlm_no_support` when the VLM
  path lands, not now), and `row_not_evaluable`.

Selected via the existing injectable `ocr_fn` seam: a future `judge_fn` (or a
`judge_kind` switch wrapping `ocr_fn`) chooses OCR vs VLM, and the dataclass is the
common output contract. The window gate, evidence-conjunction structure, and
not-evaluable handling are shared across both paths, so the payload — not the
judge — is the stable interface the report and §B audit depend on.

## Files that change when implemented

`eval/runners/measure_visual.py` (dataclass + new `frame_ocr_grounding_at_k`
helper + `_grounding_to_dict` + signature change + summary assembly),
`eval/runners/run_visual_eval.py` (`_write_per_example_results` metrics dict),
`eval/tests/test_run_visual_eval.py` and the measure-visual test module
(regression cases). No `eval/schemas/` or `eval/tests/fixtures/` change.
`db/repos/eval_runs.py` unchanged (already `Jsonb`-wraps `metrics`).
