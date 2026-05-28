# Agent A — Metric / Data Contract Design

> **2026-05-28 update (post-review):** The metric proposed below as `VisualAnswerHit@k` was renamed to `AnswerTermHit@k` during a review pass to avoid overclaim. The check runs on the retrieved chunk's TRANSCRIPT TEXT, not on any frame, OCR, or visual judge — so "visual" / "answer-grounded" in the name was misleading. The shipped code uses `answer_term_hit_at_k` (aggregator) and `answer_term_matches_at_k` (primitive); summary keys are `answer_term_hit_at_k`, `lift_answer_term_pp`, etc. The schema field stays `visual_evidence` — it describes provenance (the curator saw the term on the slide); the metric that consumes it is text-side. See `eval/reports/2026-05-28_visual_eval_v2/methodology.mdx` for the corrected framing.

**Original mission:** redesign the visual eval so lift becomes actionable. The 2026-05-27 run labelled `VisualLift@5 = -5.56pp` as NON-ACTIONABLE because the 4-channel text RRF passes 18/18 even on the `visual-required` slice. Root cause is a metric/intent mismatch: `VisualChunkTR@k` measures span overlap, not whether the surfaced chunk's text actually answers the question. When the answer lives only on the slide, the speaker still talks *around* the slide, so text channels grab a topically-relevant (but answer-empty) chunk and the overlap test passes.

This doc specifies the schema extension, the new metric, the primitives the team lead will implement, the actionability rule, and the test plan.

---

## Section 1 — Schema decision

### Recommendation: **Option X — extend `gold_example.schema.json`**

Add an optional `visual_evidence` array per row. Keep one corpus, one validator path, one loader. The new field is opt-in; existing 18 visual_gold rows + every text gold file continue to validate untouched.

#### Why not Option Y (new `visual_answer_gold.jsonl`)

Option Y would mean:

- A second `GOLD_FILES` entry → second loader → second JSONL with the same `verified: true` + visual modality constraints → a second `eval_results.example_id` namespace at risk of colliding with `visual_gold.jsonl`.
- The `visual-required` rows already live in `visual_gold.jsonl`. Splitting them out forces a duplicate `id` problem or a deletion + re-insertion, both of which churn provenance for zero gain.
- The phase-0 gate counter (`count_verified_gold_and_talks` in `eval/validate.py:75`) would have to learn yet another file, or silently skip a new committed corpus file — which would be a tripwire later.

Option X is the strictly smaller change. The conditional ("`visual_evidence` is only meaningful when modality intersects `VISUAL_MODALITIES`") sits naturally next to the existing `single_clip / synthesis / negative` conditionals, which already encode "this field is meaningful only for this question_type."

#### Concrete schema fragment

Append the property and a Python-side requirement (a JSON Schema `if/then` is fine here too; both are below).

```json
{
  "properties": {
    "visual_evidence": {
      "type": "array",
      "minItems": 1,
      "description": "Optional. Curator-supplied required-answer terms scoped to a specific visual element on the gold frame. When present, the answer-grounded visual metric (VisualAnswerHit@k) is computed for this row. Each entry asserts: at least one of `required_any` MUST appear, normalized, in any retrieved chunk's text for the row to pass.",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["visual_element", "required_any"],
        "properties": {
          "visual_element": {
            "type": "string",
            "minLength": 3,
            "description": "Short label for the visual surface the terms came from (e.g. 'plot y-axis', 'code block', 'slide title'). For audit only — never compared against retrieved text."
          },
          "required_any": {
            "type": "array",
            "minItems": 1,
            "items": { "type": "string", "minLength": 1 },
            "description": "Disjunction. The chunk passes this entry if ANY one of these normalized terms appears as a substring in the retrieved chunk's text. Terms should be the exact answer-bearing strings the curator confirmed on the frame (e.g. 'NDCG@10', 'is_resumable=True', '1/100x'). Curator-watched, not paraphrase-tolerant — see Section 2."
          },
          "normalize": {
            "type": "string",
            "enum": ["lowercase_collapse_ws", "exact"],
            "default": "lowercase_collapse_ws",
            "description": "How both the term and the candidate chunk text are normalized before substring compare. `lowercase_collapse_ws` lowercases and collapses runs of whitespace; `exact` is byte-for-byte. Code identifiers and exact UI strings use `exact`."
          }
        }
      }
    }
  },
  "allOf": [
    {
      "if": { "required": ["visual_evidence"] },
      "then": {
        "properties": {
          "modality": {
            "contains": {
              "enum": ["slide", "screen_code", "diagram", "whiteboard"]
            }
          }
        }
      }
    }
  ]
}
```

The `if/then` enforces "if `visual_evidence` is present, modality must include at least one visual tag." This is the schema-side guard. The Python side adds one more cross-field check (Section 3).

#### `GOLD_FILES` change

**No change.** `visual_gold.jsonl` is already registered at `eval/validate.py:37-48`. `test_gold` is untouched. The field is opt-in on existing rows in the dev-only `visual_gold.jsonl`.

#### How `verified: true` + visual-modality constraints carry over

- The existing `verified: true` rule at `eval/validate.py:288-292` continues to gate any row with or without `visual_evidence`. A row with `visual_evidence` is a verified row whose visual claims have ALSO been curator-confirmed on the frame; the verified flag is not split into two.
- The visual modality requirement at `eval/validate.py:297-306` already gates every `visual_gold.jsonl` row. The new schema `if/then` adds a second gate: any row carrying `visual_evidence` must declare a visual modality. Belt and suspenders — the validator catches a misclassified row before the metric reads it.

#### Fixture pair (sketched in full)

`eval/tests/fixtures/valid_visual_evidence.json`:

```json
{
  "id": "fixture_visual_evidence_ok",
  "question": "On the retrieval accuracy plot, what is the y-axis label?",
  "video_id": "abc123",
  "split": "dev",
  "question_type": "single_clip",
  "gold_spans": [{ "start_sec": 305, "end_sec": 410 }],
  "expected_claims": [
    "The y-axis label is Retrieval Quality (NDCG@10)."
  ],
  "visual_evidence": [
    {
      "visual_element": "plot y-axis label",
      "required_any": ["NDCG@10", "Retrieval Quality"],
      "normalize": "lowercase_collapse_ws"
    }
  ],
  "modality": ["slide", "diagram"],
  "difficulty": "easy",
  "verified": true,
  "curator": "fixture",
  "curated_at": "2026-05-27T10:00:00Z"
}
```

`eval/tests/fixtures/invalid_visual_evidence_without_visual_modality.json` (must fail the schema):

```json
{
  "id": "fixture_visual_evidence_no_visual_modality",
  "question": "On the retrieval accuracy plot, what is the y-axis label?",
  "video_id": "abc123",
  "split": "dev",
  "question_type": "single_clip",
  "gold_spans": [{ "start_sec": 305, "end_sec": 410 }],
  "expected_claims": [
    "The y-axis label is Retrieval Quality (NDCG@10)."
  ],
  "visual_evidence": [
    {
      "visual_element": "plot y-axis label",
      "required_any": ["NDCG@10"]
    }
  ],
  "modality": ["transcript"],
  "difficulty": "easy",
  "verified": true,
  "curator": "fixture",
  "curated_at": "2026-05-27T10:00:00Z"
}
```

`eval/tests/fixtures/invalid_visual_evidence_empty_required_any.json` (must fail the schema — catches a dead entry that would always pass):

```json
{
  "id": "fixture_visual_evidence_empty_required_any",
  "question": "On the retrieval accuracy plot, what is the y-axis label?",
  "video_id": "abc123",
  "split": "dev",
  "question_type": "single_clip",
  "gold_spans": [{ "start_sec": 305, "end_sec": 410 }],
  "expected_claims": [
    "The y-axis label is Retrieval Quality (NDCG@10)."
  ],
  "visual_evidence": [
    { "visual_element": "plot y-axis label", "required_any": [] }
  ],
  "modality": ["slide"],
  "difficulty": "easy",
  "verified": true,
  "curator": "fixture",
  "curated_at": "2026-05-27T10:00:00Z"
}
```

Three fixtures so the validator self-test asserts (a) the optional field is accepted, (b) the modality-coupling conditional fires, (c) the `minItems: 1` on `required_any` actually fires. Per `.claude/rules/patterns.md` ("Fixture Naming") — one fixture per rule, not per schema.

#### Dev-only placement

The new field is added to dev rows in `eval/corpora/ai_engineering_v0/visual_gold.jsonl` only. `test_gold.jsonl` is not touched. `visual_gold.jsonl` is already a dev-only file (`measure_visual.py` docstring §"Methodology guardrails" lines 36-44; the visual_gold path is hard-coded to the corpus dir and the runner does not consult `test_gold`).

---

## Section 2 — Metric definition

### Recommendation: **Approach 1 — substring/keyword match against retrieved chunk text**

Curator supplies a small, deterministic list of required-answer terms per visual element (already specified in Section 1's `visual_evidence.required_any`). The new metric `VisualAnswerHit@k` passes for a row iff there exists a top-k retrieved chunk whose normalized text contains at least one term from EACH `visual_evidence` entry. The metric is computed alongside the existing `VisualChunkTR@k` (does not replace it — see "Composition" below).

#### Why not Approach 2 (LLM judge per retrieved chunk)

Approach 2 is paraphrase-tolerant but expensive in three ways. **Cost:** 18 examples × top-k = 5 → 90 judge calls per visual eval run, but lift requires a second run on the same 18 examples through the 4-ch baseline → 180 calls; with a small cheap_extraction model at DeepInfra prices (Gemma 4 E4B not yet hosted there; falling back to qwen3-8b at roughly $0.07/$0.10 per Mtok with ~500-token prompts × ~50-token completions), the per-run dollar cost is small (cents) but the latency is non-trivial and adds a hosted dependency where today the eval is fully local. **Cross-family discipline:** the judge family must differ from the text-eval generator/judge families chosen in bakeoff #1 (ADR 004 §"judge.minimums.cross_family_required"). That constraint is fine but it pins another model before we want to. **Reproducibility:** judge calls drift even at temperature 0 across DeepInfra deploys; the methodology MDX would have to log per-row judge outputs (cost: more storage + a model_id pin we do not yet have). Approach 1 has none of these problems and the curator hours are bounded — see below.

#### Why not Approach 3 (curator-structured paraphrases)

Approach 3 adds curator hours per row to enumerate acceptable paraphrases. For the 10 `visual-required` rows the answer-bearing content is largely exact strings the speaker does *not* utter (e.g. `is_resumable=True`, `voyage-3-large`, `1/100x`, `RecoverableExceptions`). Paraphrase tolerance is the wrong knob — the whole point of `visual-required` is that the slide contains exact text the transcript lacks. Approach 1 leverages this directly: the curator picks 1–3 strings per visual element from the actual frame contents and we test substring containment in retrieved chunk text. Paraphrases are not the lever; precision of the term list is.

#### Justification — curator hours, runtime $, reproducibility, composition

| Dimension | Approach 1 (chosen) | Approach 2 | Approach 3 |
|---|---|---|---|
| Curator minutes per row | ~3 (already-watched frame → pick 1–3 exact terms; per `eval/curation/visual_gold_review/2026-05-27/*/visual_required_notes.md` the curator has the frame open) | ~0 (judge does the work) | ~10 (enumerate paraphrases) |
| Runtime $ per visual eval run | $0 (pure Python string ops) | ~$0.01 of judge calls + new hosted dep | $0 |
| Reproducibility | byte-identical across runs | depends on hosted model behavior + temperature | byte-identical |
| Cross-family judge concern | none (no LLM) | requires new judge family decision | none |
| Failure mode | term curator missed → false negative on a real hit | judge hallucinates pass on irrelevant chunk | paraphrase incomplete → false negative |
| Composition with existing `VisualChunkTR@k` | runs alongside; `chunk_tr` still measured for the legacy spans (Section 3) | same | same |

The dominant failure mode for Approach 1 (curator picks a term that the answer-bearing chunk happens to contain in a non-answer-bearing way) is mitigated by the `gold_span` overlap precondition (Section 3): the chunk must also overlap the gold span on the right video. So a topically-similar chunk from a *different* span on the same video cannot accidentally pass.

#### Composition — replace or alongside?

**Alongside.** Both metrics ship in `summary` and `eval_results.metrics`. Concrete rationale:

- `VisualChunkTR@k` is the legacy, span-overlap metric. It is what the n=8 text-saturated slice measures and what every prior eval reference compares against. Removing it would invalidate the cross-time comparison the methodology MDX makes between the n=8 and n=18 runs.
- `VisualAnswerHit@k` is the new, answer-grounded metric. It is the one the actionability gate (Section 4) is rewritten against.
- Both ride the same retrieval call (no extra DB roundtrip): we already pull `top_k_chunks` per example; the new metric just runs additional string ops on the text payload that the retrievers already returned.

`VisualLift@k` becomes two numbers: `lift_chunk_tr_pp` (legacy) and `lift_answer_hit_pp` (new). Headline lift in the methodology MDX uses `lift_answer_hit_pp`.

---

## Section 3 — Metric primitives

Signatures the team lead will implement in `eval/runners/measure_visual.py`. All functions are pure (no DB). The DB-bound aggregators reuse existing patterns.

```python
# --- new pure primitives -------------------------------------------------

@dataclass(frozen=True)
class VisualEvidenceItem:
    visual_element: str
    required_any: tuple[str, ...]
    normalize: str = "lowercase_collapse_ws"  # or "exact"


def normalize_text(text: str, mode: str) -> str:
    """`lowercase_collapse_ws` → lowercase + collapse runs of whitespace to a
    single space + strip. `exact` → return as-is. Unknown mode → raise ValueError
    (loud failure during loader resolves which mode the curator picked)."""


def chunk_text_contains_evidence(
    chunk_text: str,
    evidence: Sequence[VisualEvidenceItem],
) -> bool:
    """Return True iff for EVERY `VisualEvidenceItem` in `evidence`, at least one
    of its `required_any` terms (normalized per its `normalize` mode) appears as
    a substring in the normalized `chunk_text`. Empty `evidence` → True
    (vacuous, but callers must not invoke this for rows that lack visual_evidence;
    see `visual_answer_hit_at_k` for the gating check)."""


def answer_hit_at_k(
    results: Sequence[ChannelResult | FusedResult],
    gold: VisualGoldQuery,
    evidence: Sequence[VisualEvidenceItem],
    *,
    k: int,
) -> bool:
    """Return True iff there exists a chunk in the top-k that
       (a) overlaps the gold span on the matching video (same predicate as
           `chunk_passes_at_k`), AND
       (b) satisfies `chunk_text_contains_evidence(chunk.text, evidence)`.

    Combining (a) + (b) prevents the "topically-similar chunk on the wrong span
    accidentally contains the term" false positive. Empty `evidence` → False
    (caller must skip rows without visual_evidence rather than passing them
    vacuously). Empty `results` → False."""


# --- new DB-bound aggregator --------------------------------------------

def visual_answer_hit_at_k(
    conn: psycopg.Connection,
    examples: Sequence[VisualGoldQuery],
    evidence_by_example: dict[str, Sequence[VisualEvidenceItem]],
    *,
    k: int = 5,
    chunk_fn: ChunkFn | None = None,
) -> tuple[float, list[bool | None], list[list[dict]]]:
    """Aggregate VisualAnswerHit@k over the subset of `examples` that have
    visual_evidence.

    Returns (score, per_example_passes, per_example_top_k_chunks):
      * `score` = mean over the `n_evaluable` examples that carry evidence;
        rows without evidence DO NOT participate in either numerator or
        denominator (they contribute `None` in `per_example_passes`).
      * `score = 0.0` if `n_evaluable == 0` (caller surfaces as "skip
        answer-grounded metric").

    Rationale for the None convention: the metric is a strict superset of the
    text-saturated 8 rows that have no `visual_evidence` today. Counting a
    `None` as a fail would penalize ColQwen for legacy rows it was never
    measured against. Counting it as a pass would inflate the score. Excluding
    it from the denominator is the only honest aggregation."""
```

#### Loader change (`load_visual_gold`)

Extend `VisualGoldQuery` to optionally carry parsed `visual_evidence`, OR (cleaner) load evidence into a separate dict keyed by `example_id`. The runner passes it explicitly to `visual_answer_hit_at_k` — that keeps `VisualGoldQuery` lean and matches the existing pattern where `measure_embeddings` passes auxiliary state through the function call, not the gold record.

#### Edge cases

| Case | Behavior |
|---|---|
| No candidate chunk matches the gold span | `answer_hit_at_k → False` regardless of evidence (span gate fails before evidence check) |
| Empty retrieval list | `answer_hit_at_k → False` |
| Row has no `visual_evidence` | `visual_answer_hit_at_k` skips the row (contributes `None`, not 0/1) |
| Curator-provided term appears in a non-overlapping chunk | does NOT pass — the span overlap gate catches this |
| Curator-provided term appears in an overlapping chunk via paraphrase the curator did not list | false negative — known tradeoff of Approach 1, mitigated by reviewing per_example.jsonl after each run and adding terms to `required_any` |
| All `required_any` lists are empty | schema rejects at validation time (fixture `invalid_visual_evidence_empty_required_any.json`) |
| Multiple evidence entries (e.g. y-axis label + tick values) | conjunction across entries, disjunction within `required_any`. Reflects "the answer must mention BOTH the y-axis AND a tick value, but either NDCG@10 or Retrieval Quality counts as the y-axis term" |

#### Aggregate (`measure_visual`) signature change

```python
def measure_visual(
    conn: psycopg.Connection,
    examples: Sequence[VisualGoldQuery],
    evidence_by_example: dict[str, Sequence[VisualEvidenceItem]] | None = None,
    *,
    k: int = 5,
    tolerance_sec: float = FRAME_SAMPLE_EVERY_SEC,
    include_lift: bool = True,
) -> tuple[dict, dict[str, dict]]:
    ...
```

New keys in the returned `summary`:

```python
{
  # ... existing keys (visual_frame_recall_at_k, visual_chunk_tr_at_k, k, ...)
  "visual_answer_hit_at_k": float,        # mean over n_evaluable
  "visual_answer_hit_n_evaluable": int,   # how many rows had visual_evidence
  "visual_answer_hit_n_skipped": int,     # n_total - n_evaluable
  "visual_lift": {
    "with_visual_pass_rate": float,           # legacy (chunk_tr-based)
    "without_visual_pass_rate": float,        # legacy
    "lift_chunk_tr_pp": float,                # renamed from lift_pp
    "lift_answer_hit_pp": float,              # NEW — only when n_evaluable > 0
    "answer_hit_with_visual_rate": float,     # NEW
    "answer_hit_without_visual_rate": float,  # NEW
    "answer_hit_n_evaluable": int,
  },
  "per_example": {
    example_id: {
      "frame_pass": bool,
      "chunk_pass": bool,
      "answer_hit": bool | None,    # None if row has no visual_evidence
    }
  }
}
```

New keys in each `eval_results.metrics`:

```python
{
  "frame_pass_at_k": bool,
  "chunk_pass_at_k": bool,
  "answer_hit_at_k": bool | None,    # None for rows without visual_evidence
  "lift": {                          # only when include_lift
    "with_visual": bool,             # legacy (chunk_tr)
    "without_visual": bool,          # legacy
    "answer_hit_with_visual": bool | None,
    "answer_hit_without_visual": bool | None,
  }
}
```

Single-transaction write contract is preserved — `insert_run` + N `insert_result` calls remain inside the same `psycopg.connect(...)` block at `run_visual_eval.py:250-263`. The contract change is per-row-payload size only (extra ~80 bytes of JSON), not call shape.

---

## Section 4 — Methodology actionability rule

**Today's rule** (`measure_visual.py` module docstring + 2026-05-27 methodology MDX): *"Lift is actionable iff the without-visual baseline misses ≥1 example."*

**New rule, single sentence, quote verbatim in the methodology MDX:**

> `VisualLift_AnswerHit@k` is actionable iff the answer-grounded metric has `n_evaluable ≥ 5` AND the 4-channel text-only baseline fails the answer-grounded metric on ≥1 evaluable example; the legacy `VisualLift_ChunkTR@k` is reported alongside but is exploratory only and never alone justifies a candidate change.

The `n_evaluable ≥ 5` floor exists because lift on n=1 or n=2 evaluable rows is statistical noise; 5 is the smallest set where a single regression is ≤20% of the slice. The `≥1 baseline miss` clause is the analogue of today's rule — without it, the visual channel has no rescue opportunities and lift cannot be measured in the positive direction.

---

## Section 5 — Test plan

All tests land in `eval/tests/test_measure_visual.py` next to the existing primitives tests (FAST tests, no DB, no model load).

### Schema fixture pair contracts (`make validate-self-test`)

- `valid_visual_evidence.json` — validator must accept. Asserts the optional field is accepted with a minimal payload.
- `invalid_visual_evidence_without_visual_modality.json` — validator must reject. Asserts the `if/then` modality conditional fires.
- `invalid_visual_evidence_empty_required_any.json` — validator must reject. Asserts `minItems: 1` on `required_any` fires.

The existing 17-fixture set + these 3 new fixtures all run inside the existing self-test loop (`eval/validate.py:336-385`). No self-test code change required — the loop iterates `valid_*.json` / `invalid_*.json` by glob.

### Metric primitive unit tests

```python
def test_normalize_text_lowercase_collapse_ws():
    assert normalize_text("  Retrieval  Quality  (NDCG@10)  ", "lowercase_collapse_ws") \
        == "retrieval quality (ndcg@10)"

def test_normalize_text_exact_preserves_case_and_ws():
    assert normalize_text("is_resumable=True", "exact") == "is_resumable=True"

def test_chunk_text_contains_evidence_single_item_disjunction():
    evidence = [VisualEvidenceItem(
        visual_element="y-axis", required_any=("NDCG@10", "Retrieval Quality")
    )]
    assert chunk_text_contains_evidence("the y-axis is ndcg@10 from 66 to 84", evidence)
    assert chunk_text_contains_evidence("retrieval quality plot description", evidence)
    assert not chunk_text_contains_evidence("the speaker explains hybrid search", evidence)

def test_chunk_text_contains_evidence_multi_item_conjunction():
    evidence = [
        VisualEvidenceItem(visual_element="axis", required_any=("NDCG@10",)),
        VisualEvidenceItem(visual_element="tick", required_any=("$0.01",)),
    ]
    assert chunk_text_contains_evidence("ndcg@10 at $0.01 per mtok", evidence)
    assert not chunk_text_contains_evidence("ndcg@10 at one cent per mtok", evidence)

def test_chunk_text_contains_evidence_exact_normalize_is_case_sensitive():
    evidence = [VisualEvidenceItem(
        visual_element="code", required_any=("is_resumable=True",), normalize="exact",
    )]
    assert chunk_text_contains_evidence("set is_resumable=True", evidence)
    assert not chunk_text_contains_evidence("set is_resumable=true", evidence)

def test_answer_hit_at_k_requires_span_overlap_AND_terms():
    # Topically-similar chunk on the WRONG span: term present but span fails → False
    g = _gold(start=170, end=180, video="v1")
    evidence = [VisualEvidenceItem(visual_element="y", required_any=("ndcg@10",))]
    chunks = [_chunk(start=400, end=420, text="ndcg@10 discussion")]
    assert not answer_hit_at_k(chunks, g, evidence, k=5)

def test_answer_hit_at_k_passes_when_both_gates_pass():
    g = _gold(start=170, end=180, video="v1")
    evidence = [VisualEvidenceItem(visual_element="y", required_any=("ndcg@10",))]
    chunks = [_chunk(start=160, end=185, text="speaker says ndcg@10")]
    assert answer_hit_at_k(chunks, g, evidence, k=5)
```

### The edge case that distinguishes topical from answer-bearing

This is the critical test — it asserts the new metric does what `VisualChunkTR@k` failed to do for `visual-required-google-adk-resumability-config-code` on the 2026-05-27 run:

```python
def test_answer_hit_distinguishes_topical_chunk_from_answer_bearing_chunk():
    """Regression test for the 2026-05-27 visual eval bug: chunk overlapped the gold
    span and was topically about resumability, but did NOT contain the answer-bearing
    code `is_resumable=True`. chunk_tr passes; answer_hit must fail."""
    g = _gold(start=2010, end=2030, video="nXafozNIk3c")
    evidence = [VisualEvidenceItem(
        visual_element="code block",
        required_any=("is_resumable=True", "ResumabilityConfig"),
        normalize="exact",
    )]
    topical_chunk = _chunk(
        start=2015, end=2025,
        text="Shubam explains that resume tracks tools that already ran and continues",
    )
    assert chunk_passes_at_k([topical_chunk], g, k=5)            # legacy passes
    assert not answer_hit_at_k([topical_chunk], g, evidence, k=5)  # new metric fails

    answer_bearing_chunk = _chunk(
        start=2015, end=2025,
        text="the code wires resumability_config=ResumabilityConfig(is_resumable=True)",
    )
    assert chunk_passes_at_k([answer_bearing_chunk], g, k=5)     # legacy still passes
    assert answer_hit_at_k([answer_bearing_chunk], g, evidence, k=5)  # new metric passes
```

This test is the one-line proof that the new metric measures what the methodology MDX says it should measure.

### Aggregator (`visual_answer_hit_at_k`) tests

- empty examples → `(0.0, [], [])`
- all examples lack evidence → `(0.0, [None, ...], top_ks)` with `score=0.0` reported alongside `n_evaluable=0`
- mixed: 3 with evidence (2 pass), 2 without → `score = 2/3 ≈ 0.667`, `n_evaluable=3`, `n_skipped=2`

---

## Section 6 — Open questions

(omitted — no genuine user-scope tradeoffs; recommendation chosen on cost + reproducibility grounds; cross-family discipline N/A for substring approach)
