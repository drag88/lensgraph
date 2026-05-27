# ADR 005 — Chapter-slice video capture and frame-sampling resolution

**Status:** Accepted (2026-05-27). Verified end-to-end on `aie_sg_2026_d2_arize_alyx` (chapter
`[516, 1494]` of source `m12vGjfbNlo`): yt-dlp produced a 39 MB file (vs projected 5–15 GB
full-stream), all 98 chapter frames sampled cleanly, and the resulting ColQwen patches drove the
first real `visual_eval` run at commit `ec7b5b7`. Reproducibility tooling
(`scripts/fetch_video.py`, `make fetch-video` / `make fetch-videos`) landed alongside this status
flip.

**Relates to:** ADR 003 (conference-talks corpus), the `frames_handler` / `frames.sample`
contract, and the chapter-slice schema fields on `talks.yaml`
(`source_video_id`, `source_start_sec`, `source_end_sec`).

## Context

The launch corpus includes "chapter-sliced" talks — 10–25 minute windows inside
multi-hour conference livestreams. Today's only live instance is
`aie_sg_2026_d2_arize_alyx`, which is chapter `[516, 1494]` of source video
`m12vGjfbNlo` (a ~9 hr AI Engineer Singapore Day 2 livestream).

The current ingest contract:

- `ingest/handlers.py::frames_handler` calls `frames_mod.sample(...)` with
  `start_sec=talk.source_start_sec`, `duration_sec=source_end_sec - source_start_sec`
  when `source_video_id` is set, else `(0.0, talk.duration_sec)`.
- `ingest/frames.py::sample` shells out to ffmpeg using `-ss <start>` (input
  seek) then `-t <duration>` and writes one PNG per `every_sec=10` with
  `frame_sec` labelled `0, 10, 20, ...` relative to the **talk's** zero.
- `frames.default_video_path_for_talk` resolves the physical file to
  `videos/<corpus>/<source_video_id or video_id>.mp4` — i.e. one file per
  *source*, not per chapter, for sliced talks.
- `videos/` is gitignored. Frame PNGs land under `frames/<video_id>/` and are
  also gitignored.

The naive yt-dlp full download of the parent livestream is the wrong default:
the file in flight when this ADR was opened was already at 617 MB and
growing — a single 9 hr stream is 5–15 GB at 720p, of which we only need
~16 minutes. That blows the disk budget and the time budget for a corpus that
will plausibly contain a dozen such chapter-sliced entries by v1.

A first mitigation already in flight is
`yt-dlp --download-sections '*0-1500'` which trims the file at download time
but keeps the source's t=0 intact, so `frames_handler`'s `start=516` still
lines up. This ADR ratifies that approach (with one refinement) and locks in
the rest of the surrounding policy.

## Options

### A. Keep current contract; full-stream download

Operator runs `yt-dlp <url>` for the parent video, gets the full 5–15 GB
file, drops it at `videos/<corpus>/<source_video_id>.mp4`, `frames_handler`
seeks `[516, 1494]` with ffmpeg as today.

**Pros.** Zero code change. Works for any chapter range, retroactively, no
manifest. Source-video resolution is whatever YouTube serves.

**Cons.** 5–15 GB per source video, of which ~96% is wasted. For a corpus
that will plausibly hold 10–20 chapter slices spread across 5–8 source
livestreams, the on-disk footprint is 50–120 GB — punishing on a single
laptop and meaningless storage cost vs the chapter content. yt-dlp wall
time is dominated by the unused tail.

### B. `yt-dlp --download-sections '*0-END'` (head-trim only)

`END = max(source_end_sec) + small_safety_margin` across every chapter that
references this source. yt-dlp downloads `[0, END]` of the parent. The file's
internal time origin is unchanged. `frames_handler` still seeks
`talk.source_start_sec` directly. No code, schema, or handler change.

**Pros.** Zero code change to land. Preserves time origin so the entire
existing chapter-slice path keeps working. Multiple chapters from the same
source share one file (union-of-chapters semantics fall out for free as long
as `END = max(source_end_sec)`). yt-dlp respects the section flag at the
HTTP-range level for DASH/HLS streams when format permits, so the wasted
bytes are bounded by `min(source_end_sec) - 0`, not the full stream tail.

**Cons.** Bytes from `0` to `min(source_start_sec)` are wasted — for Arize at
`[516, 1494]`, that is 516 s of unused video at the head. Acceptable for
~10 min head-waste; ugly if a future chapter sits at `start_sec = 25000` in
an 9 hr stream.

### C. `yt-dlp --download-sections '*START-END'` + per-talk file-origin override

Download only `[source_start_sec, source_end_sec]` of the parent. File's t=0
is now `source_start_sec`. To keep `frames_handler` correct, add a new
`talks.yaml` field (or compute one in code) — e.g. `file_origin_sec` — and
have `frames_handler` pass
`start_sec = talk.source_start_sec - file_origin_sec` to `frames.sample`.

**Pros.** Tightest disk footprint per chapter. Aligns capture cost with
chapter cost exactly.

**Cons.** New schema field, new validator rule, breaks the invariant that
"one source video → one file → multiple chapters." Two chapters from the
same source would need either two trimmed files (duplicating the overlap) or
re-introducing the union math at download time (in which case we are back to
option B). Also: yt-dlp `--download-sections` with a non-zero start uses
keyframe-aware cuts; the file's reported start_pts may not be exactly the
requested second, and ffmpeg's `-ss` against a re-muxed segment can drift
by up to one GOP (~2–5 s). Time-origin alignment becomes operationally
fragile for a small disk-space gain.

### D. yt-dlp full + post-process ffmpeg trim

`yt-dlp <url>` → full 5–15 GB → `ffmpeg -ss START -to END -c copy` →
trimmed file. Delete the full file.

**Pros.** Final on-disk artefact is tight. Maximum control over the trim.

**Cons.** Pays full download cost AND full trim cost. Slowest path. No
benefit over A on download time; worse than B on disk while the full file
exists. Only justified if upstream rejects `--download-sections`.

### E. Stream-only, no on-disk video — sample directly from URL

`ffmpeg -ss START -i <youtube_direct_url> -t DURATION ...`, no intermediate
mp4.

**Pros.** Zero local mp4 storage.

**Cons.** YouTube direct URLs are signed and short-lived (≤6 hr); not
reproducible. ffmpeg-over-HTTPS for a 16 min window against a 9 hr DASH
stream requires multiple range requests and is brittle. Breaks the
"`videos/` is the durable local cache" contract that the rest of the
pipeline (re-runs, ColQwen re-encoding) depends on.

## Decision

**Adopt Option B** — `yt-dlp --download-sections '*0-END'` where
`END = max(source_end_sec) + 30 s` across all chapters referencing the same
source, with the trimmed file written to the existing
`videos/<corpus>/<source_video_id>.mp4` path. No schema change. No code
change to `frames_handler` or `frames.sample`. The 30 s tail-pad protects
against keyframe-aligned cuts truncating the last sampled frame.

For the Arize chapter the in-flight download used `*0-1500`. **Reconciliation —
no re-fetch needed for this specific chapter.** The 1500 s file covered
`[516, 1494]` cleanly: at `every_sec = 10` the last sampled frame within the
chapter lands at `source_start_sec + 970 s = 1486 s` (the next would be at
1496 s, outside the chapter). With the file extending to 1500 s, ffmpeg seeks
to 516 and reads through 1494 without crossing the trimmed-file boundary, and
all 98 expected frames are present (verified: `SELECT max(frame_sec) FROM frames
WHERE video_id='aie_sg_2026_d2_arize_alyx'` returns 970 — i.e. the full
[0, 970] talk-relative grid is populated).

**The formal pattern remains `*0-<max(source_end_sec) + 30>` (= `*0-1524`
for Arize).** The 30 s pad is defence-in-depth against keyframe-aligned cuts
truncating the last few seconds of the segment. The 1500 s download succeeded
in this case because the every_sec=10 cadence happened to put the last frame
8 s before the chapter end; a chapter whose end_sec aligned closer to a
multiple of `every_sec` could fall inside the truncation zone with only a 6 s
pad. `scripts/fetch_video.py` always emits the `+30 s` form, so future
operators don't have to think about this.

**Frame-sample resolution: download at 720p (yt-dlp format
`bv*[height<=720]+ba/b[height<=720]`).** Rationale below; this is the one
recommendation I am holding loosely until ColQwen2.5's processor config is
verified.

**Storage convention:** keep the existing one-file-per-source contract
(`videos/<corpus>/<source_video_id>.mp4`). Multiple chapters from the same
source livestream share the file; `END` is the union upper bound.

**Reproducibility:** add `scripts/fetch_video.py` that reads `talks.yaml`,
computes the correct yt-dlp command per source (full-stream for non-sliced
talks, `--download-sections '*0-END'` for sliced talks with `END = max(
source_end_sec) + 30`), and runs it. Plus a `make fetch-video VIDEO_ID=...`
target that resolves source_video_id and shells out. No new manifest file —
the YAML already has every fact required, and a generated manifest would
drift.

**Backwards compatibility:** Option B is a strict superset of the current
non-chapter path. For talks without `source_video_id`, the fetch script
falls through to a plain `yt-dlp <url>` and writes to
`videos/<corpus>/<video_id>.mp4`. `nXafozNIk3c` and `W_CYk2ogcDI` keep their
cheap whole-file path with no special case.

## Why this over the alternatives

Option B is the unique choice that preserves the **time-origin invariant**
that `frames_handler` already depends on, **without** adding a new schema
field or breaking the one-file-per-source convention. The disk-space win
from Option C (~5–10% smaller per chapter, on a 16 min slice) is not worth
the keyframe-alignment fragility or the chapter-sharing complication. Option
D pays for the full download anyway. Option E breaks reproducibility.

A second-order point: the ADR-004 selection rule for visual retrieval pins
ColQwen2.5 as the visual encoder. ColQwen down-samples every input image to
its processor's internal patch grid, so the *source* video resolution is the
upper bound on quality, not the final answer. Optimising the on-disk video
for *ffmpeg seek precision* matters more than for *pixel count* — and
ffmpeg seek precision is what Option B preserves.

## Frame-sample resolution — the load-bearing detail

ColQwen2.5's image processor down-samples inputs to a fixed patch grid
before encoding. The colpali-engine `ColQwen2_5_Processor` is built on
Qwen2.5-VL's vision tower, which uses a dynamic resolution strategy (28×28
patches, image side scaled to fall in a `min_pixels` / `max_pixels` window
that defaults to roughly `[256*28*28, 1280*28*28]` — i.e. images larger than
~1280 patches are down-scaled, smaller than ~256 are up-scaled). **I have
not verified the exact min/max pixels values that the colpali-engine 0.3.16
processor sets** — this should be confirmed against
`ColQwen2_5_Processor.from_pretrained(model_id).image_processor.{min_pixels,
max_pixels}` before locking in a number for the record. The headline
implication is robust: feeding ColQwen a 4K slide image is wasted bytes;
feeding it a 240p slide loses small-font legibility.

Working assumption pending verification: **720p (1280×720) is the sweet
spot.** It is comfortably above the down-sample threshold for most slide
aspect ratios so font legibility is preserved through to the vision tower,
roughly 4× smaller on disk than 1080p (a 16 min 720p H.264 video is
~150–250 MB vs ~600 MB–1 GB at 1080p), and 720p is the lowest tier
YouTube serves where projected-screen slide text (12–18 pt at 1920×1080
source) stays readable. 480p would shave another 2× but starts to mangle
small code-on-slide samples — and ColQwen's whole job here is reading the
slide text.

**Action item: before the first ColQwen run on a real chapter slice,
print `processor.image_processor.min_pixels` and
`processor.image_processor.max_pixels` and confirm that a 720p PNG falls
inside the band without being down-sampled by more than 2×.** If the
processor caps at, say, 1024 patches and a 720p PNG yields 2400+, drop
download to `bv*[height<=540]`. This is a quick check; do not lock the
default until it runs.

## Storage convention — one file per source

For a source video `S` with chapters `[c1, c2, ..., cn]`, store one
`videos/<corpus>/<S>.mp4` covering `[0, max(end_sec) + 30]`. Adding a new
chapter that extends past the current `max(end_sec)` requires re-fetching
with a larger `END`. The fetch script must detect this and either re-download
the file or warn loudly.

Rationale for one-per-source vs one-per-chapter:

1. Matches the existing `default_video_path_for_talk` contract — no handler
   change.
2. Disk amortisation: two chapters from the same source share bytes
   `[0, min(start_sec)]` and `[min(start_sec), max(end_sec)]`. One file
   stores it once; one-per-chapter would duplicate or require complex
   overlap math.
3. Operationally simpler: an operator inspecting `videos/` sees one file per
   source URL they pulled, mirroring what they would see if they had
   downloaded full-stream.

The downside — wasted head bytes `[0, min(start_sec)]` — is the same
downside Option B accepts in general and is bounded by the smallest chapter
start in the corpus.

## Reproducibility

Two pieces:

1. **`scripts/fetch_video.py`** — single-purpose CLI. Args: `--video-id <id>`
   (talk video_id, sliced or not) or `--corpus <name>` (every video in the
   corpus). Reads `eval/corpora/<corpus>/talks.yaml`. For each video:

   - Non-sliced: `yt-dlp -f 'bv*[height<=720]+ba/b[height<=720]' -o
     videos/<corpus>/<video_id>.mp4 <url>`.
   - Sliced (one entry, or a group sharing `source_video_id`): compute
     `END = max(source_end_sec) + 30`, run
     `yt-dlp --download-sections '*0-<END>' -f
     'bv*[height<=720]+ba/b[height<=720]' -o
     videos/<corpus>/<source_video_id>.mp4 <source_url>`.
     `<source_url>` is the `url` of any chapter in the group, with the
     `&t=...` fragment stripped.

   Skips files that already exist on disk unless `--force`. Prints the exact
   command it ran for each video (so an operator can copy-paste it
   manually if the script breaks).

2. **`make fetch-video VIDEO_ID=...` and `make fetch-videos CORPUS=...`** —
   thin wrappers around `scripts/fetch_video.py`. Mirror the existing
   `make ingest VIDEO_ID=...` / `make ingest-all CORPUS=...` ergonomics.

3. **Playbook update** — `eval/curation/playbook.md` Per-talk workflow §1
   already documents the transcript-only yt-dlp command. Add a sibling step
   for the video download, pointing at `make fetch-video VIDEO_ID=...`. Do
   NOT duplicate the yt-dlp incantation in the playbook — the script is the
   source of truth.

This keeps the reproducibility surface in version control (the script + the
talks.yaml + the playbook reference), with no parallel manifest file that
would drift.

## Backwards compatibility

- `frames_handler`, `frames.sample`, `default_video_path_for_talk` — **no
  change**. The existing chapter-slice path (seek `[source_start_sec,
  source_end_sec]` against `videos/<corpus>/<source_video_id>.mp4`) is
  already correct under Option B.
- `talks.yaml` — **no schema change**. No new fields. `aie_sg_2026_d2_arize_
  alyx` works unmodified.
- Validator — **no change**. The cross-field check that
  `duration_sec ≈ source_end_sec - source_start_sec` is unrelated to the
  download.
- Non-sliced talks (`nXafozNIk3c`, `W_CYk2ogcDI`) — fetch script falls
  through to plain `yt-dlp`, lands at `videos/<corpus>/<video_id>.mp4`,
  unchanged.

## Consequences

**Good.**

- Disk footprint per chapter source drops from 5–15 GB to ~200–400 MB at
  720p. For a v1 corpus of ~10 sliced talks across ~6 source streams,
  total `videos/` size is ~1.5–3 GB instead of 30–90 GB.
- No code, schema, or handler change to land the default; the existing
  contract was already correct.
- A future operator runs `make fetch-video VIDEO_ID=...` and gets a
  guaranteed-correct download — no per-talk yt-dlp incantation to memorise.
- The ColQwen visual channel sees images at a known resolution band,
  upper-bounded by 720p — reproducible behaviour across machines.

**Bad / accepted.**

- Bytes `[0, min(source_start_sec)]` are wasted on every shared source.
  Bounded by chapter selection discipline (don't pick chapters at second
  25000 of a 9 hr stream).
- 720p is a guess pending colpali-engine processor verification. The wrong
  guess wastes either bytes (too high) or recall (too low).
- Adding a chapter to an already-fetched source requires re-fetching if the
  new chapter extends past current `END`. The fetch script must surface
  this; an operator should not silently end up with a truncated file.

**Neutral.**

- yt-dlp `--download-sections` re-muxes the trimmed segment; the output is a
  valid .mp4 with a leading keyframe, so ffmpeg `-ss` against it behaves
  exactly as against a full file. Verified empirically against the in-flight
  Arize download; document the verification in the slice that lands the
  fetch script.

## Implementation slice — landed

The minimum-diff slice from v1 of this ADR landed alongside the Accepted
status flip:

- **`scripts/fetch_video.py`** *(landed)* — parses `talks.yaml`, groups
  sliced talks by `source_video_id`, builds one yt-dlp command per
  physical file (full-stream for non-sliced talks; `--download-sections
  '*0-<max(source_end_sec)+30>'` for sliced groups). Strips chapter-start
  `?t=` hints from the URL while preserving `v=<id>`. Idempotent: skips
  existing files unless `--force`. Prints the exact command per video
  so an operator can copy-paste manually if the script breaks.
- **`Makefile`** *(landed)* — `make fetch-video VIDEO_ID=...` and
  `make fetch-videos` (mirrors `make ingest` / `make ingest-all`
  ergonomics).
- **`eval/tests/test_fetch_video.py`** *(landed)* — 9 fast unit tests
  covering: full-video plain yt-dlp path, chapter group union-end +
  safety pad, lone-chapter padding, URL query stripping, corpus-all
  one-per-source aggregation, file-exists skip + `--force` override,
  unknown video_id raise, and a `yt-dlp` availability check (so the
  test suite fails loudly when the binary is not on PATH).

**Still owed — not part of this slice.** Recorded here so the next
operator knows they exist:

- **`eval/curation/playbook.md`** — append a "Pull the source video"
  step pointing at `make fetch-video`. Not yet added.
- **`docs/architecture.md`** — one-line addition under the Ingestion
  data flow box. Not yet added.
- **ColQwen2.5 processor band verification.** `ColQwen2_5_Processor.
  from_pretrained(...).image_processor.{min_pixels, max_pixels}` against
  the 720p PNG dimensions. The first real visual_eval run at
  `ec7b5b7` showed standalone visual recall at 1/8, which is below
  expectation; verifying the processor band is the cheapest next
  investigation. If 720p PNGs are being aggressively downsampled,
  switch the `YT_DLP_FORMAT` selector in `scripts/fetch_video.py` to
  the next-higher tier (1080p) or document the actual sweet spot.

No changes to `ingest/handlers.py`, `ingest/frames.py`,
`db/repos/talks.py`, `eval/schemas/talk.schema.json`,
`eval/validate.py`, or `eval/corpora/ai_engineering_v0/talks.yaml`.

## Revisit when

- A chapter selection lands with `source_start_sec` so deep into a source
  (e.g. > 1 hr) that the head-trim waste exceeds the chapter size itself.
  Re-evaluate Option C with a `file_origin_sec` field.
- colpali-engine ships a major processor change that alters the
  `min_pixels` / `max_pixels` band by more than 2×. Re-check the 720p
  decision.
- A second sliced source enters the corpus whose top-quality stream is only
  available at 480p or below; the default selector
  `bv*[height<=720]+ba/b[height<=720]` already degrades gracefully but
  document the degradation when it happens.
- We move ColQwen2.5 inference to Modal at higher input resolution
  (`max_pixels` raised); revisit whether 720p capture leaves quality on the
  table.

## Changelog

**2026-05-27 (v2):** Status flipped Proposed → Accepted after end-to-end
verification on Arize. Reconciled `*0-1500` vs `*0-1524` (1500 was OK for
this chapter's cadence; the +30 s pattern remains the documented norm and
is what `scripts/fetch_video.py` emits). Implementation slice landed:
`scripts/fetch_video.py`, `make fetch-video` / `make fetch-videos`, fast
test coverage. Playbook + architecture-doc updates and the ColQwen
processor-band verification are still owed and listed explicitly.

**2026-05-27 (v1):** Initial proposal. Locks in
`yt-dlp --download-sections '*0-<max(source_end_sec)+30>'` for chapter-sliced
sources, 720p capture by default, one-file-per-source storage,
`scripts/fetch_video.py` for reproducibility. No code or schema change to
land the default; implementation slice is reproducibility tooling only.
