# ColQwen processor band check

Date: 2026-05-28
Scope: ADR 005 follow-up flagged in prompt47. **Band print only** — full
ablation sweep (cadence x resolution x crop x prefilter_k) deferred to a
future session per prompt47 scope.

Script: `eval/reports/2026-05-28_visual_diagnostics/run_colqwen_band_check.py`
Run from repo root with `PYTHONPATH=. uv run python <script>`.

## Model

From `eval/config/model_candidates.yaml`
(`candidates.visual_retrieval.options[id=colqwen2.5]`):

- `provider`: `local`
- `provider_model_id`: `vidore/colqwen2.5-v0.2`

`embed/colqwen._model()` loads this id via
`ColQwen2_5_Processor.from_pretrained(model_id)` on top of the
`colpali-engine` 0.3.16 API (probed 2026-05-25).

## Processor band (printed)

```
processor.image_processor.min_pixels = None
processor.image_processor.max_pixels = None
processor.image_processor.size       = SizeDict(
    height=None, width=None,
    longest_edge=602112, shortest_edge=3136,
    max_height=None, max_width=None,
)
```

The `min_pixels` / `max_pixels` attributes on this build are `None`. The
band lives on `size` instead, expressed as total-pixel bounds:

- `shortest_edge` = 3,136 px  -> effective `min_pixels`
- `longest_edge`  = 602,112 px -> effective `max_pixels`

So the effective pixel band for an input image is

> **3,136 px <= W * H <= 602,112 px**

Images below the lower bound are upsampled, images above the upper bound
are downsampled (while preserving aspect ratio).

## Sampled frames (W_CYk2ogcDI)

Frames live at `frames/W_CYk2ogcDI/frame_NNNNNN.png` (cwd-relative).
113 frames total; sampled at indices 1, 25, 50, 75, 100:

| File | W x H | W * H |
|---|---|---|
| frame_000001.png | 640 x 360 | 230,400 px |
| frame_000025.png | 640 x 360 | 230,400 px |
| frame_000050.png | 640 x 360 | 230,400 px |
| frame_000075.png | 640 x 360 | 230,400 px |
| frame_000100.png | 640 x 360 | 230,400 px |

All five frames are 640x360 (16:9, ~0.23 MP). Stable across the sample —
no per-frame resolution drift.

## Verdict

**Inside band.** 230,400 px sits between 3,136 (min) and 602,112 (max),
so the ColQwen processor does **no resize** on these frames.

- 230,400 / 3,136   = ~73x above the lower bound (well clear of upsampling).
- 230,400 / 602,112 = ~38% of the upper bound (well clear of downsampling).

The frames are also notably **lower-resolution than the prompt assumed**.
The team-lead message described them as "720p (1280x720, 921,600 px)",
but the on-disk frames are 640x360 (230,400 px). 1280x720 would also have
sat inside the band (921,600 px > 602,112 px is *above* the upper bound,
so 720p would be **downsampled** to fit under 602,112 px). The actual
640x360 frames need no resize.

## Implication for the visual diagnostics report

The "frames too low-res for ColQwen" hypothesis is **not supported** at
the processor-band level for this specific corpus. The processor accepts
230,400-pixel images as-is; ColQwen is operating on the frames the
ingester wrote, untouched, not on a forcibly-downsampled version.

That does **not** rule out the broader hypothesis that **640x360 is too
low for slide legibility** — fine text on slides may genuinely be
unreadable at 360p regardless of whether ColQwen resizes the input.
This question is what the deferred resolution-ablation sweep is for. The
band check just confirms the failure mode is not "processor secretly
shrinks 720p frames below useful resolution"; the input pipeline is
delivering the captured resolution end-to-end.

A second observation worth flagging to team-lead: the ingester is
currently writing **360p**, not 720p. If the intent in ADR 005 / phase-1
design was 720p, this is a real drift worth checking (`ingest/frames.py`
defaults, ffmpeg invocation, or the source video's available streams).

## Deferred (explicit)

Full ablation sweep — cadence (1s / 2s / 5s), resolution (360p / 720p /
1080p), crop (none / center / slide-detect), prefilter_k (10 / 30 / 100)
— is **deferred to a future session** per prompt47 scope. This report
only prints and classifies the processor band against current on-disk
frames; it does not re-encode, does not touch the DB, and does not
amend ADR 005.

## Files

- `eval/reports/2026-05-28_visual_diagnostics/run_colqwen_band_check.py` -
  the print-only script.
- `eval/reports/2026-05-28_visual_diagnostics/colqwen_band_check.md` -
  this report.
