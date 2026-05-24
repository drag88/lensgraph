"""Caption quality probe — decides whether ASR fallback is required.

A "passing" transcript has enough spoken content per minute and is not
dominated by [Music] / [Inaudible] / [Applause] / similar non-speech
markers. A "failing" transcript triggers the asr step.

Heuristic, not ML: this lives here so we can tune thresholds against the
actual corpus without changing the pipeline shape. The metric details
land on QualityResult.details for logging.

Thresholds (current):
  passes = words_per_minute >= 60 AND non_speech_line_ratio < 0.20

Counting is scoped to VTT *cue text* only — header lines (WEBVTT, Kind:,
Language:, NOTE), timestamp lines (containing `-->`), cue-index lines
(pure integers), and inline tag markup (`<00:00:15.519><c>...</c>`) are
stripped before metrics are computed. Without this, an empty or
timestamp-only VTT can falsely clear the threshold by counting timestamp
digits as "words" and dilute the non-speech ratio with headers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_NON_SPEECH_MARKERS = re.compile(
    r"\[(music|inaudible|applause|laughter|silence|crosstalk)\]", re.IGNORECASE
)
_WORD_RE = re.compile(r"\b[\w']+\b")
_INLINE_TAG_RE = re.compile(r"<[^>]*>")
_CUE_INDEX_RE = re.compile(r"^\d+$")
_HEADER_PREFIXES = ("WEBVTT", "Kind:", "Language:", "NOTE")

WPM_MIN = 60.0
NON_SPEECH_RATIO_MAX = 0.20


@dataclass(frozen=True)
class QualityResult:
    """passes=True means the transcript is good enough; ASR is skipped."""

    passes: bool
    details: dict[str, Any]


def _extract_cue_lines(text: str) -> list[str]:
    """Return cue text lines with inline VTT tags stripped; drop empties."""
    cue_lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(_HEADER_PREFIXES):
            continue
        if "-->" in line:
            continue
        if _CUE_INDEX_RE.match(line):
            continue
        stripped = _INLINE_TAG_RE.sub("", line).strip()
        if not stripped:
            continue
        cue_lines.append(stripped)
    return cue_lines


def probe(transcript_path: Path, duration_sec: float) -> QualityResult:
    """Score a transcript on disk against `duration_sec`."""
    text = transcript_path.read_text(encoding="utf-8", errors="replace")
    cue_lines = _extract_cue_lines(text)

    word_count = sum(len(_WORD_RE.findall(ln)) for ln in cue_lines)
    minutes = max(duration_sec / 60.0, 1e-9)
    wpm = word_count / minutes

    non_speech = sum(1 for ln in cue_lines if _NON_SPEECH_MARKERS.search(ln))
    non_speech_ratio = non_speech / max(len(cue_lines), 1)

    passes = wpm >= WPM_MIN and non_speech_ratio < NON_SPEECH_RATIO_MAX
    return QualityResult(
        passes=passes,
        details={
            "word_count": word_count,
            "duration_sec": duration_sec,
            "wpm": wpm,
            "non_speech_lines": non_speech,
            "non_speech_ratio": non_speech_ratio,
            "cue_line_count": len(cue_lines),
        },
    )
