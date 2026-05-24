"""Caption quality probe — decides whether ASR fallback is required.

A "passing" transcript has enough spoken content per minute and is not
dominated by [Music] / [Inaudible] / [Applause] / similar non-speech
markers. A "failing" transcript triggers the asr step.

Heuristic, not ML: this lives here so we can tune thresholds against the
actual corpus without changing the pipeline shape. The metric details
land on QualityResult.details for logging.

Thresholds (current):
  passes = words_per_minute >= 60 AND non_speech_line_ratio < 0.20
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

WPM_MIN = 60.0
NON_SPEECH_RATIO_MAX = 0.20


@dataclass(frozen=True)
class QualityResult:
    """passes=True means the transcript is good enough; ASR is skipped."""

    passes: bool
    details: dict[str, Any]


def probe(transcript_path: Path, duration_sec: float) -> QualityResult:
    """Score a transcript on disk against `duration_sec`."""
    text = transcript_path.read_text(encoding="utf-8", errors="replace")
    word_count = len(_WORD_RE.findall(text))
    minutes = max(duration_sec / 60.0, 1e-9)
    wpm = word_count / minutes

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    non_speech = sum(1 for ln in lines if _NON_SPEECH_MARKERS.search(ln))
    non_speech_ratio = non_speech / max(len(lines), 1)

    passes = wpm >= WPM_MIN and non_speech_ratio < NON_SPEECH_RATIO_MAX
    return QualityResult(
        passes=passes,
        details={
            "word_count": word_count,
            "duration_sec": duration_sec,
            "wpm": wpm,
            "non_speech_lines": non_speech,
            "non_speech_ratio": non_speech_ratio,
        },
    )
