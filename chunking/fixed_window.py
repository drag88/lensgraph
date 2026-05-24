"""Fixed-window chunker: the v0 baseline strategy.

Slides a `window_sec` window across the transcript with `overlap_sec`
overlap between consecutive windows. Cues are assigned to a window when
their midpoint falls inside that window. Sentence boundaries are
preferred at the trailing edge when the natural cut would land
mid-sentence. Token count is enforced via word truncation against
`max_tokens`.

The VTT parser is private to this module — `ingest/quality.py` has a
similar helper but drops timestamps (which the chunker needs).
"""

from __future__ import annotations

import re

from chunking.types import Chunk, Frame

_STRATEGY_NAME = "fixed_window"

_INLINE_TAG_RE = re.compile(r"<[^>]*>")
_CUE_INDEX_RE = re.compile(r"^\d+$")
_HEADER_PREFIXES = ("WEBVTT", "Kind:", "Language:", "NOTE")
_TIMESTAMP_LINE_RE = re.compile(
    r"^(?P<start>\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2}\.\d{3})"
)
_SENTENCE_END_RE = re.compile(r"[.!?](?:\s|$)")


def _ts_to_seconds(ts: str) -> float:
    """Convert `HH:MM:SS.mmm` to seconds (float)."""
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _parse_vtt(content: str) -> list[tuple[float, float, str]]:
    """Parse VTT content into a list of `(start_sec, end_sec, text)` tuples.

    Header lines (WEBVTT / Kind: / Language: / NOTE), cue-index lines (pure
    integers), and inline tag markup (`<00:00:15.519><c>…</c>`) are stripped.
    Cues whose text is empty after stripping are dropped.
    """
    cues: list[tuple[float, float, str]] = []
    current_start: float | None = None
    current_end: float | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_start, current_end, current_lines
        if current_start is not None and current_end is not None and current_lines:
            text = " ".join(ln for ln in current_lines if ln).strip()
            if text:
                cues.append((current_start, current_end, text))
        current_start = None
        current_end = None
        current_lines = []

    for raw in content.splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        if line.startswith(_HEADER_PREFIXES):
            flush()
            continue
        m = _TIMESTAMP_LINE_RE.match(line)
        if m:
            flush()
            current_start = _ts_to_seconds(m.group("start"))
            current_end = _ts_to_seconds(m.group("end"))
            continue
        if _CUE_INDEX_RE.match(line):
            # cue index appears before the timestamp; only treat as index
            # when no timestamp has been seen yet for this cue
            if current_start is None:
                continue
        stripped = _INLINE_TAG_RE.sub("", line).strip()
        if stripped:
            current_lines.append(stripped)

    flush()
    return cues


def _truncate_to_max_tokens(text: str, max_tokens: int) -> tuple[str, int]:
    """Truncate `text` to at most `max_tokens` whitespace-delimited tokens.

    Prefers cutting at a sentence boundary if one exists in the last 10% of
    the kept region; otherwise a plain word cut.
    """
    words = text.split()
    if len(words) <= max_tokens:
        return text, len(words)

    kept = words[:max_tokens]
    candidate = " ".join(kept)

    tail_start = max(0, int(len(candidate) * 0.9))
    last_boundary = -1
    for match in _SENTENCE_END_RE.finditer(candidate):
        end_idx = match.start() + 1
        if end_idx >= tail_start:
            last_boundary = end_idx
    if last_boundary > 0:
        snapped = candidate[:last_boundary].rstrip()
        return snapped, len(snapped.split())

    return candidate, len(kept)


def _snap_to_sentence(text: str) -> str:
    """If `text` does not end on sentence-terminal punctuation, trim back to
    the last sentence-final delimiter. Returns `text` unchanged when no
    earlier boundary exists.
    """
    stripped = text.rstrip()
    if not stripped:
        return stripped
    if stripped[-1] in ".!?":
        return stripped
    last = max(stripped.rfind(". "), stripped.rfind("! "), stripped.rfind("? "))
    if last == -1:
        return stripped
    return stripped[: last + 1]


def chunk(
    transcript_vtt: str,
    frames: list[Frame],
    *,
    video_id: str,
    window_sec: float = 30.0,
    overlap_sec: float = 5.0,
    max_tokens: int = 512,
) -> list[Chunk]:
    """Emit fixed-window chunks over a VTT transcript.

    See module docstring for algorithm details.
    """
    if overlap_sec >= window_sec:
        raise ValueError(
            f"overlap_sec ({overlap_sec}) must be < window_sec ({window_sec})"
        )

    cues = _parse_vtt(transcript_vtt)
    if not cues:
        return []

    stride = window_sec - overlap_sec
    transcript_end = max(end for _, end, _ in cues)

    chunks: list[Chunk] = []
    window_start = 0.0

    while window_start < transcript_end:
        window_end = window_start + window_sec
        in_window = [
            (s, e, t) for (s, e, t) in cues if window_start <= (s + e) / 2.0 < window_end
        ]
        if not in_window:
            window_start += stride
            continue

        actual_start = in_window[0][0]
        actual_end = in_window[-1][1]
        joined = " ".join(t for _, _, t in in_window)
        snapped = _snap_to_sentence(joined)
        text, token_count = _truncate_to_max_tokens(snapped, max_tokens)

        frame_secs = [f.frame_sec for f in frames if actual_start <= f.frame_sec < actual_end]

        chunks.append(
            Chunk(
                video_id=video_id,
                start_sec=actual_start,
                end_sec=actual_end,
                text=text,
                chunking_strategy=_STRATEGY_NAME,
                frame_secs=frame_secs,
                token_count=token_count,
            )
        )
        window_start += stride

    return chunks
