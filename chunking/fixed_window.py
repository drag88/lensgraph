"""Fixed-window chunker: the v0 baseline strategy.

Slides a `window_sec` window across the transcript with `overlap_sec`
overlap between consecutive windows. Cues are assigned to a window when
their midpoint falls inside that window. Sentence boundaries are
preferred at the trailing edge when the natural cut would land
mid-sentence. Token count is enforced via word truncation against
`max_tokens`.

VTT parsing (roll-up dedup, time alignment) lives in `chunking/vtt.py`,
shared with every other strategy so they chunk identical input.
"""

from __future__ import annotations

import re

from chunking.types import Chunk, Frame
from chunking.vtt import parse_vtt

STRATEGY_NAME = "fixed_window"

_SENTENCE_END_RE = re.compile(r"[.!?](?:\s|$)")
_MAX_SNAP_TRIM_WORDS = 12


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
    """If `text` does not end on sentence-terminal punctuation, trim back
    to the last `. ` / `! ` / `? ` — BUT only when the trim would drop at
    most _MAX_SNAP_TRIM_WORDS words. Beyond that bound, return text
    unchanged so the chunker never emits text representing only the early
    part of a span while keeping the full timestamp range.
    """
    stripped = text.rstrip()
    if not stripped:
        return stripped
    if stripped[-1] in ".!?":
        return stripped
    last = max(stripped.rfind(". "), stripped.rfind("! "), stripped.rfind("? "))
    if last == -1:
        return stripped
    tail = stripped[last + 1 :]
    if len(tail.split()) > _MAX_SNAP_TRIM_WORDS:
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
        raise ValueError(f"overlap_sec ({overlap_sec}) must be < window_sec ({window_sec})")

    cues = parse_vtt(transcript_vtt)
    if not cues:
        return []

    stride = window_sec - overlap_sec
    transcript_end = max(end for _, end, _ in cues)

    chunks: list[Chunk] = []
    window_start = 0.0

    while window_start < transcript_end:
        window_end = window_start + window_sec
        in_window = [(s, e, t) for (s, e, t) in cues if window_start <= (s + e) / 2.0 < window_end]
        if not in_window:
            window_start += stride
            continue

        actual_start = in_window[0][0]
        actual_end = in_window[-1][1]
        joined = " ".join(t for _, _, t in in_window if t).strip()
        # A window holding only residue-collapsed (empty-text) cues — e.g. a
        # pure-music head/tail — emits no chunk. Real-speech windows always
        # carry new content, so this never drops a content chunk.
        if not joined:
            window_start += stride
            continue
        snapped = _snap_to_sentence(joined)
        text, token_count = _truncate_to_max_tokens(snapped, max_tokens)

        frame_secs = [f.frame_sec for f in frames if actual_start <= f.frame_sec < actual_end]

        chunks.append(
            Chunk(
                video_id=video_id,
                start_sec=actual_start,
                end_sec=actual_end,
                text=text,
                chunking_strategy=STRATEGY_NAME,
                frame_secs=frame_secs,
                token_count=token_count,
            )
        )
        window_start += stride

    return chunks
