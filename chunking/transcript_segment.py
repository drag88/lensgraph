"""Transcript-segment chunker: sentence-boundary-aware strategy.

The fixed_window baseline cuts on a fixed time grid, so chunks routinely start
and end mid-sentence (the 2026-06-01 boundary audit measured this — human
edge_sensibility dropped on continuous-speech talks). transcript_segment cuts
on SENTENCE boundaries instead: it accumulates consecutive cues until the chunk
both clears ``target_sec`` AND ends on terminal punctuation, then starts the
next chunk at the following cue. So every edge lands at a completed thought, and
the next chunk opens a new sentence.

Trade-offs vs fixed_window: non-overlapping (a sentence belongs to one chunk),
slightly uneven durations (a chunk runs until the current sentence finishes).
``max_tokens`` is a safety cut for a runaway sentence with no terminator.

VTT parsing (roll-up dedup, time alignment) is shared via ``chunking/vtt.py`` so
this strategy chunks the exact same input as fixed_window.
"""

from __future__ import annotations

from chunking.types import Chunk, Frame
from chunking.vtt import parse_vtt

STRATEGY_NAME = "transcript_segment"

_TERMINAL = (".", "!", "?")


def _ends_sentence(text: str) -> bool:
    s = text.rstrip()
    return bool(s) and s[-1] in _TERMINAL


def chunk(
    transcript_vtt: str,
    frames: list[Frame],
    *,
    video_id: str,
    target_sec: float = 30.0,
    max_tokens: int = 512,
) -> list[Chunk]:
    """Emit sentence-aligned chunks over a VTT transcript.

    A chunk closes when its span reaches ``target_sec`` AND its text ends on
    terminal punctuation, or when it reaches ``max_tokens`` (force cut). The
    trailing remainder is emitted as a final chunk even if it does not end on a
    sentence (the talk simply ended there).
    """
    cues = [(s, e, t) for s, e, t in parse_vtt(transcript_vtt) if t.strip()]
    if not cues:
        return []

    chunks: list[Chunk] = []
    cur: list[tuple[float, float, str]] = []

    def flush() -> None:
        if not cur:
            return
        start = cur[0][0]
        end = cur[-1][1]
        words = " ".join(t for _, _, t in cur).split()
        if len(words) > max_tokens:
            words = words[:max_tokens]
        text = " ".join(words)
        frame_secs = [f.frame_sec for f in frames if start <= f.frame_sec < end]
        chunks.append(
            Chunk(
                video_id=video_id,
                start_sec=start,
                end_sec=end,
                text=text,
                chunking_strategy=STRATEGY_NAME,
                frame_secs=frame_secs,
                token_count=len(words),
            )
        )
        cur.clear()

    for s, e, t in cues:
        cur.append((s, e, t))
        dur = e - cur[0][0]
        joined = " ".join(x[2] for x in cur)
        token_count = len(joined.split())
        if (dur >= target_sec and _ends_sentence(joined)) or token_count >= max_tokens:
            flush()

    flush()
    return chunks
