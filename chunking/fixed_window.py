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

import html
import re

from chunking.types import Chunk, Frame

STRATEGY_NAME = "fixed_window"

_INLINE_TAG_RE = re.compile(r"<[^>]*>")
_SPEAKER_RE = re.compile(r"^>>\s*")
_CUE_INDEX_RE = re.compile(r"^\d+$")
_HEADER_PREFIXES = ("WEBVTT", "Kind:", "Language:", "NOTE")
_TIMESTAMP_LINE_RE = re.compile(
    r"^(?P<start>\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2}\.\d{3})"
)
_SENTENCE_END_RE = re.compile(r"[.!?](?:\s|$)")
_MAX_SNAP_TRIM_WORDS = 12


def _ts_to_seconds(ts: str) -> float:
    """Convert `HH:MM:SS.mmm` to seconds (float)."""
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _normalize_line(line: str) -> str:
    """Strip inline word-timing tags, unescape HTML entities, drop a leading
    `>>` speaker marker, and collapse internal whitespace.

    `&gt;&gt;` -> `>>` -> dropped; `&amp;` -> `&`. Applied to every cue body
    line so the rolling-residue compare in `_dedup_rolling` matches on the
    settled text, and so chunk text carries no `&gt;` / `>>` noise.
    """
    s = _INLINE_TAG_RE.sub("", line)
    s = html.unescape(s)
    s = _SPEAKER_RE.sub("", s)
    return " ".join(s.split()).strip()


def _parse_vtt(content: str) -> list[tuple[float, float, list[str]]]:
    """Parse VTT content into `(start_sec, end_sec, lines)` tuples.

    `lines` is the ordered list of normalized, non-empty body lines for the
    cue (see `_normalize_line`). Returning the line split — rather than a
    pre-joined string — lets `_dedup_rolling` strip YouTube roll-up residue
    by comparing each cue's leading line against the previous cue's tail.

    Header lines (WEBVTT / Kind: / Language: / NOTE) and cue-index lines are
    skipped; blank/whitespace lines are in-cue padding, not separators. Cues
    are delimited by the next timestamp / header / EOF, and a cue with no
    non-empty line is dropped.

    Note: recovering the freshly-spoken (word-tagged) cues — which the
    original blank-line-flush bug discarded — changes the cue-midpoint stream
    `chunk()` windows over, so chunk start/end boundaries shift relative to
    the buggy substrate. This is the correct alignment (text now matches its
    timestamp); the change is re-verified by re-chunking + re-embedding.
    """
    cues: list[tuple[float, float, list[str]]] = []
    current_start: float | None = None
    current_end: float | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_start, current_end, current_lines
        if current_start is not None and current_end is not None and current_lines:
            cues.append((current_start, current_end, current_lines))
        current_start = None
        current_end = None
        current_lines = []

    for raw in content.splitlines():
        line = raw.strip()
        if not line:
            # Blank/whitespace lines are in-cue padding in YouTube roll-up
            # VTT, NOT cue separators: the active word-tagged line sits AFTER
            # a blank line inside the cue. Flushing here (the original bug)
            # dropped every freshly-spoken cue and kept only the residue
            # flickers, time-shifting all chunk text. Cues are delimited by
            # the next timestamp / header / EOF instead.
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
        normalized = _normalize_line(line)
        if normalized:
            current_lines.append(normalized)

    flush()
    return cues


def _dedup_rolling(
    parsed: list[tuple[float, float, list[str]]],
) -> list[tuple[float, float, str]]:
    """Collapse YouTube roll-up residue into clean `(start, end, text)` cues.

    Each roll-up cue repeats the previous cue's trailing line(s) as on-screen
    residue before adding the newly spoken line. Naively joining every line
    triplicates text and bleeds earlier words into later spans. This pass
    drops any leading line that verbatim-matches the previous emitted cue's
    last kept line.

    Boundary-preserving by construction: every input cue is emitted with its
    `(start, end)` untouched — flicker cues whose text is pure residue collapse
    to an empty string but are KEPT, so the cue-midpoint stream `chunk()`
    windows over is identical to the input. `prev_tail` advances only when a
    cue contributes genuinely new content.
    """
    out: list[tuple[float, float, str]] = []
    prev_tail: str | None = None
    for start, end, lines in parsed:
        kept = list(lines)
        while kept and prev_tail is not None and kept[0] == prev_tail:
            kept.pop(0)
        if kept:
            prev_tail = kept[-1]
        out.append((start, end, " ".join(kept).strip()))
    return out


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

    cues = _dedup_rolling(_parse_vtt(transcript_vtt))
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
