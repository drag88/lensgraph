"""Shared VTT parsing for chunking strategies.

Every chunking strategy consumes the same transcript: a YouTube WebVTT file.
This module owns parsing it into clean ``(start_sec, end_sec, text)`` cues so
each strategy works from identical input (anti-contamination rule 3: hold all
non-strategy variables constant).

YouTube auto-captions use a "roll-up" format — each cue repeats the previous
line as on-screen residue and pads with blank lines, so a naive line-join
triplicates text and bleeds earlier words into later spans. ``parse_vtt``
returns deduped, time-aligned cues. See the chunking handoff for the bug this
replaced.
"""

from __future__ import annotations

import html
import re

_INLINE_TAG_RE = re.compile(r"<[^>]*>")
_SPEAKER_RE = re.compile(r"^>>\s*")
_CUE_INDEX_RE = re.compile(r"^\d+$")
_HEADER_PREFIXES = ("WEBVTT", "Kind:", "Language:", "NOTE")
_TIMESTAMP_LINE_RE = re.compile(
    r"^(?P<start>\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2}\.\d{3})"
)


def ts_to_seconds(ts: str) -> float:
    """Convert ``HH:MM:SS.mmm`` to seconds (float)."""
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def normalize_line(line: str) -> str:
    """Strip inline word-timing tags, unescape HTML entities, drop a leading
    ``>>`` speaker marker, and collapse internal whitespace.

    ``&gt;&gt;`` -> ``>>`` -> dropped; ``&amp;`` -> ``&``. Applied to every cue
    body line so the rolling-residue compare in ``dedup_rolling`` matches on the
    settled text, and so chunk text carries no ``&gt;`` / ``>>`` noise.
    """
    s = _INLINE_TAG_RE.sub("", line)
    s = html.unescape(s)
    s = _SPEAKER_RE.sub("", s)
    return " ".join(s.split()).strip()


def parse_vtt_cues(content: str) -> list[tuple[float, float, list[str]]]:
    """Parse VTT content into ``(start_sec, end_sec, lines)`` tuples.

    ``lines`` is the ordered list of normalized, non-empty body lines for the
    cue. Returning the line split — rather than a pre-joined string — lets
    ``dedup_rolling`` strip roll-up residue by comparing each cue's leading line
    against the previous cue's tail.

    Header/cue-index lines are skipped; blank/whitespace lines are in-cue
    padding, NOT separators (the active word-tagged line sits after a blank line
    inside the cue, so flushing on blank lines would drop freshly-spoken cues).
    Cues are delimited by the next timestamp / header / EOF; a cue with no
    non-empty line is dropped.
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
            continue
        if line.startswith(_HEADER_PREFIXES):
            flush()
            continue
        m = _TIMESTAMP_LINE_RE.match(line)
        if m:
            flush()
            current_start = ts_to_seconds(m.group("start"))
            current_end = ts_to_seconds(m.group("end"))
            continue
        if _CUE_INDEX_RE.match(line):
            if current_start is None:
                continue
        normalized = normalize_line(line)
        if normalized:
            current_lines.append(normalized)

    flush()
    return cues


def dedup_rolling(
    parsed: list[tuple[float, float, list[str]]],
) -> list[tuple[float, float, str]]:
    """Collapse roll-up residue into clean ``(start, end, text)`` cues.

    Drops any leading line that verbatim-matches the previous emitted cue's last
    kept line. Boundary-preserving: every input cue is emitted with its
    ``(start, end)`` untouched — flicker cues whose text is pure residue collapse
    to an empty string but are KEPT, so the cue-midpoint stream is identical to
    the input. ``prev_tail`` advances only when a cue contributes new content.
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


def parse_vtt(content: str) -> list[tuple[float, float, str]]:
    """Parse + dedup in one call. The canonical entry point for chunkers:
    returns clean, time-aligned ``(start_sec, end_sec, text)`` cues. Cues whose
    text collapses to empty (pure roll-up residue) are retained as zero-text
    spans so the timestamp stream is complete; chunkers skip empty text."""
    return dedup_rolling(parse_vtt_cues(content))
