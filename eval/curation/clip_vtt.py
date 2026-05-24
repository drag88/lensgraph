#!/usr/bin/env python3
"""Clip a WebVTT transcript to a source time range and rebase cue times to 0.

Example:
    python eval/curation/clip_vtt.py \
      --input transcripts/ai_engineering_v0/source/m12vGjfbNlo.en.vtt \
      --output transcripts/ai_engineering_v0/aie_sg_2026_d2_arize_alyx.en.vtt \
      --start 516 \
      --end 1494
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

TIMESTAMP_RE = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2}\.\d{3}) --> "
    r"(?P<end>\d{2}:\d{2}:\d{2}\.\d{3})(?P<settings>.*)"
)

# Inline word-level karaoke tags YouTube emits inside cue payloads, e.g.
# `Good<00:08:36.560><c> morning</c>`. These carry source-livestream offsets
# that the cue-header rebase does not touch, so a chapter slice would end up
# internally inconsistent. They are display-only and safe to strip.
INLINE_TIMESTAMP_RE = re.compile(r"<\d{2}:\d{2}:\d{2}\.\d{3}>")
CUE_CLASS_RE = re.compile(r"</?c(?:\.[^>]+)?>")


def parse_time(ts: str) -> float:
    hours, minutes, seconds = ts.split(":")
    return (int(hours) * 3600) + (int(minutes) * 60) + float(seconds)


def format_time(total_seconds: float) -> str:
    total_ms = max(0, round(total_seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{millis:03}"


def strip_inline_tags(payload: str) -> str:
    payload = INLINE_TIMESTAMP_RE.sub("", payload)
    payload = CUE_CLASS_RE.sub("", payload)
    return payload


def clip_block(block: str, start_sec: float, end_sec: float) -> str | None:
    match = TIMESTAMP_RE.search(block)
    if not match:
        return None

    cue_start = parse_time(match.group("start"))
    cue_end = parse_time(match.group("end"))
    if cue_end <= start_sec or cue_start >= end_sec:
        return None

    clipped_start = max(cue_start, start_sec) - start_sec
    clipped_end = min(cue_end, end_sec) - start_sec
    new_timing = (
        f"{format_time(clipped_start)} --> {format_time(clipped_end)}"
        f"{match.group('settings')}"
    )
    rebased = TIMESTAMP_RE.sub(new_timing, block, count=1)
    return strip_inline_tags(rebased)


def clip_vtt(source: str, start_sec: float, end_sec: float) -> str:
    if end_sec <= start_sec:
        raise ValueError(f"end ({end_sec}) must be greater than start ({start_sec})")

    normalized = source.replace("\r\n", "\n").replace("\r", "\n")
    blocks = normalized.split("\n\n")
    header = blocks[0].strip()
    kept = [
        clipped
        for block in blocks[1:]
        if (clipped := clip_block(block.strip(), start_sec, end_sec))
    ]
    return "\n\n".join([header, *kept]) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", required=True, type=float)
    parser.add_argument("--end", required=True, type=float)
    args = parser.parse_args()

    clipped = clip_vtt(args.input.read_text(), args.start, args.end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(clipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
