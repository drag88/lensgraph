"""Public types for the chunking layer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Frame:
    """A sampled video frame timestamp (seconds from talk start)."""

    frame_sec: float


@dataclass(frozen=True)
class Chunk:
    """One retrieval unit: timestamped, tokenized, optionally framed."""

    video_id: str
    start_sec: float
    end_sec: float
    text: str
    chunking_strategy: str
    frame_secs: list[float]
    token_count: int
