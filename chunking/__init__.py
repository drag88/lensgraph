"""Chunking strategy registry."""

from __future__ import annotations

from collections.abc import Callable

from chunking.fixed_window import STRATEGY_NAME as FIXED_WINDOW
from chunking.fixed_window import chunk as fixed_window_chunk
from chunking.transcript_segment import STRATEGY_NAME as TRANSCRIPT_SEGMENT
from chunking.transcript_segment import chunk as transcript_segment_chunk

_REGISTRY: dict[str, Callable] = {
    FIXED_WINDOW: fixed_window_chunk,
    TRANSCRIPT_SEGMENT: transcript_segment_chunk,
}


def get_chunker(name: str) -> Callable:
    """Return the chunker callable for `name`. Raises KeyError if unknown."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown chunking strategy: {name}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]
