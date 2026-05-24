"""Chunking strategy registry."""

from __future__ import annotations

from collections.abc import Callable

from chunking.fixed_window import chunk as fixed_window_chunk

_REGISTRY: dict[str, Callable] = {
    "fixed_window": fixed_window_chunk,
}


def get_chunker(name: str) -> Callable:
    """Return the chunker callable for `name`. Raises KeyError if unknown."""
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown chunking strategy: {name}; available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]
