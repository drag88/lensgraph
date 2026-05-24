"""Video + caption acquisition.

`Fetcher` is a Protocol; anything that can produce a transcript on disk for
a given video_id conforms. Two concrete implementations in this slice:

  LocalFsFetcher  reads a pre-staged transcript from a local directory.
                  Used by tests and by offline re-ingestion of an already-
                  downloaded corpus. No network I/O.

  YtDlpFetcher    surface-only stub for the production path. The actual
                  yt-dlp shell-out lands when the ingest CLI ships; this
                  slice only commits to the type + constructor surface so
                  the rest of the codebase can take a Fetcher and not
                  branch on local-vs-network.

A Fetcher returns a FetchResult: transcript_path + sha256 + captions_source.
The pipeline reconciler uses that to write the talks row, probe quality,
and decide downstream skips.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class FetchResult:
    """Output of a Fetcher.fetch() call."""

    transcript_path: Path
    transcript_sha256: str  # 64-char lowercase hex
    captions_source: str  # one of: youtube_auto | youtube_manual | manual_transcript | whisperx_large_v3


class Fetcher(Protocol):
    def fetch(self, video_id: str) -> FetchResult: ...


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class LocalFsFetcher:
    """Reads pre-staged transcripts from a local directory.

    Resolves `<root>/<video_id>.vtt`. Raises FileNotFoundError if absent —
    the pipeline transaction rolls back, leaving no partial talks row.
    """

    root: Path
    captions_source: str = "manual_transcript"

    def fetch(self, video_id: str) -> FetchResult:
        path = self.root / f"{video_id}.vtt"
        if not path.exists():
            raise FileNotFoundError(f"no transcript at {path}")
        return FetchResult(
            transcript_path=path,
            transcript_sha256=_sha256_file(path),
            captions_source=self.captions_source,
        )


@dataclass(frozen=True)
class YtDlpFetcher:
    """yt-dlp-backed fetcher. Surface only in this slice.

    The actual subprocess + auto-vs-manual caption negotiation + sha256 of
    the downloaded VTT lands when the ingest CLI ships. Until then any
    call raises NotImplementedError so callers fail loudly rather than
    silently no-op.
    """

    cache_root: Path

    def fetch(self, video_id: str) -> FetchResult:
        raise NotImplementedError(
            f"YtDlpFetcher.fetch({video_id!r}) — integration lands in the "
            "ingest-CLI slice; use LocalFsFetcher for offline / test paths."
        )
