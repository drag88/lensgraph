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

import yaml


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

    Two resolution modes:

      paths={video_id: Path}  explicit map; preferred for production use
                              with talks.yaml-derived transcript_path values.
      root=<dir>              <root>/<video_id>.vtt convention; kept for
                              fixture-style tests staging synthetic VTTs.

    `paths` takes precedence; `root` is fallback for any video_id absent
    from `paths`. Raises FileNotFoundError if neither resolves or the
    resolved file does not exist — the pipeline transaction rolls back,
    leaving no partial talks row.
    """

    paths: dict[str, Path] | None = None
    root: Path | None = None
    captions_source: str = "manual_transcript"

    def fetch(self, video_id: str) -> FetchResult:
        if self.paths is not None and video_id in self.paths:
            path = self.paths[video_id]
        elif self.root is not None:
            path = self.root / f"{video_id}.vtt"
        else:
            raise FileNotFoundError(
                f"no transcript for {video_id!r}: neither paths nor root configured"
            )
        if not path.exists():
            raise FileNotFoundError(f"no transcript at {path}")
        return FetchResult(
            transcript_path=path,
            transcript_sha256=_sha256_file(path),
            captions_source=self.captions_source,
        )


def paths_from_talks_yaml(
    corpus_dir: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Path]:
    """Read <corpus_dir>/talks.yaml and return {video_id: absolute transcript Path}.

    transcript_path values in talks.yaml are repo-relative; resolved against
    repo_root. Default repo_root = corpus_dir.parent.parent.parent (i.e.
    eval/corpora/<corpus> backs out to repo root).
    """
    corpus_dir = Path(corpus_dir)
    if repo_root is None:
        repo_root = corpus_dir.resolve().parent.parent.parent
    else:
        repo_root = Path(repo_root).resolve()

    talks_yaml = corpus_dir / "talks.yaml"
    with talks_yaml.open(encoding="utf-8") as f:
        talks = yaml.safe_load(f)
    return {
        talk["video_id"]: (repo_root / talk["transcript_path"]).resolve()
        for talk in talks
    }


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
