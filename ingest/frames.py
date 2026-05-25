"""Frame sampling — ffmpeg subprocess wrapper for the visual ingest stage.

`sample()` extracts one PNG per `every_sec` seconds from a window
`[start_sec, start_sec + duration_sec)` of a local video file. Returns
`FrameSample` rows ready for `db.repos.frames.upsert`. The handler in
`ingest/handlers.py` is the producer of the `frames` table; this module
owns the on-disk artefacts and the sha256 audit trail.

Design contracts:

  - `frame_sec` on every returned `FrameSample` is RELATIVE to the talk
    (i.e. 0, every_sec, 2*every_sec, ...) — for chapter slices the caller
    passes `start_sec=talk.source_start_sec` and the values align to the
    talk's own zero, not the parent video's.
  - PNG output (lossless) so sha256 round-trips deterministically across
    re-runs. The lavfi colour/test sources used by the synthetic fixtures
    produce byte-identical frames every time the same args are passed.
  - Frame files live under `frames/<video_id>/frame_NNNNNN.png` by default;
    callers can override via `out_dir`. The directory is purged of stale
    `frame_*.png` files on every call so leftover frames from a longer
    earlier sample cannot bleed into the new sha256 set.
  - `ffmpeg` must be on PATH. Missing binary raises `FfmpegNotAvailableError`
    so the worker crashes loudly rather than producing zero-frame samples.

Local video path convention (`default_video_path_for_talk`):

  `videos/<corpus>/<source_video_id or video_id>.mp4`, where `<corpus>`
  is the parent dir name of `talk.transcript_path` (e.g. talks ingested
  under `transcripts/ai_engineering_v0/` look for videos under
  `videos/ai_engineering_v0/`). Chapter-sliced talks use the parent
  `source_video_id` — there is exactly one physical .mp4 on disk per
  source video regardless of how many chapter slices reference it.

  As of 2026-05-25 there are NO local .mp4 files in this repo — the
  convention is wired but unexercised against the real corpus. The slow
  tests synthesise their own lavfi videos for the hard gate.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from db.repos.talks import Talk


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FRAMES_DIR = REPO_ROOT / "frames"


class FfmpegNotAvailableError(RuntimeError):
    """`shutil.which('ffmpeg')` returned None — frame sampling cannot proceed."""


@dataclass(frozen=True)
class FrameSample:
    """One sampled frame, ready for `db.repos.frames.upsert`.

    `frame_sec` is RELATIVE to the talk's zero (see module docstring).
    `image_path` is the absolute on-disk path as a string — matches the
    `frames.image_path text` column.
    """

    video_id: str
    frame_sec: float
    image_path: str
    sha256: str


def _check_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None:
        raise FfmpegNotAvailableError(
            "ffmpeg not on PATH; install via `brew install ffmpeg` (macOS) "
            "or the platform equivalent."
        )


def _frames_dir_for(video_id: str, out_dir: Path | None) -> Path:
    base = out_dir if out_dir is not None else DEFAULT_FRAMES_DIR / video_id
    base = base.resolve()
    base.mkdir(parents=True, exist_ok=True)
    return base


def _purge_stale_frames(target_dir: Path) -> None:
    for old in target_dir.glob("frame_*.png"):
        old.unlink()


def sample(
    video_path: Path,
    *,
    video_id: str,
    every_sec: float = 10.0,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
    out_dir: Path | None = None,
) -> list[FrameSample]:
    """Extract one PNG per `every_sec` from `[start_sec, start_sec+duration_sec)`.

    Returns `FrameSample` rows with `frame_sec` aligned to the talk's zero
    (0, every_sec, 2*every_sec, ...), `image_path` as the absolute on-disk
    path of the PNG, and `sha256` of the final file bytes.
    """
    if not isinstance(video_path, Path):
        video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found at {video_path}")
    if every_sec <= 0:
        raise ValueError(f"every_sec must be > 0, got {every_sec}")
    if start_sec < 0:
        raise ValueError(f"start_sec must be >= 0, got {start_sec}")
    if duration_sec is not None and duration_sec <= 0:
        raise ValueError(f"duration_sec must be > 0 when set, got {duration_sec}")

    _check_ffmpeg_available()

    target_dir = _frames_dir_for(video_id, out_dir)
    _purge_stale_frames(target_dir)

    cmd: list[str] = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-fflags",
        "+bitexact",
        "-ss",
        f"{start_sec:.3f}",
        "-i",
        str(video_path),
    ]
    if duration_sec is not None:
        cmd += ["-t", f"{duration_sec:.3f}"]
    cmd += [
        "-vf",
        f"fps=1/{every_sec}",
        "-flags",
        "+bitexact",
        "-pix_fmt",
        "rgb24",
        str(target_dir / "frame_%06d.png"),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    written = sorted(target_dir.glob("frame_*.png"))
    samples: list[FrameSample] = []
    for i, p in enumerate(written):
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        samples.append(
            FrameSample(
                video_id=video_id,
                frame_sec=float(i) * every_sec,
                image_path=str(p),
                sha256=sha,
            )
        )
    return samples


def default_video_path_for_talk(talk: Talk) -> Path:
    """Convention helper — `videos/<corpus>/<physical_id>.mp4` from a `Talk`.

    `<corpus>` is the parent dir name of `talk.transcript_path` (matches
    how transcripts are laid out). `<physical_id>` is `source_video_id`
    when set (chapter slice) else `video_id`. The returned path may not
    exist on disk — the caller raises if so.
    """
    transcript_p = Path(talk.transcript_path)
    parts = transcript_p.parts
    if len(parts) >= 3 and parts[0] == "transcripts":
        corpus_dir = parts[1]
    else:
        corpus_dir = "_default"
    physical_id = talk.source_video_id or talk.video_id
    return REPO_ROOT / "videos" / corpus_dir / f"{physical_id}.mp4"
