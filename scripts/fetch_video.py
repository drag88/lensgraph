"""Download source video(s) into ``videos/<corpus>/`` per ADR 005.

Reads ``eval/corpora/<corpus>/talks.yaml`` and constructs the right
yt-dlp invocation per talk:

* **Non-chapter talks** (no ``source_video_id``): download the whole
  video to ``videos/<corpus>/<video_id>.mp4``.
* **Chapter-sliced talks** (``source_video_id`` set): group by
  ``source_video_id`` and download
  ``yt-dlp --download-sections '*0-<END>'`` where
  ``END = max(source_end_sec across the group) + SAFETY_PAD_SEC`` (30 s
  default). The trimmed file preserves the source's t=0 so
  ``ingest.handlers.frames_handler`` keeps working unchanged. Output
  lands at ``videos/<corpus>/<source_video_id>.mp4``.

Idempotent: skips files that already exist on disk unless ``--force``.
Prints the exact command it ran for each video so an operator can
reproduce manually if the script breaks.

Lives under ``scripts/`` (not ``eval/runners/``) because it is
operational glue — it does not write to ``eval_runs`` / ``eval_results``
and has nothing to do with eval logic.

CLI::

    python -m scripts.fetch_video --video-id <id> [--corpus <name>] [--force]
    python -m scripts.fetch_video --corpus <name> [--force]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = "ai_engineering_v0"

# ADR 005 §"Decision": 30 s tail-pad past max(source_end_sec) so the
# last sampled frame in [start, end) lands safely inside the trimmed
# file even if yt-dlp's keyframe-aligned cut drifts by up to one GOP.
SAFETY_PAD_SEC = 30

# ADR 005 §"Frame-sample resolution": 720p is the working default;
# verify against ColQwen2_5_Processor.image_processor.min_pixels /
# max_pixels before locking. The selector falls back gracefully when
# 720p is not served by the source.
YT_DLP_FORMAT = "bv*[height<=720]+ba/b[height<=720]"


def _strip_query(url: str) -> str:
    """Drop chapter-start hints from a YouTube URL, keep ``v=<id>``.

    Talks.yaml entries for chapter slices typically carry a ``&t=...``
    parameter pointing at the chapter start; yt-dlp ignores it for
    ``--download-sections``, but stripping it makes the printed command
    canonical (sibling chapters from the same source produce identical
    URLs). The video id parameter ``v`` MUST stay — without it the URL
    no longer identifies a video.
    """
    parsed = urlparse(url)
    kept = [(k, v) for (k, v) in parse_qsl(parsed.query) if k == "v"]
    return urlunparse(parsed._replace(query=urlencode(kept), fragment=""))


def _load_talks(corpus_dir: Path) -> list[dict]:
    path = corpus_dir / "talks.yaml"
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or []
    if not isinstance(data, list):
        raise ValueError(f"{path}: top-level must be a YAML list")
    return data


def _video_path(corpus: str, physical_id: str) -> Path:
    return REPO_ROOT / "videos" / corpus / f"{physical_id}.mp4"


def build_commands(
    corpus: str,
    talks: list[dict],
    *,
    force: bool = False,
    only_video_id: str | None = None,
) -> list[tuple[Path, list[str], str]]:
    """Build the per-physical-file yt-dlp commands.

    Returns a list of ``(target_path, argv, reason)`` triples. ``reason``
    is a short human-readable string for the printed plan.

    If ``only_video_id`` is set, the result is filtered to commands that
    cover that talk's physical file. Chapter siblings sharing the same
    source are merged into one command per source — that is the whole
    point of one-file-per-source (ADR 005 §"Storage convention").
    """
    if only_video_id is not None:
        target_talk = next((t for t in talks if t["video_id"] == only_video_id), None)
        if target_talk is None:
            raise KeyError(
                f"video_id {only_video_id!r} not found in talks.yaml for corpus {corpus!r}"
            )
        target_physical = target_talk.get("source_video_id") or target_talk["video_id"]
        relevant = [
            t
            for t in talks
            if (t.get("source_video_id") or t["video_id"]) == target_physical
        ]
    else:
        relevant = list(talks)

    # Group chapter-sliced talks by source_video_id; keep non-chapter
    # talks as singletons (their "group" is just themselves).
    groups: dict[str, list[dict]] = defaultdict(list)
    for t in relevant:
        physical_id = t.get("source_video_id") or t["video_id"]
        groups[physical_id].append(t)

    commands: list[tuple[Path, list[str], str]] = []
    for physical_id, members in groups.items():
        sliced_members = [m for m in members if m.get("source_video_id") is not None]
        target = _video_path(corpus, physical_id)
        if target.exists() and not force:
            commands.append((target, [], "skip (exists; --force to overwrite)"))
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        url = _strip_query(members[0]["url"])
        argv = [
            "yt-dlp",
            "-f",
            YT_DLP_FORMAT,
            "--merge-output-format",
            "mp4",
            "-o",
            str(target),
        ]
        if sliced_members:
            end = max(int(m["source_end_sec"]) for m in sliced_members) + SAFETY_PAD_SEC
            argv += ["--download-sections", f"*0-{end}"]
            reason = (
                f"chapter source: end={end}s (max source_end_sec + {SAFETY_PAD_SEC}s pad), "
                f"chapters={[m['video_id'] for m in sliced_members]}"
            )
        else:
            reason = "full video (no chapter slice)"
        argv.append(url)
        commands.append((target, argv, reason))

    commands.sort(key=lambda item: item[0])
    return commands


def run(commands: list[tuple[Path, list[str], str]]) -> int:
    """Execute the planned commands serially. Returns 0 on success."""
    failures = 0
    for target, argv, reason in commands:
        rel = target.relative_to(REPO_ROOT)
        if not argv:
            sys.stdout.write(f"  - {rel}: {reason}\n")
            continue
        sys.stdout.write(f"  - {rel}: {reason}\n")
        sys.stdout.write(f"    $ {' '.join(argv)}\n")
        sys.stdout.flush()
        rc = subprocess.run(argv, check=False).returncode
        if rc != 0:
            sys.stderr.write(f"yt-dlp failed (rc={rc}) for {rel}\n")
            failures += 1
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--video-id", help="single video_id from talks.yaml")
    g.add_argument("--corpus-all", action="store_true", help="fetch every talk in --corpus")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--force", action="store_true", help="re-download even if file exists")
    args = ap.parse_args(argv)

    corpus_dir = REPO_ROOT / "eval" / "corpora" / args.corpus
    if not corpus_dir.exists():
        sys.stderr.write(f"corpus dir not found: {corpus_dir}\n")
        return 2

    talks = _load_talks(corpus_dir)
    commands = build_commands(
        args.corpus,
        talks,
        force=args.force,
        only_video_id=None if args.corpus_all else args.video_id,
    )
    if not commands:
        sys.stdout.write("nothing to do\n")
        return 0
    sys.stdout.write(f"fetch plan ({len(commands)} command(s)):\n")
    return run(commands)


if __name__ == "__main__":
    raise SystemExit(main())
