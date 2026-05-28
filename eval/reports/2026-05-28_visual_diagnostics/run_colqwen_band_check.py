"""One-off ColQwen processor band check (prompt47, ADR 005 follow-up).

Prints the processor's min_pixels / max_pixels / size band, then prints the
W*H of 5 sampled frames for W_CYk2ogcDI and reports whether they sit inside
the band (so ColQwen does no resize), or whether the processor downsamples
or upsamples them.

Scope: PRINT and CHECK only. No ablation, no DB writes, no embed run.
Run from repo root:

    uv run python eval/reports/2026-05-28_visual_diagnostics/run_colqwen_band_check.py

Output is plain text on stdout; the companion ``colqwen_band_check.md``
report quotes the numbers this script prints.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from embed.colqwen import _model, _resolve_model_id

VIDEO_ID = "W_CYk2ogcDI"
FRAMES_DIR = Path(__file__).resolve().parents[3] / "frames" / VIDEO_ID
SAMPLE_INDICES = [1, 25, 50, 75, 100]  # spread across the 113 frames


def main() -> None:
    model_id = _resolve_model_id()
    print(f"[model] colqwen2.5 provider_model_id = {model_id!r}")

    _, processor, _ = _model()
    ip = processor.image_processor

    min_pixels = getattr(ip, "min_pixels", None)
    max_pixels = getattr(ip, "max_pixels", None)
    size = getattr(ip, "size", None)

    print(f"[processor.image_processor.min_pixels] = {min_pixels}")
    print(f"[processor.image_processor.max_pixels] = {max_pixels}")
    print(f"[processor.image_processor.size]       = {size}")

    # ColQwen2.5 (built on Qwen2.5-VL) exposes the band via size.shortest_edge
    # (min_pixels) and size.longest_edge (max_pixels) when min_pixels /
    # max_pixels are None. Fall back to those when present.
    if (min_pixels is None or max_pixels is None) and size is not None:
        shortest = getattr(size, "shortest_edge", None) or (
            size.get("shortest_edge") if isinstance(size, dict) else None
        )
        longest = getattr(size, "longest_edge", None) or (
            size.get("longest_edge") if isinstance(size, dict) else None
        )
        if shortest is not None and longest is not None:
            if min_pixels is None:
                min_pixels = int(shortest)
            if max_pixels is None:
                max_pixels = int(longest)
            print(
                f"[band-from-size] min_pixels<-shortest_edge={min_pixels:,}, "
                f"max_pixels<-longest_edge={max_pixels:,}"
            )

    print()
    print(f"[frames] sampling 5 frames from {FRAMES_DIR}")

    if not FRAMES_DIR.exists():
        raise SystemExit(f"frames dir missing: {FRAMES_DIR}")

    rows: list[tuple[str, int, int, int]] = []
    for i in SAMPLE_INDICES:
        p = FRAMES_DIR / f"frame_{i:06d}.png"
        if not p.exists():
            print(f"  missing: {p.name}")
            continue
        with Image.open(p) as im:
            w, h = im.size
        rows.append((p.name, w, h, w * h))
        print(f"  {p.name}  {w}x{h}  =  {w * h:,} px")

    if min_pixels is None or max_pixels is None:
        print("\n[verdict] processor exposes no min/max_pixels — cannot classify band.")
        return

    print()
    print(f"[band] {min_pixels:,} .. {max_pixels:,} px")
    for name, w, h, n in rows:
        if n < min_pixels:
            verdict = f"BELOW band (would be UPSAMPLED toward >= {min_pixels:,})"
        elif n > max_pixels:
            verdict = f"ABOVE band (would be DOWNSAMPLED toward <= {max_pixels:,})"
        else:
            verdict = "INSIDE band (no resize)"
        print(f"  {name}  {w}x{h}={n:,} px -> {verdict}")


if __name__ == "__main__":
    main()
