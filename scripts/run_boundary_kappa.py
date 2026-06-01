"""Boundary judge kappa — boundary_v1 vs human labels (PROVISIONAL).

Runs the content-hashed boundary judge (``boundary_v1``, deepseek-v3.2 —
cross-family vs the human labeler) on the 21 fixed_window boundary-audit clips,
then computes Cohen's kappa (linear / quadratic-weighted / binarized) against the
human labels for edge_sensibility (n=13, where human edge exists) and standalone
(n=21).

PROVISIONAL: edge labels were human-verified against the transcript after seeing
rubric-derived suggestions (defensible for a near-objective property, not
blind-by-construction); standalone labels are blind. fixed_window clips only.

Reads:
  dev/active/phase-2-week-10/boundary_clips.json
  dev/active/phase-2-week-10/boundary_human_labels.csv
  eval/runners/judges/boundary_v1.md   (system prompt, content-hashed)
Writes:
  dev/active/phase-2-week-10/boundary_kappa_results.json  (provisional, untracked)

Provenance: clip text comes from fixed_window chunks (verified against the local
VTTs under transcripts/ai_engineering_v0/ at each span — token-Jaccard 0.82-1.00);
the Arize talk is a chapter slice of m12vGjfbNlo, so its watch links use absolute
time (talk-relative + 516s) while spans here are talk-relative. Human edge labels
are `recheck_verified` (scored after seeing rubric-derived suggestions — not
blind); standalone rejudge rows are `cold_rejudge` (blind). The kappa is
measured-and-failing / provisional, NOT a validated boundary judge.

Inputs (clips, labels) and the results JSON are local/untracked dev artifacts:
this runner documents how the provisional kappa was produced and is not
reproducible from a clean checkout. The pure functions are covered by
eval/tests/test_boundary_kappa.py (no file or network dependency).

CLI:  DEEPINFRA_API_KEY=... uv run python -m scripts.run_boundary_kappa
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

from eval.runners import providers

_ROOT = Path(__file__).resolve().parent.parent
_CLIPS_PATH = _ROOT / "dev" / "active" / "phase-2-week-10" / "boundary_clips.json"
_LABELS_PATH = _ROOT / "dev" / "active" / "phase-2-week-10" / "boundary_human_labels.csv"
_JUDGE_PATH = _ROOT / "eval" / "runners" / "judges" / "boundary_v1.md"
_OUT_PATH = _ROOT / "dev" / "active" / "phase-2-week-10" / "boundary_kappa_results.json"

_JUDGE_CANDIDATE_ID = "deepseek-v3.2"
_SCORE_LABELS = [1, 2, 3, 4, 5]
_CLEAN_THRESHOLD = 4  # >= 4 counts as a "clean" score for the binarized kappa


# ---- pure functions (unit-tested) ----------------------------------------


def content_sha(text: str, *, n: int = 12) -> str:
    """First ``n`` hex chars of the SHA-256 of ``text`` — the judge-prompt
    provenance hash logged with every run (matches the recorded boundary_v1 sha)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def build_messages(judge_prompt: str, clip: dict) -> list[dict]:
    """System = the judge rubric; user = one clip (video_id, span, TEXT) in the
    shape boundary_v1.md says it receives."""
    user = (
        f"video_id: {clip['video_id']}\n"
        f"span: [{clip['start_sec']}, {clip['end_sec']}]\n"
        f"TEXT:\n{clip['text']}"
    )
    return [
        {"role": "system", "content": judge_prompt},
        {"role": "user", "content": user},
    ]


def parse_judge_scores(raw: str) -> dict:
    """Extract ``{edge_sensibility, standalone, rationale}`` from the judge's
    raw text. Tolerates code fences and surrounding prose by grabbing the first
    balanced JSON object. Raises ValueError if scores are missing or not ints
    in [1, 5]."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in judge output: {raw[:120]!r}")
    obj = json.loads(match.group(0))
    out = {}
    for key in ("edge_sensibility", "standalone"):
        val = obj.get(key)
        if not isinstance(val, int) or isinstance(val, bool) or not (1 <= val <= 5):
            raise ValueError(f"{key} must be an int in [1,5], got {val!r}")
        out[key] = val
    out["rationale"] = str(obj.get("rationale", ""))[:200]
    return out


def binarize(score: int, *, threshold: int = _CLEAN_THRESHOLD) -> int:
    """1 if the score is 'clean' (>= threshold), else 0 — for binarized kappa."""
    return 1 if score >= threshold else 0


def cohens_kappa(
    a: list[int], b: list[int], *, labels: list[int], weights: str | None = None
) -> float:
    """Cohen's kappa, matching ``sklearn.metrics.cohen_kappa_score``.

    ``weights``: ``None`` (unweighted — 0 on the diagonal, 1 off), ``"linear"``
    (``|i-j|``), or ``"quadratic"`` (``(i-j)**2``) over the ordinal ``labels``.

    Formula (sklearn): with confusion matrix ``C`` (rows=a, cols=b), per-label
    column sums ``c`` and row sums ``r`` over ``N`` pairs, expected
    ``E[i][j] = c[i]*r[j]/N``, and weight matrix ``w``:
    ``kappa = 1 - sum(w*C) / sum(w*E)``. Returns ``0.0`` when the weighted
    expected agreement is zero (degenerate single-category case)."""
    n = len(labels)
    idx = {v: i for i, v in enumerate(labels)}
    conf = [[0] * n for _ in range(n)]
    for x, y in zip(a, b, strict=True):
        conf[idx[x]][idx[y]] += 1
    total = sum(sum(row) for row in conf)
    col = [sum(conf[i][j] for i in range(n)) for j in range(n)]
    row = [sum(conf[i][j] for j in range(n)) for i in range(n)]
    expected = [[col[i] * row[j] / total for j in range(n)] for i in range(n)]
    if weights is None:
        w = [[0 if i == j else 1 for j in range(n)] for i in range(n)]
    elif weights == "linear":
        w = [[abs(i - j) for j in range(n)] for i in range(n)]
    elif weights == "quadratic":
        w = [[(i - j) ** 2 for j in range(n)] for i in range(n)]
    else:
        raise ValueError(f"unknown weights: {weights!r}")
    num = sum(w[i][j] * conf[i][j] for i in range(n) for j in range(n))
    den = sum(w[i][j] * expected[i][j] for i in range(n) for j in range(n))
    if den == 0:
        return 0.0
    return 1 - num / den


def kappa_set(human: list[int], judge: list[int]) -> dict:
    """Linear, quadratic-weighted, and binarized Cohen's kappa for one paired
    ordinal set. Returns floats rounded to 3 dp plus the pair count. Weighted
    variants pass the full 1–5 label set so absent categories are handled."""
    if len(human) != len(judge):
        raise ValueError("human and judge score lists must be the same length")
    if not human:
        return {"n": 0, "linear": None, "quadratic": None, "binarized": None}
    hb = [binarize(s) for s in human]
    jb = [binarize(s) for s in judge]
    return {
        "n": len(human),
        "linear": round(cohens_kappa(human, judge, labels=_SCORE_LABELS, weights="linear"), 3),
        "quadratic": round(
            cohens_kappa(human, judge, labels=_SCORE_LABELS, weights="quadratic"), 3
        ),
        "binarized": round(cohens_kappa(hb, jb, labels=[0, 1]), 3),
    }


# ---- I/O + run ------------------------------------------------------------


def _load_human_labels(path: Path = _LABELS_PATH) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[row["example_id"]] = {
                "edge": int(row["edge_human"]) if row["edge_human"] else None,
                "standalone": int(row["standalone_human"]) if row["standalone_human"] else None,
            }
    return out


def run_judge(clip: dict, judge_prompt: str) -> dict:
    resp = providers.chat_completion(
        _JUDGE_CANDIDATE_ID,
        build_messages(judge_prompt, clip),
        component="judge",
        temperature=0.0,
        max_tokens=256,
    )
    return parse_judge_scores(resp.raw_text)


def main(argv: list[str] | None = None) -> int:
    clips_doc = json.loads(_CLIPS_PATH.read_text(encoding="utf-8"))
    clips = clips_doc["clips"]
    strategy = clips_doc["chunking_strategy"]
    judge_prompt = _JUDGE_PATH.read_text(encoding="utf-8")
    prompt_sha = content_sha(judge_prompt)
    human = _load_human_labels()

    sys.stdout.write(f"boundary kappa: {len(clips)} {strategy} clips, judge sha {prompt_sha}\n")

    per_clip = []
    for clip in clips:
        scored = run_judge(clip, judge_prompt)
        per_clip.append({"example_id": clip["example_id"], "judge": scored})
        sys.stdout.write(
            f"  {clip['example_id']:16} judge edge={scored['edge_sensibility']} "
            f"standalone={scored['standalone']}\n"
        )

    # Pair human vs judge, per dimension, only where the human score exists.
    edge_h, edge_j, standalone_h, standalone_j = [], [], [], []
    for row in per_clip:
        h = human.get(row["example_id"], {})
        if h.get("edge") is not None:
            edge_h.append(h["edge"])
            edge_j.append(row["judge"]["edge_sensibility"])
        if h.get("standalone") is not None:
            standalone_h.append(h["standalone"])
            standalone_j.append(row["judge"]["standalone"])

    results = {
        "provisional": True,
        "chunking_strategy": strategy,
        "judge_candidate_id": _JUDGE_CANDIDATE_ID,
        "judge_prompt_sha": prompt_sha,
        "edge_kappa": kappa_set(edge_h, edge_j),
        "standalone_kappa": kappa_set(standalone_h, standalone_j),
        "per_clip": per_clip,
        "human_standalone_dist": {s: standalone_h.count(s) for s in sorted(set(standalone_h))},
    }
    _OUT_PATH.write_text(json.dumps(results, indent=2))

    sys.stdout.write("\n=== BOUNDARY KAPPA (PROVISIONAL, fixed_window) ===\n")
    for dim in ("edge_kappa", "standalone_kappa"):
        k = results[dim]
        sys.stdout.write(
            f"{dim:18} n={k['n']:>2}  linear={k['linear']}  "
            f"quadratic={k['quadratic']}  binarized={k['binarized']}\n"
        )
    sys.stdout.write(f"human standalone distribution: {results['human_standalone_dist']}\n")
    sys.stdout.write(f"wrote {_OUT_PATH}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
