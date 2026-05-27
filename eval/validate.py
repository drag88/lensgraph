#!/usr/bin/env python3
"""Schema validator for LensGraph eval corpora.

Modes:
    python eval/validate.py              # validate schemas + corpora + self-test
    python eval/validate.py --self-test  # only run fixture self-test (fast iteration)
    python eval/validate.py --strict     # also fail on missing transcript files

The default run does BOTH corpora validation and the fixture self-test, so a
single command catches both data drift and schema regressions.

Notes:
- JSON Schema 2020-12 cannot express cross-field constraints (e.g. end_sec > start_sec).
  Those are checked in Python after schema validation.
- `format` keywords (date-time, uri) require a FormatChecker plus the
  jsonschema[format-nongpl] extra; both are wired in below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator, FormatChecker

EVAL_ROOT = Path(__file__).resolve().parent
SCHEMAS_DIR = EVAL_ROOT / "schemas"
CORPORA_DIR = EVAL_ROOT / "corpora"
CONFIG_DIR = EVAL_ROOT / "config"
FIXTURES_DIR = EVAL_ROOT / "tests" / "fixtures"
PROJECT_ROOT = EVAL_ROOT.parent

GOLD_FILES = [
    "dev_gold.jsonl",
    "test_gold.jsonl",
    "negative.jsonl",
    "synthesis.jsonl",
    # Visual eval gate — slide/screen_code/diagram/whiteboard gold examples.
    # Currently scaffolded as an empty committed file: visual retrieval needs
    # ingested MP4 frames + ColQwen patches AND verified visual gold examples
    # before the eval scores anything. Schema + verified:true rules apply
    # exactly as they do for the four text-eval files above, AND every entry
    # must carry at least one VISUAL_MODALITY_TAG.
    "visual_gold.jsonl",
]

# visual_gold.jsonl entries must include at least one of these modality tags
# (matches eval.runners.measure_visual.VISUAL_MODALITIES). Transcript-only or
# audio-only rows belong in dev_gold; the validator rejects them here so a
# misfiled entry cannot reach the visual eval.
VISUAL_MODALITY_TAGS = frozenset({"slide", "screen_code", "diagram", "whiteboard"})
VISUAL_GOLD_FILENAME = "visual_gold.jsonl"


def load_validator(name: str) -> Draft202012Validator:
    schema = json.loads((SCHEMAS_DIR / name).read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def iter_jsonl(path: Path):
    for i, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            yield i, json.loads(line)
        except json.JSONDecodeError as exc:
            yield i, exc


def count_verified_gold_and_talks() -> tuple[int, int]:
    """Count verified non-negative examples (single_clip + synthesis) and the distinct talks
    they reference. Negatives do not contribute to the phase-0 gate because they teach the
    system nothing about retrieval; 10 negatives could trivially unlock implementation work
    that has no signal-bearing examples to validate against.
    """
    n_examples = 0
    talk_ids: set[str] = set()
    if not CORPORA_DIR.exists():
        return 0, 0
    for corpus_dir in sorted(CORPORA_DIR.iterdir()):
        if not corpus_dir.is_dir():
            continue
        for jsonl_name in GOLD_FILES:
            f = corpus_dir / jsonl_name
            if not f.exists():
                continue
            for _, parsed in iter_jsonl(f):
                if not isinstance(parsed, dict):
                    continue
                if parsed.get("verified") is not True:
                    continue
                qt = parsed.get("question_type")
                if qt not in ("single_clip", "synthesis"):
                    continue
                n_examples += 1
                if qt == "single_clip" and parsed.get("video_id"):
                    talk_ids.add(parsed["video_id"])
                elif qt == "synthesis":
                    for span in parsed.get("gold_spans") or []:
                        if span.get("video_id"):
                            talk_ids.add(span["video_id"])
    return n_examples, len(talk_ids)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def check_span_ranges(spans, source: str, field: str = "gold_spans") -> list[str]:
    errs: list[str] = []
    for i, span in enumerate(spans or []):
        s, e = span.get("start_sec"), span.get("end_sec")
        if s is not None and e is not None and e <= s:
            errs.append(f"{source}: {field}[{i}] has end_sec ({e}) <= start_sec ({s})")
    return errs


def check_talk_source_range(talk: dict, source: str) -> list[str]:
    """Semantic checks for chapter-sliced talk records."""
    errs: list[str] = []
    s, e = talk.get("source_start_sec"), talk.get("source_end_sec")
    if s is not None and e is not None:
        if e <= s:
            errs.append(f"{source}: source_end_sec ({e}) <= source_start_sec ({s})")
        duration = talk.get("duration_sec")
        if isinstance(duration, int):
            expected = round(e - s)
            if abs(duration - expected) > 1:
                errs.append(
                    f"{source}: duration_sec ({duration}) does not match source slice "
                    f"length ({expected})"
                )
    return errs


def validate_example(ex: dict, gold_v: Draft202012Validator, source: str) -> list[str]:
    """Schema + cross-field semantic checks for a gold example."""
    errs = [f"{source}: {e.message}" for e in gold_v.iter_errors(ex)]
    errs.extend(check_span_ranges(ex.get("gold_spans"), source))
    return errs


def validate_boundary_record(ex: dict, audit_v: Draft202012Validator, source: str) -> list[str]:
    errs = [f"{source}: {e.message}" for e in audit_v.iter_errors(ex)]
    clip = ex.get("system_clip") or {}
    s, e = clip.get("start_sec"), clip.get("end_sec")
    if s is not None and e is not None and e <= s:
        errs.append(f"{source}: system_clip has end_sec ({e}) <= start_sec ({s})")
    return errs


def validate_model_candidates() -> list[str]:
    """Validate eval/config/model_candidates.yaml against its schema. Returns error messages."""
    errors: list[str] = []
    mc_path = CONFIG_DIR / "model_candidates.yaml"
    if not mc_path.exists():
        return [f"{mc_path}: model_candidates.yaml is missing (required by ADR 004)"]
    mc_v = load_validator("model_candidates.schema.json")
    try:
        data = yaml.safe_load(mc_path.read_text())
    except yaml.YAMLError as exc:
        return [f"{mc_path}: invalid YAML: {exc}"]
    if not isinstance(data, dict):
        return [f"{mc_path}: top-level must be a mapping, got {type(data).__name__}"]
    for err in mc_v.iter_errors(data):
        loc = "/".join(str(p) for p in err.absolute_path) or "<root>"
        errors.append(f"{mc_path}::{loc}: {err.message}")
    # Cross-check: inference_providers.primary must appear as a provider on at least one option.
    primary = data.get("inference_providers", {}).get("primary")
    if primary:
        providers_used = {
            opt.get("provider")
            for group in (data.get("candidates") or {}).values()
            if isinstance(group, dict)
            for opt in (group.get("options") or [])
            if isinstance(opt, dict)
        }
        if primary not in providers_used:
            errors.append(
                f"{mc_path}: inference_providers.primary='{primary}' is not used by any "
                f"candidate option (providers seen: {sorted(providers_used)})"
            )
    return errors


def validate_corpora(strict: bool = False) -> int:
    errors: list[str] = []
    warnings: list[str] = []

    talk_v = load_validator("talk.schema.json")
    gold_v = load_validator("gold_example.schema.json")
    audit_v = load_validator("boundary_audit.schema.json")

    # Validate the model candidates config (source of truth for the bakeoff runner).
    errors.extend(validate_model_candidates())

    if not CORPORA_DIR.exists():
        print(f"OK: no corpora directory at {CORPORA_DIR}")
        return 0

    corpus_dirs = [d for d in sorted(CORPORA_DIR.iterdir()) if d.is_dir()]
    if not corpus_dirs:
        print("OK: corpora directory is empty")
        return 0

    for corpus_dir in corpus_dirs:
        talks_by_id: dict[str, dict] = {}

        # talks.yaml
        talks_file = corpus_dir / "talks.yaml"
        if talks_file.exists():
            try:
                talks = yaml.safe_load(talks_file.read_text()) or []
            except yaml.YAMLError as exc:
                errors.append(f"{talks_file}: invalid YAML: {exc}")
                talks = []
            if not isinstance(talks, list):
                errors.append(
                    f"{talks_file}: top-level YAML must be a list, got {type(talks).__name__}"
                )
                talks = []
            for talk in talks:
                if not isinstance(talk, dict):
                    errors.append(
                        f"{talks_file}: each entry must be a mapping, got {type(talk).__name__}"
                    )
                    continue
                source = f"{talks_file}::{talk.get('video_id', '?')}"
                for err in talk_v.iter_errors(talk):
                    errors.append(f"{source}: {err.message}")
                errors.extend(check_talk_source_range(talk, source))
                video_id = talk.get("video_id")
                if not video_id:
                    continue
                talks_by_id[video_id] = talk

                # transcript drift check
                tpath_rel = talk.get("transcript_path", "")
                tpath = PROJECT_ROOT / tpath_rel if tpath_rel else None
                expected_sha = talk.get("transcript_sha256")
                if not tpath or not tpath.exists():
                    msg = f"{source}: transcript_path '{tpath_rel}' missing on disk"
                    (errors if strict else warnings).append(msg)
                else:
                    actual = sha256_file(tpath)
                    if expected_sha and actual != expected_sha:
                        errors.append(
                            f"{source}: transcript_sha256 mismatch "
                            f"(expected {expected_sha[:12]}..., got {actual[:12]}...)"
                        )

        # gold jsonl files
        for jsonl_name in GOLD_FILES:
            f = corpus_dir / jsonl_name
            if not f.exists():
                continue
            for line_no, parsed in iter_jsonl(f):
                source = f"{f}:{line_no}"
                if isinstance(parsed, json.JSONDecodeError):
                    errors.append(f"{source}: invalid JSON: {parsed}")
                    continue
                ex = parsed
                errors.extend(validate_example(ex, gold_v, source))

                # cross-reference: video_ids must exist in talks (no empty-talks bypass)
                qt = ex.get("question_type")
                referenced_vids: list[str] = []
                if qt == "synthesis":
                    referenced_vids = [
                        s.get("video_id") for s in ex.get("gold_spans") or [] if s.get("video_id")
                    ]
                elif ex.get("video_id"):
                    referenced_vids = [ex["video_id"]]
                for vid in referenced_vids:
                    if vid not in talks_by_id:
                        errors.append(f"{source}: references unknown video_id={vid}")

                # committed examples must be verified
                if not ex.get("verified", False):
                    errors.append(
                        f"{source}: unverified example committed (id={ex.get('id', '?')}). "
                        "Flip verified:true after watching the actual clip."
                    )

                # visual_gold.jsonl entries must carry at least one visual
                # modality tag — otherwise they cannot meaningfully be
                # scored by the visual retrieval channel.
                if jsonl_name == VISUAL_GOLD_FILENAME:
                    modality = set(ex.get("modality") or [])
                    if not (modality & VISUAL_MODALITY_TAGS):
                        errors.append(
                            f"{source}: visual_gold entry (id={ex.get('id', '?')}) "
                            f"modality={sorted(modality)} lacks any of "
                            f"{sorted(VISUAL_MODALITY_TAGS)} — visual_gold is for "
                            "visual-bearing examples only; transcript-only "
                            "examples belong in dev_gold."
                        )

        # boundary audit
        audit_file = corpus_dir / "boundary_audit.jsonl"
        if audit_file.exists():
            for line_no, parsed in iter_jsonl(audit_file):
                source = f"{audit_file}:{line_no}"
                if isinstance(parsed, json.JSONDecodeError):
                    errors.append(f"{source}: invalid JSON: {parsed}")
                    continue
                errors.extend(validate_boundary_record(parsed, audit_v, source))

    for w in warnings:
        print(f"warn: {w}")

    if errors:
        print(f"\nFAIL: {len(errors)} validation error(s)")
        for err in errors[:50]:
            print(f"  {err}")
        if len(errors) > 50:
            print(f"  ... {len(errors) - 50} more")
        return 1

    print(
        f"OK: corpora valid ({len(corpus_dirs)} corpus dir(s), "
        f"{len(warnings)} warning(s), strict={strict})"
    )
    return 0


def run_self_test() -> int:
    gold_v = load_validator("gold_example.schema.json")
    failures: list[str] = []

    if not FIXTURES_DIR.exists():
        print(f"FAIL: fixtures directory missing at {FIXTURES_DIR}")
        return 1

    valid_files = sorted(FIXTURES_DIR.glob("valid_*.json"))
    invalid_files = sorted(FIXTURES_DIR.glob("invalid_*.json"))
    other_files = [
        f
        for f in FIXTURES_DIR.glob("*.json")
        if not (f.name.startswith("valid_") or f.name.startswith("invalid_"))
    ]

    for f in other_files:
        failures.append(f"{f.name}: must be named valid_*.json or invalid_*.json")

    for f in valid_files:
        try:
            ex = json.loads(f.read_text())
        except json.JSONDecodeError as exc:
            failures.append(f"{f.name}: not valid JSON: {exc}")
            continue
        errs = validate_example(ex, gold_v, f.name)
        if errs:
            failures.append(f"{f.name}: expected VALID, got {len(errs)} error(s); first: {errs[0]}")

    for f in invalid_files:
        try:
            ex = json.loads(f.read_text())
        except json.JSONDecodeError as exc:
            failures.append(f"{f.name}: not valid JSON: {exc}")
            continue
        errs = validate_example(ex, gold_v, f.name)
        if not errs:
            failures.append(f"{f.name}: expected INVALID, but validator accepted it")

    if failures:
        print(f"SELF-TEST FAIL: {len(failures)} failure(s)")
        for fail in failures:
            print(f"  {fail}")
        return 1

    print(
        f"SELF-TEST OK: {len(valid_files)} valid + {len(invalid_files)} invalid "
        f"fixtures all behaved as expected"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Only run the fixture self-test (fast iteration during schema work).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing transcript files (commit gate). Default is warn-only for CI.",
    )
    parser.add_argument(
        "--min-gold",
        type=int,
        default=0,
        help=(
            "Phase-0 gate: fail unless the corpus contains at least N verified non-negative "
            "examples (single_clip + synthesis). Negatives are excluded — they cannot unlock "
            "implementation work because they have no signal-bearing answer. Use 0 to skip."
        ),
    )
    parser.add_argument(
        "--min-talks",
        type=int,
        default=0,
        help=(
            "Phase-0 gate companion: fail unless the verified non-negative examples reference "
            "at least N distinct talks. Prevents 10-examples-on-one-talk false signals. "
            "Use 0 to skip."
        ),
    )
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()

    rc_corpora = validate_corpora(strict=args.strict)
    rc_self = run_self_test()
    rc_gate = 0
    if args.min_gold > 0 or args.min_talks > 0:
        n_gold, n_talks = count_verified_gold_and_talks()
        gold_ok = args.min_gold == 0 or n_gold >= args.min_gold
        talks_ok = args.min_talks == 0 or n_talks >= args.min_talks
        if not (gold_ok and talks_ok):
            print(
                f"\nPHASE-0 GATE FAIL: {n_gold} verified non-negative examples "
                f"(need >= {args.min_gold}) across {n_talks} distinct talks "
                f"(need >= {args.min_talks}). "
                "Implementation code (ingest/, chunking/, retrieve/, generate/, api/, web/) "
                "must not be modified until this gate passes. See CLAUDE.md hard rule 1."
            )
            rc_gate = 1
        else:
            print(
                f"PHASE-0 GATE OK: {n_gold} verified non-negative examples across "
                f"{n_talks} distinct talks (>= {args.min_gold}/{args.min_talks})"
            )
    return rc_corpora or rc_self or rc_gate


if __name__ == "__main__":
    sys.exit(main())
