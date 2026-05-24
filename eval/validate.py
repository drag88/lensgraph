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
FIXTURES_DIR = EVAL_ROOT / "tests" / "fixtures"
PROJECT_ROOT = EVAL_ROOT.parent

GOLD_FILES = ["dev_gold.jsonl", "test_gold.jsonl", "negative.jsonl", "synthesis.jsonl"]


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


def validate_corpora(strict: bool = False) -> int:
    errors: list[str] = []
    warnings: list[str] = []

    talk_v = load_validator("talk.schema.json")
    gold_v = load_validator("gold_example.schema.json")
    audit_v = load_validator("boundary_audit.schema.json")

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
                video_id = talk.get("video_id")
                if not video_id:
                    continue
                talks_by_id[video_id] = talk

                # transcript drift check
                tpath_rel = talk.get("transcript_path", "")
                tpath = PROJECT_ROOT / tpath_rel if tpath_rel else None
                expected_sha = talk.get("transcript_sha256")
                if not tpath or not tpath.exists():
                    msg = (
                        f"{source}: transcript_path '{tpath_rel}' missing on disk"
                    )
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
            failures.append(
                f"{f.name}: expected VALID, got {len(errs)} error(s); first: {errs[0]}"
            )

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
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()

    rc_corpora = validate_corpora(strict=args.strict)
    rc_self = run_self_test()
    return rc_corpora or rc_self


if __name__ == "__main__":
    sys.exit(main())
