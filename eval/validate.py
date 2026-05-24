#!/usr/bin/env python3
"""Schema validator for LensGraph eval corpora.

Two modes:

    python eval/validate.py              # validate schemas + corpora (CI use)
    python eval/validate.py --self-test  # run fixture tests proving schema rules

In --self-test mode, every file in eval/tests/fixtures/ named `valid_*.json`
must pass schema validation, and every `invalid_*.json` must fail. This is
how we prove the conditional rules (synthesis must have per-span video_id,
corpus negatives must not have video_id, etc.) actually work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

EVAL_ROOT = Path(__file__).resolve().parent
SCHEMAS_DIR = EVAL_ROOT / "schemas"
CORPORA_DIR = EVAL_ROOT / "corpora"
FIXTURES_DIR = EVAL_ROOT / "tests" / "fixtures"
TRANSCRIPTS_DIR = EVAL_ROOT.parent / "transcripts"

GOLD_FILES = ["dev_gold.jsonl", "test_gold.jsonl", "negative.jsonl", "synthesis.jsonl"]


def load_validator(name: str) -> Draft202012Validator:
    schema = json.loads((SCHEMAS_DIR / name).read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


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


def validate_corpora() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    talk_v = load_validator("talk.schema.json")
    gold_v = load_validator("gold_example.schema.json")
    audit_v = load_validator("boundary_audit.schema.json")

    if not CORPORA_DIR.exists():
        print(f"OK: no corpora directory at {CORPORA_DIR} (nothing to validate)")
        return 0

    corpus_dirs = [d for d in sorted(CORPORA_DIR.iterdir()) if d.is_dir()]
    if not corpus_dirs:
        print("OK: corpora directory is empty (nothing to validate)")
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
                errors.append(f"{talks_file}: top-level YAML must be a list, got {type(talks).__name__}")
                talks = []
            for talk in talks:
                if not isinstance(talk, dict):
                    errors.append(f"{talks_file}: each entry must be a mapping, got {type(talk).__name__}")
                    continue
                for err in talk_v.iter_errors(talk):
                    errors.append(f"{talks_file}::{talk.get('video_id', '?')}: {err.message}")
                video_id = talk.get("video_id")
                if video_id:
                    talks_by_id[video_id] = talk
                    # transcript drift check (warn-only if file missing in v0)
                    tpath = EVAL_ROOT.parent / talk.get("transcript_path", "")
                    if not tpath.exists():
                        warnings.append(
                            f"{talks_file}::{video_id}: transcript_path '{talk.get('transcript_path')}' missing on disk"
                        )
                    else:
                        actual = sha256_file(tpath)
                        if actual != talk.get("transcript_sha256"):
                            errors.append(
                                f"{talks_file}::{video_id}: transcript_sha256 mismatch (expected {talk.get('transcript_sha256')[:12]}..., got {actual[:12]}...)"
                            )

        # gold jsonl files
        for jsonl_name in GOLD_FILES:
            f = corpus_dir / jsonl_name
            if not f.exists():
                continue
            for line_no, parsed in iter_jsonl(f):
                if isinstance(parsed, json.JSONDecodeError):
                    errors.append(f"{f}:{line_no}: invalid JSON: {parsed}")
                    continue
                ex = parsed
                for err in gold_v.iter_errors(ex):
                    errors.append(f"{f}:{line_no}: {err.message}")
                # cross-reference: video_ids must exist in talks
                qt = ex.get("question_type")
                if qt == "synthesis":
                    for span in ex.get("gold_spans", []):
                        vid = span.get("video_id")
                        if vid and talks_by_id and vid not in talks_by_id:
                            errors.append(f"{f}:{line_no}: span references unknown video_id={vid}")
                else:
                    vid = ex.get("video_id")
                    if vid and talks_by_id and vid not in talks_by_id:
                        errors.append(f"{f}:{line_no}: references unknown video_id={vid}")
                # committed examples must be verified
                if not ex.get("verified", False):
                    errors.append(
                        f"{f}:{line_no}: unverified example committed (id={ex.get('id', '?')}). "
                        "Flip verified:true after watching the actual clip."
                    )

        # boundary audit
        audit_file = corpus_dir / "boundary_audit.jsonl"
        if audit_file.exists():
            for line_no, parsed in iter_jsonl(audit_file):
                if isinstance(parsed, json.JSONDecodeError):
                    errors.append(f"{audit_file}:{line_no}: invalid JSON: {parsed}")
                    continue
                for err in audit_v.iter_errors(parsed):
                    errors.append(f"{audit_file}:{line_no}: {err.message}")

    for w in warnings:
        print(f"warn: {w}")

    if errors:
        print(f"\nFAIL: {len(errors)} validation error(s)")
        for err in errors[:50]:
            print(f"  {err}")
        if len(errors) > 50:
            print(f"  ... {len(errors) - 50} more")
        return 1

    print(f"OK: corpora valid ({len(corpus_dirs)} corpus dir(s), {len(warnings)} warning(s))")
    return 0


def run_self_test() -> int:
    gold_v = load_validator("gold_example.schema.json")
    failures: list[str] = []

    if not FIXTURES_DIR.exists():
        print(f"FAIL: fixtures directory missing at {FIXTURES_DIR}")
        return 1

    valid_files = sorted(FIXTURES_DIR.glob("valid_*.json"))
    invalid_files = sorted(FIXTURES_DIR.glob("invalid_*.json"))
    other_files = [f for f in FIXTURES_DIR.glob("*.json") if not (f.name.startswith("valid_") or f.name.startswith("invalid_"))]

    for f in other_files:
        failures.append(f"{f.name}: fixture must be named valid_*.json or invalid_*.json")

    for f in valid_files:
        try:
            ex = json.loads(f.read_text())
        except json.JSONDecodeError as exc:
            failures.append(f"{f.name}: fixture is not valid JSON: {exc}")
            continue
        errs = list(gold_v.iter_errors(ex))
        if errs:
            failures.append(
                f"{f.name}: expected VALID, got {len(errs)} error(s); first: {errs[0].message}"
            )

    for f in invalid_files:
        try:
            ex = json.loads(f.read_text())
        except json.JSONDecodeError as exc:
            failures.append(f"{f.name}: fixture is not valid JSON: {exc}")
            continue
        errs = list(gold_v.iter_errors(ex))
        if not errs:
            failures.append(f"{f.name}: expected INVALID, but schema accepted it")

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
        help="Run fixture tests proving schema conditionals work",
    )
    args = parser.parse_args()
    if args.self_test:
        return run_self_test()
    return validate_corpora()


if __name__ == "__main__":
    sys.exit(main())
