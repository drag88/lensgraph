"""Fast tests for the shared generation-output parser.

The parser is consumed by both ``eval/runners/minimal_generation.py`` and
``generate/nodes/generate.py`` (Teammate B). It must:

  - Always return a ``GenerationOutput`` (never ``None``); ``parse_ok``
    signals success.
  - Populate ``raw_response`` unconditionally, success or failure.
  - Salvage a fenced ```json ... ``` payload exactly once (LLMs love
    code fences), but otherwise NOT retry — parse failures are real
    bakeoff signal (design §6).
  - Leave cross-validation of ``citation.answer_claim_index`` against
    ``len(claims)`` to the Cite node — the parser only enforces the
    Pydantic field constraint (``ge=0``).
"""

from __future__ import annotations

import json

from generate.parser import GenerationOutput, parse_generation_output


def test_happy_path_well_formed_json():
    payload = {
        "answer": "Tengyu compares a library to RAG.",
        "claims": [
            {"text": "The library is the index."},
            {"text": "Retrieval is the lookup."},
        ],
        "citations": [
            {
                "video_id": "W_CYk2ogcDI",
                "start_sec": 168.0,
                "end_sec": 286.0,
                "answer_claim_index": 0,
            },
            {
                "video_id": "W_CYk2ogcDI",
                "start_sec": 200.0,
                "end_sec": 240.0,
                "answer_claim_index": 1,
            },
        ],
        "abstain": False,
    }
    raw = json.dumps(payload)
    out = parse_generation_output(raw)

    assert isinstance(out, GenerationOutput)
    assert out.parse_ok is True
    assert out.error is None
    assert out.raw_response == raw
    assert out.answer == "Tengyu compares a library to RAG."
    assert [c.text for c in out.claims] == [
        "The library is the index.",
        "Retrieval is the lookup.",
    ]
    assert len(out.citations) == 2
    assert out.citations[0].video_id == "W_CYk2ogcDI"
    assert out.citations[1].answer_claim_index == 1
    assert out.abstain is False


def test_malformed_json_sets_parse_ok_false():
    raw = "not json at all {{{"
    out = parse_generation_output(raw)

    assert isinstance(out, GenerationOutput)
    assert out.parse_ok is False
    assert out.raw_response == raw
    assert out.error is not None
    assert out.answer == ""
    assert out.claims == []
    assert out.citations == []


def test_fenced_json_payload_is_salvaged():
    """LLMs commonly emit ```json ... ``` fences. One de-fence pass is the
    only retry allowed by design step 28."""
    payload = {
        "answer": "Fenced answer.",
        "claims": [{"text": "Claim."}],
        "citations": [],
        "abstain": False,
    }
    raw = "```json\n" + json.dumps(payload) + "\n```"
    out = parse_generation_output(raw)

    assert out.parse_ok is True
    assert out.error is None
    assert out.raw_response == raw  # raw stays raw; salvage is internal
    assert out.answer == "Fenced answer."
    assert len(out.claims) == 1


def test_citation_index_zero_with_one_claim_is_valid():
    """Parser does NOT cross-validate index vs len(claims) — that's the
    Cite node's job. ``answer_claim_index >= 0`` is the only check here."""
    payload = {
        "answer": "x",
        "claims": [{"text": "only claim"}],
        "citations": [
            {
                "video_id": "v",
                "start_sec": 0.0,
                "end_sec": 1.0,
                "answer_claim_index": 0,
            }
        ],
        "abstain": False,
    }
    out = parse_generation_output(json.dumps(payload))
    assert out.parse_ok is True
    assert out.citations[0].answer_claim_index == 0


def test_abstain_payload():
    payload = {
        "answer": "",
        "claims": [],
        "citations": [],
        "abstain": True,
    }
    out = parse_generation_output(json.dumps(payload))
    assert out.parse_ok is True
    assert out.abstain is True
    assert out.answer == ""
    assert out.claims == []


def test_empty_string_input():
    out = parse_generation_output("")
    assert out.parse_ok is False
    assert out.raw_response == ""
    assert out.error is not None
