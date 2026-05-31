"""Shared parser for model-emitted generation output.

Two callers depend on this module:
  * ``eval/runners/minimal_generation.py`` — phase-2 generator bakeoff
    harness (one provider call → one ``eval_runs`` row).
  * ``generate/nodes/generate.py`` — the LangGraph product path.

Both must measure the same thing. A shared parser prevents drift on JSON
parse semantics between the bakeoff and the LangGraph comparison.

The model emits a JSON object with keys ``{answer, claims, citations,
abstain}``. The parser wraps that payload in ``GenerationOutput`` and
adds three observability fields the *model* never sets: ``parse_ok``,
``raw_response``, ``error``.

Salvage policy: exactly one retry. If raw JSON parse fails, strip a
single `````json ... ````` code fence (LLMs love
fenced output) and try again. Anything else is signal — design §6
"Parse failures are real bakeoff signal. No retry."

The parser deliberately does NOT cross-validate ``citation.answer_claim_index``
against ``len(claims)``. The Cite node owns that bounds check because it
also reconciles citations against retrieved chunks; doing partial work
here would split the contract.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, ConfigDict, Field, ValidationError

# Match a ```json ... ``` fence (case-insensitive on the json hint, optional
# leading/trailing whitespace, single-line or multi-line). Captures the
# fenced body. We only de-fence once — see module docstring.
_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*\n?(?P<body>.*?)\n?```\s*$",
    re.DOTALL | re.IGNORECASE,
)


class AnswerClaim(BaseModel):
    """One atomic claim in the model's own decomposition of its answer."""

    model_config = ConfigDict(extra="ignore")

    text: str = Field(min_length=1)


class Citation(BaseModel):
    """One citation. ``answer_claim_index`` indexes into the model's own
    ``claims`` list (NOT the gold ``expected_claims``, which the model
    never sees — design §6 anti-contamination invariant)."""

    model_config = ConfigDict(extra="ignore")

    video_id: str
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)
    answer_claim_index: int = Field(ge=0)


class GenerationOutput(BaseModel):
    """Parser output. Always returned (never ``None``).

    Fields ``parse_ok``, ``raw_response``, ``error`` are populated by the
    parser, NOT by the model. The model emits ``{answer, claims,
    citations, abstain}`` only; the parser wraps and reports."""

    model_config = ConfigDict(extra="ignore")

    answer: str = ""
    claims: list[AnswerClaim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    abstain: bool = False
    parse_ok: bool = True
    raw_response: str = ""
    error: str | None = None


def _try_json_loads(s: str) -> tuple[object | None, str | None]:
    try:
        return json.loads(s), None
    except (json.JSONDecodeError, ValueError) as e:
        return None, f"json decode: {e}"


def _strip_fence(raw: str) -> str | None:
    """Return the fenced body if ``raw`` matches a ```json ... ``` fence,
    else ``None`` (no salvage attempt warranted)."""
    m = _FENCE_RE.match(raw)
    if not m:
        return None
    body = m.group("body").strip()
    return body or None


def parse_generation_output(raw_str: str) -> GenerationOutput:
    """Parse a model-emitted JSON string into ``GenerationOutput``.

    Strategy:
      1. Try ``json.loads`` on the raw string.
      2. On JSON failure, attempt a single salvage pass: strip a
         `````json ... ````` fence and try again.
      3. On parse success, ``GenerationOutput.model_validate`` the parsed
         payload; any Pydantic error short-circuits to failure.
      4. ``raw_response`` is set unconditionally; ``error`` is ``None`` on
         success and a short string on failure.
    """
    parsed, err = _try_json_loads(raw_str)
    if parsed is None:
        fenced_body = _strip_fence(raw_str)
        if fenced_body is not None:
            parsed, err = _try_json_loads(fenced_body)

    if parsed is None:
        return GenerationOutput(
            parse_ok=False,
            raw_response=raw_str,
            error=err or "json decode: empty input",
        )

    if not isinstance(parsed, dict):
        return GenerationOutput(
            parse_ok=False,
            raw_response=raw_str,
            error=f"expected object, got {type(parsed).__name__}",
        )

    try:
        model = GenerationOutput.model_validate(parsed)
    except ValidationError as e:
        return GenerationOutput(
            parse_ok=False,
            raw_response=raw_str,
            error=f"schema: {e.errors()[0]['msg'] if e.errors() else str(e)}",
        )

    # model_validate may have copied default parse_ok/raw_response from the
    # payload if the model emitted them. Overwrite — these are parser-owned.
    return model.model_copy(update={"parse_ok": True, "raw_response": raw_str, "error": None})
