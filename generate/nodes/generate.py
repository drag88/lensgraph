"""Generate node — produces the final answer JSON.

Calls ``providers.chat_completion`` with the resolved generator
candidate over the top-8 reranked chunks. The model must emit strict
JSON conforming to ``GenerationOutput`` (see ``generate/parser.py``).

No retry. Parse failures are real signal (design §6) and the Cite node
handles ``parse_ok=False`` by emitting an abstention.
"""

from __future__ import annotations

from eval.runners import providers
from generate.parser import GenerationOutput, parse_generation_output
from generate.state import AgentState

_MAX_TOKENS = 1024
# Qwen3-235B / DeepSeek V3.2 over top-8 chunks comfortably exceeds the
# 30s providers default. 90s is the live-EXIT-GATE-tested ceiling on a
# warm DeepInfra connection; bump if Cite starts seeing systematic
# ReadTimeout abstains. Slow but acceptable for batch eval; phase-4
# streaming is the right long-term answer.
_TIMEOUT_SEC = 90.0


def _format_chunks(chunks: list) -> str:
    parts: list[str] = []
    for i, c in enumerate(chunks):
        parts.append(f"[{i + 1}] [{c.video_id} @ {c.start_sec:.1f}-{c.end_sec:.1f}]\n{c.text}")
    return "\n\n".join(parts)


def _prompt(query: str, chunks: list) -> list[dict]:
    """Build the generation prompt. Strict-JSON instruction; cite by
    ``answer_claim_index`` into the model's own ``claims`` list (NOT
    into gold ``expected_claims`` — the generator never sees those).

    The shape mirrors design §6 GenerationOutput.
    """
    sys = (
        "You answer questions about engineering talks using ONLY the provided "
        "transcript chunks. Decompose your answer into atomic claims; cite "
        "each claim by index into your OWN claims list (answer_claim_index). "
        "Cite with [video_id, start_sec, end_sec] taken from the chunk labels. "
        "If the chunks do not contain the answer, set abstain=true and leave "
        "answer empty.\n\n"
        "Emit STRICT JSON only, no prose, no code fences:\n"
        '  {"answer": "<prose>", "claims": [{"text": "<claim>"}, ...], '
        '"citations": [{"video_id": "<vid>", "start_sec": <float>, '
        '"end_sec": <float>, "answer_claim_index": <int>}, ...], '
        '"abstain": <bool>}'
    )
    user = f"User query: {query!r}\n\nChunks:\n{_format_chunks(chunks)}\n\nAnswer now."
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]


def generate(state: AgentState) -> AgentState:
    """Generate the final JSON; set ``raw_answer``, ``parsed``,
    ``parse_ok``. ``conn`` is not threaded — generation is pure HTTP."""
    chunks = state.get("reranked") or []
    candidate = state["generator_candidate"]
    try:
        resp = providers.chat_completion(
            candidate_id=candidate.candidate_id,
            messages=_prompt(state["query"], chunks),
            component="generator",
            max_retries=1,
            max_tokens=_MAX_TOKENS,
            timeout_sec=_TIMEOUT_SEC,
        )
        raw = resp.raw_text
    except providers.ProviderError as e:
        # Provider failure → empty raw + parse_ok=False; Cite emits abstain.
        state["raw_answer"] = ""
        state["parsed"] = GenerationOutput(
            parse_ok=False,
            raw_response="",
            error=f"provider: {e}",
        )
        state["parse_ok"] = False
        return state

    parsed = parse_generation_output(raw)
    state["raw_answer"] = raw
    state["parsed"] = parsed
    state["parse_ok"] = parsed.parse_ok
    return state
