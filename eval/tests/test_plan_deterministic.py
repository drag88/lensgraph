"""Fast tests for the deterministic Plan path.

The deterministic path is the DEFAULT (planner_candidate is None). It
must classify + degenerate without touching a model or the network. The
"no LLM" invariant is asserted by monkeypatching
``eval.runners.providers.chat_completion`` to raise on any call —
deterministic tests must complete without invoking it.

Synthesis cues exercised:
  1. regex (``compare``, ``contrast``, ``both``, ``and X says``)
  2. 2+ distinct speaker mentions from ``talks.yaml``
  3. plain single_clip (control)

Sub-queries are always ``[query]`` in deterministic mode regardless of
question_type — the synthesis decomposition requires an LLM.
"""

from __future__ import annotations

from generate.nodes.plan import plan


def _state(query: str, corpus_id: str = "ai_engineering_v0") -> dict:
    return {"query": query, "corpus_id": corpus_id}


def test_single_clip_classification_for_plain_factual_query():
    s = plan(_state("How does Tengyu Ma define long context?"))
    assert s["question_type"] == "single_clip"
    assert s["sub_queries"] == ["How does Tengyu Ma define long context?"]


def test_synthesis_classification_on_compare_regex():
    s = plan(_state("Compare RAG, fine-tuning, and long context."))
    assert s["question_type"] == "synthesis"
    # sub_queries STILL [query] — deterministic path does not decompose.
    assert s["sub_queries"] == ["Compare RAG, fine-tuning, and long context."]


def test_synthesis_classification_on_contrast_regex():
    s = plan(_state("Contrast Gemma and DeepSeek on agentic eval."))
    assert s["question_type"] == "synthesis"


def test_synthesis_classification_on_two_speaker_mentions():
    # "Tengyu Ma" and "Shubam Sabu" both appear in the ai_engineering_v0
    # corpus talks.yaml. A query mentioning both speakers — phrased
    # so it does NOT also match the regex (no "compare", no "and X
    # say") — should still classify as synthesis via the
    # speaker-mention heuristic.
    q = "Tengyu vs Shubam on agent design philosophy"
    s = plan(_state(q))
    assert s["question_type"] == "synthesis"
    assert s["sub_queries"] == [q]


def test_single_speaker_mention_stays_single_clip():
    # One speaker → single_clip. Two would tip it to synthesis.
    q = "What does Tengyu say about RAG?"
    s = plan(_state(q))
    assert s["question_type"] == "single_clip"


def test_no_llm_call_on_deterministic_path(monkeypatch):
    """The deterministic path must not import or invoke the provider.

    We monkeypatch ``chat_completion`` to raise; if the plan node calls
    it (e.g. by accidentally falling into the LLM path with no
    planner_candidate) the test fails loudly."""
    from eval.runners import providers

    def boom(*a, **k):  # pragma: no cover — must never run
        raise AssertionError("deterministic plan must not call chat_completion")

    monkeypatch.setattr(providers, "chat_completion", boom)

    s = plan(_state("How does Tengyu Ma define long context?"))
    assert s["question_type"] == "single_clip"


def test_missing_talks_yaml_degrades_to_regex_only(tmp_path, monkeypatch):
    """If talks.yaml is unreadable, the speaker-mention heuristic
    silently degrades — only the regex remains. A query that mentions
    two speakers (but no regex cue) should now classify as single_clip."""
    from generate.nodes import plan as plan_mod

    # Point CORPORA_DIR at an empty temp dir to simulate missing yaml.
    monkeypatch.setattr(plan_mod, "_CORPORA_DIR", tmp_path)
    plan_mod._speakers_for_corpus.cache_clear()

    q = "Tengyu vs Shubam on agent design philosophy"
    s = plan(_state(q))
    assert s["question_type"] == "single_clip"
    # Re-clear so other tests that depend on the real cache get a fresh load.
    plan_mod._speakers_for_corpus.cache_clear()
