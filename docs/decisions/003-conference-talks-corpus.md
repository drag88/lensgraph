# ADR 003 — Engineering conference talks as the launch corpus

**Status:** Accepted
**Date:** 2026-05-24

## Context

LensGraph needs a launch corpus that is simultaneously interesting to demo, varied in modality, manageable to curate, and recruiter-legible. The corpus choice is a marketing decision as much as a technical one.

## Options

**Earnings calls.** Abundant, regular cadence, named entities. But: visually uniform (talking head, no slides), niche demo appeal, recruiter audience does not care.

**Congressional hearings / government proceedings.** Free, ground-truth-rich (official transcripts), high stakes. But: politically charged, partisan demo questions, modality is mostly transcript.

**University lectures (Stanford CS25, MIT 6.S191).** Excellent multimodal content (slides + chalkboard + screen). Recruiter-relevant for ML roles. Long-form. Best academic fit but pace can be uneven.

**Conference talks (AI Engineer Summit, NeurIPS, Strange Loop, PyData, LangChain Interrupt).** Multimodal (slides + screen-shared code + occasional whiteboard). Speakers are often recruiter-recognized names. Topical fit to the target audience. Lengths range 15–60 min — useful for chunking ablation.

**Podcasts with video (Latent Space, AI Engineer, Dwarkesh).** Conversational, low visual signal, tests transcript-only retrieval. Good as a stress-test slice but bad as the primary corpus.

## Decision

AI engineering conference talks as the primary corpus. Target distribution for v0:

- 4 slides-heavy (e.g., AI Engineer Summit keynotes)
- 3 code/screen-heavy (e.g., LangChain Interrupt, Modal Labs talks)
- 2 whiteboard / live-demo (e.g., Karpathy reproductions)
- 2 panel / conversational (tests transcript-only behavior)
- 1 lightning (15 min — checks short-form chunking)

## Reasoning

1. **Recruiter alignment.** The hiring audience watches these talks. Demos using familiar speakers and recent topics earn instant credibility.
2. **Modality richness.** Slides + screen code + whiteboard stresses every retrieval channel. ColPali earns its keep here, not on earnings calls.
3. **Copyright posture.** Public YouTube uploads under standard license; transcripts stored locally for research use, not redistributed. Safer than paywalled material.
4. **Demo legibility.** A 60-second demo with a Karpathy clip is self-explanatory. A 60-second demo with a Vietnamese textile supplier customs filing is not.
5. **Ground-truth feasibility.** YouTube auto-captions are usable as a baseline; high-value talks have community transcripts; WhisperX fills gaps.

## Consequences

- **Pro:** every constraint above satisfied.
- **Pro:** the corpus is naturally extensible — every month produces new talks worth ingesting.
- **Con:** the project lacks the "I built this for enterprise X" credibility of a supply-chain or compliance build. Mitigated by the eval rigor: the methodology is the enterprise-credible thing.
- **Con:** speaker name recognition skews the audience. Someone outside AI/ML may not appreciate the choices. Acceptable — the hiring target IS AI/ML.

## Revisit when

- A specific paid pivot emerges where a different corpus would be more credible.
- A v2 expansion adds enterprise meeting recordings, at which point the corpus split becomes "public talks + private meetings."
