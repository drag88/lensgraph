# Curation Playbook — `ai_engineering_v0`

Target: ~80 verified examples across 10–12 talks in 5 working days.

## Day-by-day budget

| Day | Output |
|---|---|
| 1 | 10–12 talks selected, `talks.yaml` complete (transcripts pulled + hashed), license confirmed. |
| 2 | 4 talks curated end-to-end (~25–30 examples). Schema bumped against reality; any missing enum value added. |
| 3 | 4 more talks curated (~25–30 examples). |
| 4 | Remaining talks; negative and synthesis examples crafted. |
| 5 | Boundary audit pass (20–30 examples manually scored); `test_gold.jsonl` locked and hash committed to README. |

## Talk selection criteria, in priority order

1. **Format diversity** to stress chunking: 4 slides-heavy, 3 code/screen, 2 whiteboard/live-demo, 2 conversational, 1 lightning.
2. **Length variety:** mix 15–20 min lightning, 35–50 min standard, 60+ min keynote.
3. **Speaker name recognition** when possible (helps demo legibility).
4. **Clean audio.** Bad-mic talks add ASR noise that pollutes downstream eval. Defer them to a v1 robustness slice.
5. **Permissive use posture.** Public YouTube uploads under standard license. Store transcripts locally; do not redistribute.

### Starter shortlist (replace as you watch and judge)

- Andrej Karpathy — *Intro to LLMs* (1h, slides-heavy)
- Jerry Liu — recent LlamaIndex agents talk (~45m, code+slides)
- Hamel Husain — *Your AI Product Needs Evals* (~45m, slides-heavy, on-topic)
- A Latent Space pod episode (~60m, conversational, transcript-only stress)
- Jason Liu — Instructor / pydantic for LLMs (~30m, code-heavy)
- An AI Engineer Summit lightning talk (~15m)
- Chip Huyen — recent talk (~45m, slides+narrative)
- A Stanford CS25 lecture (~60m, academic, whiteboard + slides)
- A Modal Labs or Replicate engineering talk (~30m, code-heavy)
- A panel from any conference (tests multi-speaker handling)
- NeurIPS keynote (~60m, dense)
- One outlier — pick something off-pattern

## Per-talk workflow (~45 min/talk)

1. **Pull the video and transcript.**
   ```bash
   yt-dlp --write-auto-subs --skip-download --sub-format vtt --sub-lang en \
     -o "transcripts/%(id)s.%(ext)s" "<youtube_url>"
   sha256sum transcripts/<video_id>.en.vtt
   ```
   Use WhisperX only if YouTube captions are visibly poor. Velocity matters more than ASR perfection in v0.

2. **Add the talk to `talks.yaml`** with full provenance:
   ```yaml
   - video_id: zjkBMFhNj_g
     title: "Intro to Large Language Models"
     speaker: "Andrej Karpathy"
     url: "https://www.youtube.com/watch?v=zjkBMFhNj_g"
     duration_sec: 3597
     format_tags: [slides_heavy, narrative]
     license: youtube_standard
     captions_source: youtube_auto
     transcript_path: "transcripts/zjkBMFhNj_g.en.vtt"
     transcript_sha256: "<paste from sha256sum>"
     accessed_at: "2026-05-24T10:00:00Z"
     notes: "Clean audio, clear slides — calibration talk."
   ```

3. **Skim the talk at 1.75x** with the transcript open. Note 8–12 candidate moments where the speaker says something concrete and searchable.

4. **Draft candidate questions with an LLM** (Claude or GPT-4 — see anti-contamination rules). Prompt scaffold:
   > Given this transcript segment and timestamp range, draft 1–2 questions a viewer would naturally ask whose answer lives precisely in this span. Each question must have 2–4 atomic claims that fully constitute a correct answer. Reject questions answerable from general knowledge or the talk title alone.

5. **Verify every triplet by watching the actual clip.** If the span needs to grow or shrink by more than 10 seconds, fix it. If the answer is vague, rewrite. This step cannot be automated and always takes longer than you expect.

6. **Tag** `modality`, `difficulty`, `question_type`, and add a couple of `tags`.

7. **Assign split** by stratified hash on `(modality, difficulty)`. Target: dev 35, test 15, negative 20, synthesis 10.

8. **Flip `verified: true`** only after you have watched the clip. The validator rejects unverified examples committed to corpus files.

## Anti-contamination rules

- **Curate with one model family, judge with another.** If you draft candidates with Claude, the LLM judge for faithfulness must be OpenAI-class, and vice versa.
- **Never expose retrieval output to curation tooling.** The curator only sees source transcripts.
- **Lock the test set** after curation. Commit the SHA256 of `test_gold.jsonl` to the project README. Any future change invalidates published metrics.

## Quality bar — calibration examples

**Good:**
> "What error rate does Karpathy report for the GPT-2 reproduction at the smallest model size?"

Single concrete answer; defensible span; fails gracefully if the system retrieves the wrong scale.

**Good:**
> "Why does Hamel argue vibes-based eval is dangerous for RAG systems specifically?"

Reasoning question, but the justification lives in a specific 60–90s span and decomposes into 2–3 atomic claims.

**Bad:**
> "What does the speaker discuss?" — no measurable answer.

**Bad:**
> "What is retrieval-augmented generation?" — answerable without the video.

**Bad:**
> "At 12:34, what does the speaker say?" — tests transcript lookup, not retrieval.

## Crafting negative examples

A negative is an honest question your system should refuse to answer because the corpus genuinely does not contain it.

- **`negative_scope: corpus`** (primary): the answer is nowhere in the corpus. These drive the main abstention metric.
- **`negative_scope: video`** (secondary): this specific video does not contain it, but the corpus might. Tests single-video reasoning.

A good negative looks like a real question. Avoid obvious adversarial bait ("What is the capital of Mars?") — those teach the model nothing useful.

## Crafting synthesis examples

A synthesis question requires combining content from two or more talks. Every `gold_span` must include its own `video_id`. Top-level `video_id` is forbidden (schema-enforced).

Example shapes:
- "Two speakers disagree about X — what is each speaker's position?"
- "Speaker A introduces a concept; Speaker B critiques it. What is the critique?"
- "Both Karpathy and Howard reference the same paper — what point does each make about it?"
