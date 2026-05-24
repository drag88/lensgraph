# LensGraph `ai_engineering_v0` Talk Shortlist

Generated: 2026-05-24

This shortlist is a curation input, not the final corpus. A talk becomes corpus material only after the transcript is pulled, hashed, watched, and at least one verified example is committed.

Selection bias:

- Prefer 2025-2026 talks with direct public video URLs.
- Prefer talks that are useful to AI engineering and applied ML researchers, not just recruiter-recognizable.
- Prefer conference/workshop/tutorial recordings over product demos.
- Require English VTT captions for v0 unless the talk is unusually valuable.
- Keep at least half of v0's verified examples on 2026 material when public captions are available.

Verification performed:

- `yt-dlp` metadata lookup for direct YouTube candidates.
- English subtitle probe with `yt-dlp --write-auto-subs --sub-langs en --simulate`.
- AI Engineer Singapore 2026 livestreams returned chapter metadata and English `vtt` subtitles.
- All recommended direct-video and livestream-chapter candidates below have an English subtitle path.

## Selection Rule

The original shortlist was too 2025-heavy. The correction is **not** "use AI Engineer Singapore because it was suggested"; the correction is:

1. Prefer 2026 talks when they are technically dense and public.
2. Prefer directly ingestible videos with captions over better-branded pages that require brittle scraping.
3. Keep a few 2025 anchors only when they are still the strongest compact source for a core LensGraph capability.
4. Avoid source monoculture: no conference, vendor, or channel should dominate the corpus unless the alternatives fail ingestion or quality checks.

AI Engineer Singapore 2026 is useful because it has:

- public livestream recordings from 2026-05-16 and 2026-05-17,
- rich YouTube chapters,
- English VTT subtitles,
- topics aligned with LensGraph: agent harnesses, coding agents, evals, document parsing, sandboxing, voice, inference, and robotics.

Do **not** ingest the 9-10 hour livestreams as single talks. Treat selected chapters as pseudo-talks in `talks.yaml` with stable IDs like `aie_sg_2026_d2_arize_alyx`, URL timestamps, `source_video_id`, `source_start_sec`, and `source_end_sec`. This preserves the "distinct talk" semantics used by the phase-0 gate while keeping manual review bounded.

Duration policy for v0:

- Recommended items should be 10–60 minutes.
- Chapter-sliced livestream items should usually be 10–25 minutes.
- Anything over 60 minutes moves to alternates unless it is the single deliberate long-form stress test.
- The first curation pass should avoid all >60 minute items.

## Recommended v0 Set

| Priority | Video ID | Talk | Speaker / Source | Date | Duration | Format tags | Why it belongs |
|---:|---|---|---|---|---:|---|---|
| 1 | `aie_sg_2026_d2_arize_alyx` | [Alyx planning states, large JSON abstractions, and reliable agent checkpoints](https://www.youtube.com/watch?v=m12vGjfbNlo&t=516s) | SallyAnn DeLucia, Arize AI / AI Engineer Singapore | 2026-05-17 | 16:18 | slides_heavy, narrative | Most directly aligned with LensGraph's eval-first product story: agent states, checkpoints, and reliability. |
| 2 | `nXafozNIk3c` | [Google's Agents CLI is here: Build, eval, and deploy AI agents in minutes](https://www.youtube.com/watch?v=nXafozNIk3c) | Google Cloud Tech | 2026-04-27 | 52:59 | code_heavy, live_demo | Non-AI-Engineer 2026 source with build/eval/deploy coverage and enough length for chunking tests. |
| 3 | `OnlN-2Q5QsE` | [AI with Zero Trust Security](https://www.youtube.com/watch?v=OnlN-2Q5QsE) | Microsoft Mechanics | 2026-02-17 | 10:56 | slides_heavy, live_demo | Adds security and least-privilege agent constraints; useful negative/abstention examples. |
| 4 | `aie_sg_2026_d2_gdm_boundaries` | [Applied AI at scale with deterministic boundaries around non deterministic models](https://www.youtube.com/watch?v=m12vGjfbNlo&t=5562s) | JJ Geewax, Google DeepMind / AI Engineer Singapore | 2026-05-17 | 20:51 | slides_heavy, narrative | Production-scale boundary thinking from a research lab; high relevance beyond the conference brand. |
| 5 | `aie_sg_2026_d1_openai_codex` | [Codex across the software lifecycle, agent reviews, and approval fatigue](https://www.youtube.com/watch?v=_xQnSNlBP_w&t=5107s) | Thibault Sottiaux, OpenAI / AI Engineer Singapore | 2026-05-16 | 23:14 | slides_heavy, live_demo | Current coding-agent material; useful for questions about review loops and human approval. |
| 6 | `aie_sg_2026_d2_llamaparse_failure_modes` | [LlamaParse failure modes, whitespace loops, and parsing at internet scale](https://www.youtube.com/watch?v=m12vGjfbNlo&t=12350s) | Pierre-Loic Doulcet, LlamaIndex / AI Engineer Singapore | 2026-05-17 | 12:19 | slides_heavy, live_demo | Directly relevant to LensGraph's visual document retrieval and transcript/layout parsing failure modes. |
| 7 | `96G7FLab8xc` | [Your MCP Server is Bad](https://www.youtube.com/watch?v=96G7FLab8xc) | Jeremiah Lowin, Prefect / AI Engineer | 2026-01-12 | 54:32 | code_heavy, live_demo | Strong long-form tool-protocol stress test; keep because it is recent and technically dense, not because of the event brand. |
| 8 | `d5EltXhbcfA` | [Building and evaluating AI Agents](https://www.youtube.com/watch?v=d5EltXhbcfA) | Sayash Kapoor, AI Engineer | 2025-04-17 | 19:59 | slides_heavy, narrative | Keep as a high-quality eval anchor even though it is 2025. |
| 9 | `GL0XhAj5LPE` | [LLM Evals: Common Mistakes](https://www.youtube.com/watch?v=GL0XhAj5LPE) | Hamel Husain | 2025-05-06 | 28:15 | slides_heavy, narrative | Keep for methodology and judge failure modes; likely high question yield. |
| 10 | `W_CYk2ogcDI` | [RAG in 2025: State of the Art and the Road Forward](https://www.youtube.com/watch?v=W_CYk2ogcDI) | Tengyu Ma, AI Engineer | 2025-06-27 | 18:48 | slides_heavy, narrative | Still the cleanest compact RAG/retrieval baseline found. |
| 11 | `V_xlQWLUWdQ` | [Visual Document Retrieval: Enhancing Accuracy with Text and Visual Embeddings](https://www.youtube.com/watch?v=V_xlQWLUWdQ) | Mark Hamazaspyan, PyData | 2025-04-03 | 48:59 | slides_heavy, live_demo | Keep for visual retrieval depth; pair with the LlamaParse chapter for doc-processing coverage. |
| 12 | `flf_IKnFYnE` | [From Stateless Nightmares to Durable Agents](https://www.youtube.com/watch?v=flf_IKnFYnE) | Samuel Colvin, Pydantic / AI Engineer | 2025-11-24 | 22:12 | slides_heavy, code_heavy | Shorter production-agent anchor covering state, durability, and workflow reliability. |

Recommended v0 mix:

- 2026 material: 7 talks / chapter-sliced talks
- 2025 anchor material: 5 talks
- Non-AI-Engineer sources: Google Cloud, Microsoft Mechanics, PyData, Hamel's independent channel
- Evals/reliability: 3 talks
- RAG/retrieval: 2 talks
- Visual/document retrieval and parsing: 2 talks
- Agent design and tool use: 5 talks
- Security / execution boundaries: 2 talks

This is intentionally current-heavy without making AI Engineer Singapore the center of gravity. The cost is less conversational variety in v0; add a podcast/panel later only if the first 20 examples show the corpus is too polished-slide-heavy.

## Bench Alternates

Use these if any recommended talk has weak captions, poor audio, duplicated content, or low question yield during manual review.

| Video ID | Talk | Source | Duration | Replace when |
|---|---|---|---:|---|
| `aie_sg_2026_d2_ibm_agent_harness` | [Agent harness primitives, loop control, and trust over black box models](https://www.youtube.com/watch?v=m12vGjfbNlo&t=2930s) | Tejas Kumar, IBM / AI Engineer Singapore | 19:04 | Add if the Google Cloud Agents CLI video is too product-specific. |
| `aie_sg_2026_d1_daytona_sandboxes` | [Why autonomous agents need sandboxes, isolation, and strict boundaries](https://www.youtube.com/watch?v=_xQnSNlBP_w&t=10871s) | Vedran Jukic, Daytona / AI Engineer Singapore | 10:02 | Add if the Microsoft zero-trust video is too enterprise/security-policy-heavy. |
| `aie_sg_2026_d1_groq_latency` | [GroqCloud, low latency inference, custom hardware, and global routing](https://www.youtube.com/watch?v=_xQnSNlBP_w&t=24053s) | Andrew Tan, Groq / AI Engineer Singapore | 10:52 | Add if you need more serving/cost/latency material. |
| `aie_sg_2026_d2_southbridge_context_runtime` | [High context agent runtimes, declarative budgets, and legacy system reliability](https://www.youtube.com/watch?v=m12vGjfbNlo&t=30667s) | Hrishi Olickel, Southbridge / AI Engineer Singapore | 13:23 | Add if you want more production reliability and budget-control material. |
| `aie_sg_2026_d2_smithery_mcp_harness` | [MCP, CLIs, and the harness era of agent agency](https://www.youtube.com/watch?v=m12vGjfbNlo&t=31470s) | Henry Mao, Smithery / AI Engineer Singapore | 12:00 | Add if MCP/tooling becomes central to the corpus. |
| `aie_sg_2026_d1_sonar_executable_evals` | [Code quality agents, remediation loops, and executable evals](https://www.youtube.com/watch?v=_xQnSNlBP_w&t=12069s) | Yuntong Zhang, Sonar / AI Engineer Singapore | 11:45 | Add if the first Arize/Hamel/Sayash eval examples are not code-agent-specific enough. |
| `aie_sg_2026_d1_stripe_minions_judge_loop` | [Minions, one shot coding agents, and the LLM judge loop](https://www.youtube.com/watch?v=_xQnSNlBP_w&t=14040s) | Mark Doyle, Stripe / AI Engineer Singapore | 15:04 | Strong candidate if you want more LLM-judge-loop examples from a production company. |
| `aie_sg_2026_d1_elevenlabs_voice_agents` | [Speech engines, turn taking, and conversational voice agents](https://www.youtube.com/watch?v=_xQnSNlBP_w&t=28531s) | Boris Starkov, ElevenLabs / AI Engineer Singapore | 11:58 | Use if v0 should cover audio/voice tasks. |
| `aie_sg_2026_d2_bifrost_robotics_evals` | [Sim generated worlds, robotics evals, and faster edge case discovery](https://www.youtube.com/watch?v=m12vGjfbNlo&t=21352s) | Aravind Kandiah, Bifrost / AI Engineer Singapore | 14:19 | Use if v0 should include physical AI / robotics evaluation. |
| `spvXj9tnWAQ` | [Engineering Better Evals: Scalable LLM Evaluation Pipelines That Work](https://www.youtube.com/watch?v=spvXj9tnWAQ) | Arize / AI Engineer | 24:45 | Swap in if the Hamel or Lenny eval content is too product/process-heavy. |
| `bk0TmxoZlUY` | [Evals 101](https://www.youtube.com/watch?v=bk0TmxoZlUY) | Doug Guthrie, Braintrust / AI Engineer | 48:31 | Use if you need a longer eval tutorial with more beginner-friendly examples. |
| `a4BV0gGmXgA` | [Five hard earned lessons about Evals](https://www.youtube.com/watch?v=a4BV0gGmXgA) | Ankur Goyal, Braintrust / AI Engineer | 19:45 | Good short replacement for either eval talk. |
| `RVN-K9MXbzk` | [Reasoning Models as LLM Judges: Better or Worse?](https://www.youtube.com/watch?v=RVN-K9MXbzk) | Alex Volkov | 19:10 | Add if judge calibration becomes a central writeup theme. |
| `pnacsAWnjV8` | [ColPali's Vision-Powered RAG for Enterprise Documents](https://www.youtube.com/watch?v=pnacsAWnjV8) | Zain Hasan, PyData | 35:51 | Swap with the PyData visual retrieval talk if ColPali-specific content yields better questions. |
| `gNufpEYBvsw` | [Text Search on Images with Quantized ColPali](https://www.youtube.com/watch?v=gNufpEYBvsw) | Sonam Pankaj, Buzzconf | 20:18 | Use if you want quantization/storage-cost questions for visual retrieval. |
| `W1MiZChnkfA` | [Scaling Enterprise-Grade RAG: Lessons from Legal Frontier](https://www.youtube.com/watch?v=W1MiZChnkfA) | Harvey/Lance, AI Engineer | 16:40 | Add if you want a more enterprise/domain-specific retrieval slice. |
| `640KMYtxCeI` | [Building Multimodal AI Agents From Scratch](https://www.youtube.com/watch?v=640KMYtxCeI) | Apoorva Joshi, AI Engineer | 36:58 | Use if visual/multimodal agent coverage is more important than pure visual document retrieval. |
| `NmblVxyBhi8` | [How LinkedIn Built Their First AI Agent for Hiring with LangGraph](https://www.youtube.com/watch?v=NmblVxyBhi8) | LangChain Interrupt | 15:24 | Keep as an implementation case study; avoid overusing LangChain/LangGraph vendor content. |
| `KUEmEb71vzQ` | [Function Calling is All You Need - Full Workshop](https://www.youtube.com/watch?v=KUEmEb71vzQ) | Ilan Bigio, OpenAI / AI Engineer | 1:42:54 | Useful long-form code/screen-heavy stress test, but too expensive for the first curation pass. |
| `UQjAZrTM0MI` | [Your MCP Server is Bad](https://www.youtube.com/watch?v=UQjAZrTM0MI) | Prefect | 54:00 | Same talk as `96G7FLab8xc`; prefer whichever has better captions/video quality. |
| `XueTa4qrMpg` | [How Engineers and PMs should collaborate on Evals](https://www.youtube.com/watch?v=XueTa4qrMpg) | Hamel Husain | 33:57 | More product/process oriented; useful if the corpus lacks PM/eval workflow questions. |
| `EUxkKELGChM` | [RAG in 2025: State of the Art and the Road Forward](https://www.youtube.com/watch?v=EUxkKELGChM) | AI Council | 25:02 | Same title as the AI Engineer version; pick one, not both. |

## Research-High, Curation-Later

These are valuable to the research community, but should not enter v0 until the ingestion path is verified.

| Source | Candidate | Why defer |
|---|---|---|
| [CVPR 2025](https://cvpr.thecvf.com/virtual/2025/tutorial/35902) | Cognitive AI for the Future: Agentic Multimodal Models and RAG for Vision Language Applications | Excellent multimodal/RAG research fit, but use only after extracting a stable direct video URL and captions. |
| [NeurIPS 2025](https://nips.cc/virtual/2025/talk/127757) | Measuring Emergent Behavior in AI Agents - Weights & Biases | Strong agent-evaluation fit; NeurIPS virtual pages need a direct media/caption verification pass. |
| [NVIDIA GTC 2026](https://www.nvidia.com/en-us/on-demand/session/gtc26-dlit82006/) | Build Cost-Effective, Scalable RAG Pipelines: From Ingestion to Response Generation | Very relevant for cost/scalability; On-Demand site is JS-heavy and may not work cleanly with the v0 `yt-dlp` pipeline. |
| [NVIDIA GTC 2026](https://www.nvidia.com/zh-tw/on-demand/session/gtc26-s82448/) | Open, Trusted, and Observable: Deploying AI Agents at Enterprise Scale | Strong observability/eval fit; verify direct media and captions before moving into v0. |
| [Microsoft Research 2026](https://www.microsoft.com/en-us/research/video/agent-lightning-one-learning-system-that-makes-all-agents-evolve/) | Agent Lightning: One learning system that makes all agents evolve | Research-useful agent learning/evaluation candidate; official page is valuable even if YouTube ingestion is not direct. |
| [AAAI 2026 PLAN-FM](https://plan-fm.github.io/) | Planning and reasoning with foundation models workshop videos | High research relevance and Singapore 2026 recency; use after direct video/caption extraction is verified. |
| [ICLR Blogposts 2026](https://iclr-blogposts.github.io/2026/blog/2026/general-agent-evaluation/) | Ready For General Agents? Let's Test It. | Excellent methodology reference for agent evaluation; not a video, so use as background or pair with a related talk. |
| [Stanford Online](https://www.youtube.com/watch?v=k1njvbBmfsw) | CS230 Lecture 8: Agents, Prompts, and RAG | Good academic baseline, but nearly two hours and broad; add only if the corpus needs a university lecture slice. |

## Rejected For v0

- Short product promos under 5 minutes: too little span density.
- Creator tutorials with "beginner course" framing: useful for learning, weaker as an eval corpus.
- Videos without direct public URLs or verified captions: high operational drag for v0.
- Multiple versions of the same talk: keep one canonical recording to avoid duplicate answers leaking across splits.

## First Curation Pass

Start with these three talks before filling all 12:

1. `nXafozNIk3c` - 2026 build/eval/deploy agent workflow from a non-AI-Engineer source
2. `aie_sg_2026_d2_arize_alyx` - 2026 agent reliability and checkpoints
3. `W_CYk2ogcDI` - compact RAG/retrieval baseline

Target 3-4 verified examples from each. If those examples validate the schema and split logic, continue with the rest of the recommended set.
