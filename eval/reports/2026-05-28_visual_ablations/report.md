# Visual Retrieval Ranking Ablations — 2026-05-28

Side-channel report. n = 10 visual-required examples (tagged `visual-required` in `eval/corpora/ai_engineering_v0/visual_gold.jsonl`).

Two ablations, both diagnostic, no production code modified, no writes to `eval_runs` / `eval_results`.

Substrate: 529 total frame rows, **514 with `pooled_embedding`** (15 NaN/inf-skipped on the 720p re-ingest), **375,734 frame_patches**, 212 chunks. Frames are **1280×720 PNGs** (720p re-fetch, verified on disk via `sips` and in the DB). Numbers below are on this 720p substrate; the per-example tables show `cap=514` because that is the count of frames carrying a non-null pooled embedding.

Pipeline replicated stage-for-stage from `retrieve.visual.retrieve`. Final top-k cap = `200` (wide so we can locate the gold-overlapping chunk's true rank in the MaxSim ordering).

## Ablation 1 — Prefilter K width

Per example, run the 3-stage pipeline at `prefilter_k ∈ {200, 500, 1000}`. Note: 1000 is capped by the substrate (514 of 529 frames have a non-null pooled embedding); the `returned` column shows the actual candidate count.

Each cell shows `frame_rank_in_prefilter / chunk_tr_rank_after_maxsim`. `-` means the gold-window frame was absent from the prefilter at that K; `(no overlap)` means stage 2 produced no gold-overlapping chunks; `abs` means the gold-overlapping chunk reached stage 3 but fell out of the final top-200.

| example_id | K=200 | K=500 | K=1000 (cap=529) | category |
|---|---|---|---|---|
| `visual-required-google-adk-resumability-config-code` | `-` / `-` (200) | `444` / `164` (500) | `444` / `165` (514) | ii_maxsim_demote |
| `visual-required-google-adk-resume-version-and-caveat` | `-` / `-` (200) | `442` / `159` (500) | `442` / `160` (514) | ii_maxsim_demote |
| `visual-required-google-agents-cli-seven-skills-list` | `-` / `-` (200) | `400` / `144` (500) | `400` / `144` (514) | ii_maxsim_demote |
| `visual-required-sally-current-plan-block` | `162` / `64` (200) | `162` / `83` (500) | `162` / `83` (514) | ii_maxsim_demote |
| `visual-required-sally-jq-grepjson-queries` | `141` / `60` (200) | `141` / `70` (500) | `141` / `70` (514) | ii_maxsim_demote |
| `visual-required-sally-lessons-recoverable-exceptions` | `136` / `81` (200) | `136` / `170` (500) | `136` / `172` (514) | ii_maxsim_demote |
| `visual-required-tengyu-accuracy-plot-axes` | `-` / `-` (200) | `238` / `71` (500) | `238` / `71` (514) | ii_maxsim_demote |
| `visual-required-tengyu-accuracy-plot-models` | `-` / `-` (200) | `245` / `63` (500) | `245` / `64` (514) | ii_maxsim_demote |
| `visual-required-tengyu-hybrid-search-diagram` | `197` / `100` (200) | `197` / `137` (500) | `197` / `137` (514) | ii_maxsim_demote |
| `visual-required-tengyu-matryoshka-chart-axes` | `-` / `-` (200) | `258` / `91` (500) | `258` / `91` (514) | ii_maxsim_demote |

### Aggregate K-rescue

- `pass@5` at K=200: **0 / 10**
- additional rescues at K=500: **0** (examples that pass@5 only when K is widened to 500)
- additional rescues at K=1000 (cap=529): **0** (examples that pass@5 only when K is widened past 500)

## Ablation 2 — MaxSim demotion diagnosis

For rows where the gold frame reached the prefilter (at any K) but MaxSim still demoted the gold-overlapping chunk past top-5, expose the top-5 chunks MaxSim returned vs the gold chunk.

The hypothesis under test: MaxSim is preferring whole-frame visual similarity (e.g. speaker-on-stage frames, similar slide template) over actual slide-text relevance. We check this by inspecting whether the top-5 chunks come from the same video as the gold and whether their transcript text is topically related to the question.

### `visual-required-google-adk-resumability-config-code`  (deepdive at prefilter_k=500)

Question: On the ADK Resume stopped agents page, what is the exact code shown for adding resumable configuration to an ADK app?

Gold chunk: `chunk_id=126` `[2001, 2030]` sec, MaxSim = `28.504`, rank = `164` / `211` chunks. Gap to top-5 cutoff (cutoff = `30.223`): **`+1.718`** MaxSim units.

Gold text preview: _"any production agent survive that any production agent survive that reality. Uh let's look at the resume reality. Uh let's look at the resume reality. Uh let's look at the resume agent first. So what "_

| rank | chunk_id | video_id | span (sec) | MaxSim | same_video | overlaps_gold | text preview |
|---|---|---|---|---|---|---|---|
| 1 | `162` | `nXafozNIk3c` | [2900, 2931] | 30.395 | True | False | _'but at least understand the core but at least understand the core concepts uh at least understand what concepts uh at least understand what concepts uh at least understand what changes when you change'_ |
| 2 | `161` | `nXafozNIk3c` | [2874, 2906] | 30.347 | True | False | _"there &gt;&gt; okay I think that's like a very hot &gt;&gt; okay I think that's like a very hot &gt;&gt; okay I think that's like a very hot topic so a lot of people will be topic so a lot of people w"_ |
| 3 | `138` | `nXafozNIk3c` | [2299, 2331] | 30.302 | True | False | _"looking through your recent publications on X, which you're doing a great job of, on X, which you're doing a great job of, on X, which you're doing a great job of, by the way. by the way. by the way. "_ |
| 4 | `166` | `nXafozNIk3c` | [3001, 3029] | 30.299 | True | False | _"anything and everything that you're anything and everything that you're doing. Just a simple model call will doing. Just a simple model call will doing. Just a simple model call will mostly do the thi"_ |
| 5 | `100` | `nXafozNIk3c` | [1349, 1381] | 30.223 | True | False | _'buy more H100s H2 good advice of course we need more compute. Yeah. Yeah. So we need more compute. Yeah. Yeah. So we need more compute. Yeah. Yeah. So yeah overall this is working. Just to yeah overal'_ |

### `visual-required-google-adk-resume-version-and-caveat`  (deepdive at prefilter_k=500)

Question: On the ADK Resume stopped agents docs page, what ADK Python version supports the Resume feature, and what caution does the page raise about it?

Gold chunk: `chunk_id=126` `[2001, 2030]` sec, MaxSim = `29.809`, rank = `159` / `211` chunks. Gap to top-5 cutoff (cutoff = `32.257`): **`+2.447`** MaxSim units.

Gold text preview: _"any production agent survive that any production agent survive that reality. Uh let's look at the resume reality. Uh let's look at the resume reality. Uh let's look at the resume agent first. So what "_

| rank | chunk_id | video_id | span (sec) | MaxSim | same_video | overlaps_gold | text preview |
|---|---|---|---|---|---|---|---|
| 1 | `162` | `nXafozNIk3c` | [2900, 2931] | 32.560 | True | False | _'but at least understand the core but at least understand the core concepts uh at least understand what concepts uh at least understand what concepts uh at least understand what changes when you change'_ |
| 2 | `161` | `nXafozNIk3c` | [2874, 2906] | 32.447 | True | False | _"there &gt;&gt; okay I think that's like a very hot &gt;&gt; okay I think that's like a very hot &gt;&gt; okay I think that's like a very hot topic so a lot of people will be topic so a lot of people w"_ |
| 3 | `171` | `nXafozNIk3c` | [3124, 3154] | 32.343 | True | False | _'are being mostly more and more used by agents and not just humans. Uh so my AI agents and not just humans. Uh so my AI agents and not just humans. Uh so my AI agents is something that I cannot really '_ |
| 4 | `170` | `nXafozNIk3c` | [3100, 3130] | 32.267 | True | False | _'months ago, it would be very different. Now I would say like my AI agents. Uh so Now I would say like my AI agents. Uh so Now I would say like my AI agents. Uh so my AI agents is my interface to any a'_ |
| 5 | `101` | `nXafozNIk3c` | [1375, 1404] | 32.257 | True | False | _"AI agents and I can see a lot of AI agents and I can see a lot of developers being really excited about developers being really excited about developers being really excited about this. Especially I'v"_ |

### `visual-required-google-agents-cli-seven-skills-list`  (deepdive at prefilter_k=500)

Question: On the Agent CLI Getting Started page, what are the seven skill IDs the CLI bundles and what does each one teach the coding agent?

Gold chunk: `chunk_id=66` `[499, 531]` sec, MaxSim = `26.588`, rank = `144` / `208` chunks. Gap to top-5 cutoff (cutoff = `27.595`): **`+1.006`** MaxSim units.

Gold text preview: _'agent CLI in agent platform is a CLI and skills packages combined to give the skills packages combined to give the skills packages combined to give the models your coding agent. You can use it models '_

| rank | chunk_id | video_id | span (sec) | MaxSim | same_video | overlaps_gold | text preview |
|---|---|---|---|---|---|---|---|
| 1 | `174` | `aie_sg_2026_d2_arize_alyx` | [2, 29] | 27.822 | False | False | _"&gt;&gt; Good morning everyone. Thanks so much &gt;&gt; Good morning everyone. Thanks so much for spending your morning with me. It's for spending your morning with me. It's for spending your morning "_ |
| 2 | `175` | `aie_sg_2026_d2_arize_alyx` | [29, 55] | 27.733 | False | False | _'Sorry, I got to reconnect to my hotspot. Sorry, I got to reconnect to my hotspot. I thought I did this already. Cool. There we go. Good morning Cool. There we go. Good morning everyone. Uh thanks so m'_ |
| 3 | `185` | `aie_sg_2026_d2_arize_alyx` | [274, 305] | 27.696 | False | False | _'um this is something that we borrowed from some of our our favorite tools like from some of our our favorite tools like from some of our our favorite tools like Claude. Um and this has been a real Cla'_ |
| 4 | `45` | `W_CYk2ogcDI` | [1100, 1127] | 27.653 | False | False | _'has all the details uh of the has all the details uh of the corresponding trunk and also has some corresponding trunk and also has some corresponding trunk and also has some kind of like cross informa'_ |
| 5 | `209` | `aie_sg_2026_d2_arize_alyx` | [876, 905] | 27.595 | False | False | _'also things like G-Cloud logs. Um, so we also things like G-Cloud logs. Um, so we found had an example with like out of found had an example with like out of found had an example with like out of memo'_ |

### `visual-required-sally-current-plan-block`  (deepdive at prefilter_k=200)

Question: On SallyAnn DeLucia's "Plans Lives Outside Conversation History" slide, what exact three-item current plan does the slide show Alyx seeing, what status marker is on each item, and what todo_update call does the slide instruct Alyx to make when it finishes the in-progress task?

Gold chunk: `chunk_id=186` `[300, 329]` sec, MaxSim = `50.308`, rank = `64` / `101` chunks. Gap to top-5 cutoff (cutoff = `52.162`): **`+1.854`** MaxSim units.

Gold text preview: _'trying to accomplish. Um, and just really improved our ability to complete really improved our ability to complete really improved our ability to complete our our task correctly. our our task correctl'_

| rank | chunk_id | video_id | span (sec) | MaxSim | same_video | overlaps_gold | text preview |
|---|---|---|---|---|---|---|---|
| 1 | `161` | `nXafozNIk3c` | [2874, 2906] | 53.098 | False | False | _"there &gt;&gt; okay I think that's like a very hot &gt;&gt; okay I think that's like a very hot &gt;&gt; okay I think that's like a very hot topic so a lot of people will be topic so a lot of people w"_ |
| 2 | `116` | `nXafozNIk3c` | [1749, 1781] | 52.426 | False | False | _'do this like if you would have asked me 6 months ago is it possible I would like 6 months ago is it possible I would like 6 months ago is it possible I would like maybe but now we can do it uh which i'_ |
| 3 | `170` | `nXafozNIk3c` | [3100, 3130] | 52.231 | False | False | _'months ago, it would be very different. Now I would say like my AI agents. Uh so Now I would say like my AI agents. Uh so Now I would say like my AI agents. Uh so my AI agents is my interface to any a'_ |
| 4 | `138` | `nXafozNIk3c` | [2299, 2331] | 52.216 | False | False | _"looking through your recent publications on X, which you're doing a great job of, on X, which you're doing a great job of, on X, which you're doing a great job of, by the way. by the way. by the way. "_ |
| 5 | `169` | `nXafozNIk3c` | [3075, 3106] | 52.162 | False | False | _"into production don't just fly uh by good wipes okay this looks good and let good wipes okay this looks good and let good wipes okay this looks good and let me put it in production because high me put"_ |

### `visual-required-sally-jq-grepjson-queries`  (deepdive at prefilter_k=200)

Question: On SallyAnn DeLucia's "Small, composable tools == Infinite Context" slide, what four exact jq and grep_json query strings are shown as examples of composable tools Alyx uses over serialized data?

Gold chunk: `chunk_id=196` `[551, 581]` sec, MaxSim = `40.140`, rank = `60` / `104` chunks. Gap to top-5 cutoff (cutoff = `41.225`): **`+1.085`** MaxSim units.

Gold text preview: _'line and GP JSON which is able to do line and GP JSON which is able to do reex search over serialized data. Um and reex search over serialized data. Um and reex search over serialized data. Um and the'_

| rank | chunk_id | video_id | span (sec) | MaxSim | same_video | overlaps_gold | text preview |
|---|---|---|---|---|---|---|---|
| 1 | `163` | `nXafozNIk3c` | [2924, 2954] | 41.316 | False | False | _'retrieval is more important than ever agree strongly agree that people should agree strongly agree that people should agree strongly agree that people should understand and really learn about understa'_ |
| 2 | `161` | `nXafozNIk3c` | [2874, 2906] | 41.286 | False | False | _"there &gt;&gt; okay I think that's like a very hot &gt;&gt; okay I think that's like a very hot &gt;&gt; okay I think that's like a very hot topic so a lot of people will be topic so a lot of people w"_ |
| 3 | `169` | `nXafozNIk3c` | [3075, 3106] | 41.259 | False | False | _"into production don't just fly uh by good wipes okay this looks good and let good wipes okay this looks good and let good wipes okay this looks good and let me put it in production because high me put"_ |
| 4 | `162` | `nXafozNIk3c` | [2900, 2931] | 41.257 | False | False | _'but at least understand the core but at least understand the core concepts uh at least understand what concepts uh at least understand what concepts uh at least understand what changes when you change'_ |
| 5 | `166` | `nXafozNIk3c` | [3001, 3029] | 41.225 | False | False | _"anything and everything that you're anything and everything that you're doing. Just a simple model call will doing. Just a simple model call will doing. Just a simple model call will mostly do the thi"_ |

### `visual-required-sally-lessons-recoverable-exceptions`  (deepdive at prefilter_k=200)

Question: On SallyAnn DeLucia's "Lessons in context management" summary slide, what five bullet-point lessons are listed, and what is the exact code-shaped identifier the slide uses for the lesson about feedback loops?

Gold chunk: `chunk_id=197` `[575, 606]` sec, MaxSim = `39.469`, rank = `81` / `103` chunks. Gap to top-5 cutoff (cutoff = `42.212`): **`+2.743`** MaxSim units.

Gold text preview: _'always like to kind of make this to the liking of like a a UX programmer. You liking of like a a UX programmer. You liking of like a a UX programmer. You can think of your tools and then like can thin'_

| rank | chunk_id | video_id | span (sec) | MaxSim | same_video | overlaps_gold | text preview |
|---|---|---|---|---|---|---|---|
| 1 | `162` | `nXafozNIk3c` | [2900, 2931] | 42.491 | False | False | _'but at least understand the core but at least understand the core concepts uh at least understand what concepts uh at least understand what concepts uh at least understand what changes when you change'_ |
| 2 | `170` | `nXafozNIk3c` | [3100, 3130] | 42.464 | False | False | _'months ago, it would be very different. Now I would say like my AI agents. Uh so Now I would say like my AI agents. Uh so Now I would say like my AI agents. Uh so my AI agents is my interface to any a'_ |
| 3 | `161` | `nXafozNIk3c` | [2874, 2906] | 42.401 | False | False | _"there &gt;&gt; okay I think that's like a very hot &gt;&gt; okay I think that's like a very hot &gt;&gt; okay I think that's like a very hot topic so a lot of people will be topic so a lot of people w"_ |
| 4 | `158` | `nXafozNIk3c` | [2799, 2829] | 42.283 | False | False | _"context so I would say the debate is more around rag versus long context now more around rag versus long context now more around rag versus long context now and yes it's complicated and yes it's compl"_ |
| 5 | `157` | `nXafozNIk3c` | [2776, 2804] | 42.212 | False | False | _'wrong framing. It used to make sense a wrong framing. It used to make sense a year ago. I would reframe it as year ago. I would reframe it as year ago. I would reframe it as uh rag versus long context'_ |

### `visual-required-tengyu-accuracy-plot-axes`  (deepdive at prefilter_k=500)

Question: On the 'Significant Improvements in Retrieval Accuracy Over Last 2 Years' plot, what are the two axis labels and what are the x-axis tick values shown?

Gold chunk: `chunk_id=17` `[400, 432]` sec, MaxSim = `35.616`, rank = `71` / `211` chunks. Gap to top-5 cutoff (cutoff = `36.234`): **`+0.618`** MaxSim units.

Gold text preview: _'that are common uh I think you can get already very high accuracy in the already very high accuracy in the already very high accuracy in the retrieval step. Um and another thing retrieval step. Um and'_

| rank | chunk_id | video_id | span (sec) | MaxSim | same_video | overlaps_gold | text preview |
|---|---|---|---|---|---|---|---|
| 1 | `170` | `nXafozNIk3c` | [3100, 3130] | 36.313 | False | False | _'months ago, it would be very different. Now I would say like my AI agents. Uh so Now I would say like my AI agents. Uh so Now I would say like my AI agents. Uh so my AI agents is my interface to any a'_ |
| 2 | `161` | `nXafozNIk3c` | [2874, 2906] | 36.286 | False | False | _"there &gt;&gt; okay I think that's like a very hot &gt;&gt; okay I think that's like a very hot &gt;&gt; okay I think that's like a very hot topic so a lot of people will be topic so a lot of people w"_ |
| 3 | `165` | `nXafozNIk3c` | [2975, 3005] | 36.264 | False | False | _"If it's a simple workflow, that's that's If it's a simple workflow, that's that's all you need. Start simple and decide all you need. Start simple and decide all you need. Start simple and decide base"_ |
| 4 | `116` | `nXafozNIk3c` | [1749, 1781] | 36.237 | False | False | _'do this like if you would have asked me 6 months ago is it possible I would like 6 months ago is it possible I would like 6 months ago is it possible I would like maybe but now we can do it uh which i'_ |
| 5 | `156` | `nXafozNIk3c` | [2749, 2779] | 36.234 | False | False | _'parse through the first thing that they see. Every interface you design has to see. Every interface you design has to see. Every interface you design has to work for humans as well as AI agents. work '_ |

### `visual-required-tengyu-accuracy-plot-models`  (deepdive at prefilter_k=500)

Question: Which embedding-model providers are color-coded in the legend of Tengyu Ma's retrieval-accuracy plot, and which provider's models sit on the upper-right Pareto frontier?

Gold chunk: `chunk_id=15` `[351, 380]` sec, MaxSim = `37.080`, rank = `63` / `211` chunks. Gap to top-5 cutoff (cutoff = `37.659`): **`+0.579`** MaxSim units.

Gold text preview: _'becomes cheaper. Um and all of these are becomes cheaper. Um and all of these are through kind of like you know optimizing through kind of like you know optimizing through kind of like you know optimi'_

| rank | chunk_id | video_id | span (sec) | MaxSim | same_video | overlaps_gold | text preview |
|---|---|---|---|---|---|---|---|
| 1 | `163` | `nXafozNIk3c` | [2924, 2954] | 37.860 | False | False | _'retrieval is more important than ever agree strongly agree that people should agree strongly agree that people should agree strongly agree that people should understand and really learn about understa'_ |
| 2 | `162` | `nXafozNIk3c` | [2900, 2931] | 37.829 | False | False | _'but at least understand the core but at least understand the core concepts uh at least understand what concepts uh at least understand what concepts uh at least understand what changes when you change'_ |
| 3 | `159` | `nXafozNIk3c` | [2824, 2855] | 37.774 | False | False | _'to run it. They should be able to understand the code that it has uh all understand the code that it has uh all understand the code that it has uh all of that in the easiest way possible. So of that i'_ |
| 4 | `100` | `nXafozNIk3c` | [1349, 1381] | 37.706 | False | False | _'buy more H100s H2 good advice of course we need more compute. Yeah. Yeah. So we need more compute. Yeah. Yeah. So we need more compute. Yeah. Yeah. So yeah overall this is working. Just to yeah overal'_ |
| 5 | `169` | `nXafozNIk3c` | [3075, 3106] | 37.659 | False | False | _"into production don't just fly uh by good wipes okay this looks good and let good wipes okay this looks good and let good wipes okay this looks good and let me put it in production because high me put"_ |

### `visual-required-tengyu-hybrid-search-diagram`  (deepdive at prefilter_k=200)

Question: On the 'Hybrid Search and Rerankers' diagram, what specific keyword-search algorithm is named, and what two annotations describe the role of hybrid search and the role of the reranker?

Gold chunk: `chunk_id=19` `[450, 480]` sec, MaxSim = `37.689`, rank = `100` / `108` chunks. Gap to top-5 cutoff (cutoff = `39.157`): **`+1.468`** MaxSim units.

Gold text preview: _'maybe with like one or 2% loss and quantization is kind of this in a quantization is kind of this in a quantization is kind of this in a similar vein. So where you are even you similar vein. So where '_

| rank | chunk_id | video_id | span (sec) | MaxSim | same_video | overlaps_gold | text preview |
|---|---|---|---|---|---|---|---|
| 1 | `49` | `nXafozNIk3c` | [77, 105] | 39.340 | False | False | _'released from ADK 2.0 O and agent CLI released from ADK 2.0 O and agent CLI and how the experience of building and how the experience of building and how the experience of building agents has just bee'_ |
| 2 | `182` | `aie_sg_2026_d2_arize_alyx` | [199, 231] | 39.215 | False | False | _'think that this is something that everybody really tries to solve. Um everybody really tries to solve. Um everybody really tries to solve. Um people commonly ask me like well why is people commonly as'_ |
| 3 | `170` | `nXafozNIk3c` | [3100, 3130] | 39.189 | False | False | _'months ago, it would be very different. Now I would say like my AI agents. Uh so Now I would say like my AI agents. Uh so Now I would say like my AI agents. Uh so my AI agents is my interface to any a'_ |
| 4 | `169` | `nXafozNIk3c` | [3075, 3106] | 39.179 | False | False | _"into production don't just fly uh by good wipes okay this looks good and let good wipes okay this looks good and let good wipes okay this looks good and let me put it in production because high me put"_ |
| 5 | `56` | `nXafozNIk3c` | [251, 280] | 39.157 | False | False | _"list and me being not a hardcore coder list and me being not a hardcore coder coder so it's a it's a huge deal uh and coder so it's a it's a huge deal uh and coder so it's a it's a huge deal uh and th"_ |

### `visual-required-tengyu-matryoshka-chart-axes`  (deepdive at prefilter_k=500)

Question: On the Matryoshka Learning and Quantization-Aware Training slide, what does the right-hand chart's x-axis show, and what three marker shapes appear in the Embedding Quantization legend?

Gold chunk: `chunk_id=18` `[426, 456]` sec, MaxSim = `38.875`, rank = `91` / `211` chunks. Gap to top-5 cutoff (cutoff = `39.918`): **`+1.043`** MaxSim units.

Gold text preview: _'you have like a high dimensional you have like a high dimensional embedding right you can use a subset of embedding right you can use a subset of embedding right you can use a subset of the uh the coo'_

| rank | chunk_id | video_id | span (sec) | MaxSim | same_video | overlaps_gold | text preview |
|---|---|---|---|---|---|---|---|
| 1 | `170` | `nXafozNIk3c` | [3100, 3130] | 40.056 | False | False | _'months ago, it would be very different. Now I would say like my AI agents. Uh so Now I would say like my AI agents. Uh so Now I would say like my AI agents. Uh so my AI agents is my interface to any a'_ |
| 2 | `161` | `nXafozNIk3c` | [2874, 2906] | 40.025 | False | False | _"there &gt;&gt; okay I think that's like a very hot &gt;&gt; okay I think that's like a very hot &gt;&gt; okay I think that's like a very hot topic so a lot of people will be topic so a lot of people w"_ |
| 3 | `171` | `nXafozNIk3c` | [3124, 3154] | 39.949 | False | False | _'are being mostly more and more used by agents and not just humans. Uh so my AI agents and not just humans. Uh so my AI agents and not just humans. Uh so my AI agents is something that I cannot really '_ |
| 4 | `47` | `nXafozNIk3c` | [26, 56] | 39.919 | False | False | _'where he works on the AI products that where he works on the AI products that millions of developers build with every millions of developers build with every millions of developers build with every si'_ |
| 5 | `163` | `nXafozNIk3c` | [2924, 2954] | 39.918 | False | False | _'retrieval is more important than ever agree strongly agree that people should agree strongly agree that people should agree strongly agree that people should understand and really learn about understa'_ |

## Per-example failure classification

Categories:

- **(i) cosine_blind** — gold frame absent from prefilter at ALL K values.
- **(ii) maxsim_demote** — gold frame present at some K, but MaxSim demotes the gold-overlapping chunk past top-5 at every K.
- **(iii) passes** — gold-overlapping chunk reaches top-5 in the final MaxSim ranking at at least one K.
- **(iv) other** — frame present at some K, but post-join produces no gold-overlapping chunk, or other structural anomaly.

| category | n |
|---|---|
| i_cosine_blind | 0 |
| ii_maxsim_demote | 10 |
| iii_passes | 0 |
| iv_other | 0 |

| example_id | category | gold_frame_at_K=200 | gold_frame_at_K=500 | gold_frame_at_K=1000 | chunk_rank_at_K=200 |
|---|---|---|---|---|---|
| `visual-required-google-adk-resumability-config-code` | ii_maxsim_demote | absent (`0`/`200`) | `444`/`500` | `444`/`514` | absent |
| `visual-required-google-adk-resume-version-and-caveat` | ii_maxsim_demote | absent (`0`/`200`) | `442`/`500` | `442`/`514` | absent |
| `visual-required-google-agents-cli-seven-skills-list` | ii_maxsim_demote | absent (`0`/`200`) | `400`/`500` | `400`/`514` | absent |
| `visual-required-sally-current-plan-block` | ii_maxsim_demote | `162`/`200` | `162`/`500` | `162`/`514` | 64 |
| `visual-required-sally-jq-grepjson-queries` | ii_maxsim_demote | `141`/`200` | `141`/`500` | `141`/`514` | 60 |
| `visual-required-sally-lessons-recoverable-exceptions` | ii_maxsim_demote | `136`/`200` | `136`/`500` | `136`/`514` | 81 |
| `visual-required-tengyu-accuracy-plot-axes` | ii_maxsim_demote | absent (`0`/`200`) | `238`/`500` | `238`/`514` | absent |
| `visual-required-tengyu-accuracy-plot-models` | ii_maxsim_demote | absent (`0`/`200`) | `245`/`500` | `245`/`514` | absent |
| `visual-required-tengyu-hybrid-search-diagram` | ii_maxsim_demote | `197`/`200` | `197`/`500` | `197`/`514` | 100 |
| `visual-required-tengyu-matryoshka-chart-axes` | ii_maxsim_demote | absent (`0`/`200`) | `258`/`500` | `258`/`514` | absent |

## Dominant failure mode

Counts (n=10): (i) cosine_blind = **0**, (ii) maxsim_demote = **10**, (iii) passes = **0**, (iv) other = **0**.

The data above does **not** support the earlier "cosine-blind" reading. On the 720p substrate every one of the 10 examples is `ii_maxsim_demote`, and `i_cosine_blind = 0`. The precise picture is two-stage:

1. **At production `prefilter_k=200` the channel is prefilter-limited.** The gold-window frame is absent from the top-200 pooled-cosine prefilter on 7 of 10 rows (the three Google-ADK rows, both Tengyu accuracy-plot rows, the Tengyu matryoshka row, and one more). Those rows can never pass at K=200 regardless of MaxSim.
2. **Widening the prefilter exposes a MaxSim/ranking failure.** At `prefilter_k=500` all 10 gold frames enter the candidate set (ranked 136–444 of 514), but MaxSim then demotes the gold-overlapping chunk to rank 60–172 of ~210. `pass@5` stays 0/10 at K ∈ {200, 500, 1000}. The MaxSim gap from the gold chunk to the top-5 cutoff is tiny and consistent across K (+0.6 to +2.5 units), and the chunks that out-rank it are same-talk but topically unrelated — a near-uniform MaxSim similarity floor across frames.

So the correct conclusion is: **production is still prefilter-limited (7/10 miss the gold frame at K=200); widening the prefilter to K=500 rescues frame recall but uncovers a MaxSim/reranking failure that K-width cannot fix.** Both stages must be addressed — wider prefilter AND a better aggregator/reranker over ≥200 candidates (the gold chunks rank 137–172, so a top-100 rerank would miss them).
