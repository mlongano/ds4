# Upstream intake verdicts, 2026-10-05

Per-commit take/skip verdicts for everything in `git log --oneline HEAD..origin/main`,
as required by `docs/UPSTREAM.md` step 2. Written by the MiMo-V2.6-Pro session of
2026-10-05 against the fetch of that day: 156 incoming commits, merge base `b0a147a`.
The reasoning behind the grouping lives in
`docs/STATE_OF_THE_CODEBASE_MiMo-V2.6-Pro_2026-10-05.md`.

## What the verdicts mean

A merge of `origin/main` brings in every commit regardless of its verdict. The verdict
records whether the commit's *intent* has to survive. In a conflict, "take" intent is
preserved and re-applied; "skip" intent may be dropped when it fights our delta. That is
the whole operational meaning.

Verdict vocabulary:

- **take** — a bug fix, correctness improvement or measured win that sits in paths we
  run (loading, KV lifetime, SSD streaming, session and cache policy, the ROCm build).
- **take, no effect here** — correctness we merge for free but which cannot change the
  R9700 deployment (Metal-only paths).
- **conditional** — worth having only under a stated condition.
- **decision** — needs a human call before intake, not a routine merge.
- **skip** — refactor or feature work whose intent our delta already rewrites or does not
  serve. Not an insult, just not ours to defend in a conflict.

## Summary

| Verdict | Count | Notes |
|---|---:|---|
| take (must, correctness in deployed paths) | 7 | Read every diff before merging |
| take (streaming, cache and session policy) | 17 | Shared policy code our RAM-disk mode rides on |
| take (server and tool rendering correctness) | 14 | `ds4_server.c`, ours is nearly untouched, so free |
| take (build hygiene) | 8 | Protects `make rdna4` and the test builds |
| take, no effect here (Metal-only) | 4 | |
| skip (gfx1151 tuning) | 4 | R9700-only, decided 2026-10-05 |
| take as-is (DeepSeek v4.1, no ROCm path upstream) | 4 | Wanted on the R9700; needs the port described below |
| skip (Qwen3.8 cluster) | 20 | |
| skip (Metal batched decode, speculative batches) | 29 | |
| skip (CUDA and DSpark tuning) | 5 | |
| skip (docs, QA, eval bookkeeping) | 32 | Read two of them for recorded bugs |
| skip (agent and terminal UI) | 6 | |
| merge commits (verdict follows content) | 6 | |
| total | 156 | |

## Take: must, correctness in deployed paths

These touch models and memory paths this machine runs. Read each diff.

- `9d9e129` Budget ROCm allocations against usable pinned memory. The big one. Reroutes
  every `cudaMalloc`, `cudaMallocHost` and `cudaMemGetInfo` through wrappers, replaces
  `cuda_model_arena_chunk_bytes` (the DeepSeek Q4 span logic `c229363` builds on) and
  rewrites the Q8 reserve, all in `rocm/ds4_rocm_runtime.cuh`. Its admission layer is a
  no-op on discrete GPUs, so after merging we need a discrete counterpart to its tests.
- `0e9cc2d` Fix GLM attention masking and reductions across GPU backends. Correctness in
  `rocm/ds4_rocm_glm.cuh` for a model we serve. Strix continued prefill 21 to 68 t/s came
  with it, which shows how broken the masked path was.
- `233eeb8` Fix session rollback, TP restore and GLM memory accounting. Same area as our
  `575a4c7`, plus TP protocol v10 (peers must update together).
- `582d809` Fix GLM-5.3 Metal decode masking for padded selections.
- `9775e26` Test GLM pooled decode padding against an independent reference.
- `e965352` Preserve RoPE in full GLM compact prefill.
- `fb2abb2` Align Metal GLM attention scratch for arbitrary context lengths.

## Take: streaming, cache and session policy

The machinery our RAM-disk SSD mode extends (`ds4_ssd.*`, the expert caches, the read
pipeline). These were written against Metal or CUDA but the policy code is shared.

- `85837d0` Use more of the safe Metal working set for DeepSeek expert caching. The only
  upstream commit touching `ds4_ssd.c/h`; adds two API bits and `tests/test_ssd_cache.c`.
  Re-check the RAM-disk integration against the new interface.
- `b6af0ad` Bound GLM SSD prefill cache priorities so decode can adapt. A real eviction
  bug: inflated hotness pinned the seed and starved demand-loaded experts. GLM 5.3 SSD
  agent decode at 100K went 0.48 to 4.38 t/s.
- `437dbcb` Read full IQ2 GLM layers ahead during large continued prefills.
- `660e1d4` Bound Metal streaming model views without losing auxiliary mappings.
- `715ed7a` Age Metal expert caches from the first routed layer.
- `91dda0b` Let Metal expert caching adapt beyond the built-in preload ranking.
- `edcafe4` Reuse GLM router readback during IQ2 expert streaming.
- `d7ade9f` Reuse cached IQ2 experts for small Metal GLM prefills.
- `6ccc1e7` Use the available Metal cache and overlap the first GLM streaming prefill.
- `22e4172` Keep static Metal streaming weights resident within the runtime budget.
- `d6580a5` Let parallel Metal prefill reads run without serial read-ahead.
- `e9e1baa` Process medium CUDA SSD appends in one layer sweep.
- `c2c3ce3` Overlap CUDA SSD expert reads with V4.1 prefill.
- `c0a6119` Preserve the final byte of in-memory session snapshots. Session state
  correctness, no performance story attached, which usually means it matters.
- `7a8c1b2` dspark: stop verification at the first EOS. Speculative verification
  correctness on the DeepSeek path.
- `4d7028e` server: reuse-aware slot routing via shared live-state probe.
- `ada805a` server: staleness-aware eviction tiers in slot routing.

## Take: server and tool rendering correctness

Our `ds4_server.c` delta is one commit, so none of this can conflict with local intent.

- `5994f06` Validate image positions before token-prefix slot reuse.
- `81a7658` Preserve live vision context across server tool calls.
- `63bbba4` Advertise the loaded GLM model id in /v1/models.
- `ac43e90` Fix progressive JPEG AC refinement ZRL handling in Iris decoder. Vision input
  correctness.
- `fff391e` GLM: render the tool call block inline, as the chat template does.
- `d108ae4` Preserve literal tool data across agent and server rendering.
- `fc6414c` Keep literal tool and thinking markers inside argument data.
- `d077fa6` Keep split tool delimiters in the server sampling tracker.
- `759dd7c` Do not repair truncated tool actions or exceed output limits.
- `5b3cc8b` Respect GLM tool syntax and speculative server boundaries.
- `e1711c2` Report tool observation errors without compacting the conversation.
- `6eac69e` Bound tool recovery by the request budget and explicit stops.
- `930ab73` Keep valid opportunistic drafts across tool sampling boundaries.
- `558cd34` Restore text-only agent tool results for DeepSeek.

## Take: build hygiene

Small, protective, no downside.

- `6687c6f` Guard ROCm memory baseline out of CPU-only builds.
- `49ec955` Keep ROCm QA builds clean for partial tiles and CPU-only fixtures.
- `064bc22` Fix Metal object header dependencies.
- `ade4d62` Fix Linux builds after Metal TP changes. Directly reduces the surface our
  `575a4c7` portability commit guards.
- `932896f` Fix standalone Metal test linkage.
- `bab2029` Keep the Qwen Metal graph out of CUDA and ROCm builds.
- `96f547a` Remove unused ROCm kernels and helpers. Checked: no caller in our tree for any
  deleted symbol.
- `4a14eb8` Remove unused CUDA and ROCm Q8 matvec kernels. Checked: the helpers our added
  `matmul_q8_0_f32_batch_sharedx_pair_rows_kernel` uses (`q8_0_scale_broadcast_w32`,
  `warp_sum_f32`) survive upstream. Still re-run the `DS4_ROCM_NO_WMMA` build after merge.

## Take, no effect here: Metal-only correctness

Merge is free on the ROCm path. Take the intent anyway.

- `0719a0b` Improve Metal router softplus accuracy for small logits.
- `4814f44` Remove managed-storage notifications from shared Metal buffers.
- `8fcd61d` Fix Metal scratch size for double-buffered dense kernels.
- `2ea3839` Reuse routed activations in Metal tensor prefill.

## Skip: gfx1151 tuning, decided 2026-10-05

The fork is R9700-only from here on, so this group is a skip: the intent need not
survive a conflict. The commits still land with the merge. `394865c` touches `ds4.c`, so
its hunk can appear in a conflict where the skip applies, and the other three rewrite
`rocm/` files our perf chain also rewrites, which is the only reason they were ever a
question.

- `1ef9bba` rocm: select version-matched gfx1151 Lt prefill plans.
- `e83b106` rocm: accelerate gfx1151 DSpark verifier attention projection.
- `64e490b` rocm: specialize gfx1151 IQ2 prefill input staging.
- `394865c` rocm: align DSpark target features and retain their rolling window. Touches
  `ds4.c`, so slightly less conditional than the three above.

## Take as-is: DeepSeek v4.1 Flash support

To be clear about what the choice here is, because "own intake branch" is not about git
mechanics. The merge drags this code in regardless of any verdict. The open question is
what this machine does with it afterwards.

Upstream wrote v4.1 for Metal (`bd66c40`) and CUDA (`a04f46f`) only. Nothing in the 156
commits adds v4.1 to the ROCm backend. So after the merge the code is present but inert
on the R9700: it does nothing unless a v4.1 model is loaded, and loading one would mean
porting work first.

Also model-bound and dead if the model changes: the refusal vector is 43 x 4096 for V4
Flash and has to be rebuilt, not copied, and the vision encoder pairing and expert
cache defaults need re-checking.

Verdict: take the code as-is. v4.1 is wanted on this machine (decided 2026-10-05),
which turns "port v4.1 to the ROCm backend" into planned work rather than a footnote.
The feasibility picture, the weights and disk blocker, and the phased plan live in
`docs/STATE_OF_THE_CODEBASE_MiMo-V2.6-Pro_2026-10-05.md`. The refusal direction must be
rebuilt for v4.1 either way, and rebuilding it on a published dataset with recorded
calibration is the chance to improve on `refusal_train400`, not just match it.

- `bd66c40` DeepSeek v4.1 Flash support for Metal.
- `a04f46f` DeepSeek v4.1 Flash support for CUDA.
- `d9bc45d` Add fresh official V4.1 Flash continuations and router quality comparisons.
- `6c00e2d` Let V4.1 single-token decode use the concurrent Engram reader.

## Skip: Qwen3.8 Flash Next cluster

New model family this fork does not run. Skip means the intent need not survive a
conflict. One footnote: `67bd55c` touches steering, an area we care about, but only for
Qwen standalone generation.

- `76` `ccea768` Add Qwen3.8 Flash Next model support
- `7` `077a257` Keep decode-sized n-gram concurrency on Apple dispatch workers
- `17` `39bc628` Cover split Qwen reasoning markers in both streaming APIs
- `18` `d31089d` Fix Qwen tool content streaming
- `23` `8db1d1d` Document Qwen Metal batching and its quality checks
- `24` `ab38bc6` Serve Qwen batched MTP with safe session state and output limits
- `26` `8e608a6` Trim Qwen batching controls and benchmark documentation
- `52` `f396751` Optimize Qwen CUDA projections and expert kernels
- `55` `f74d410` Check Qwen CUDA split attention against CPU references
- `56` `02a2558` Share Qwen CUDA prefill attention across grouped query heads
- `57` `7fe838b` Reuse Qwen CUDA prefill normalization across injection chunks
- `59` `df6eedd` Add optimized CUDA inference for Qwen3.8 Flash Next
- `60` `9139e2a` Document and download self-contained Qwen BF16 n-gram releases
- `61` `67bd55c` Honor steering in standalone greedy Qwen generation
- `62` `8464282` Keep short Qwen prefill tails on FP32-query attention
- `63` `97ba327` Read Qwen native BF16 n-grams directly from the GGUF
- `64` `e68bcc8` Package Qwen with original BF16 n-grams
- `67` `cbaf6cd` Use explicit Qwen scoring inputs and remove an unused CUDA stub
- `68` `923b819` Harden Qwen checkpoints, speculative decode and tool calls
- `69` `efc27d6` Keep literal reasoning and tool tags inside Qwen arguments

## Skip: Metal batched decode and speculative batch features

The batched Qwen decode stack and the DSpark speculative work on Metal. Feature work on
a backend this deployment does not execute.

- `27` `9ba464d` Price the batched cycle kinds after eight drafted cycles, not sixteen
- `28` `3077786` Decide per cycle whether a speculative batch drafts
- `29` `548cdaf` Bound speculative batches and invalidate failed predictor state
- `30` `e2cba0b` Batch the predictor layer across a speculative batch's sessions
- `31` `d569d05` Run the speculative batch's recurrent and attention steps as rows
- `32` `a504136` Speculate over a batch of Qwen3.8 sessions
- `33` `467b6f0` Harden Qwen Metal batch integration and quality checks
- `34` `9e91bac` Keep prefill chunks on the generic F16 tile
- `35` `b4da720` Count a prefill chunk's token tiles in the k-split rule
- `36` `65d2eb3` Replace the shared session arena when a wider session finds it idle
  (generic arena policy, but landed for this cluster; only matters if a conflict drags it in)
- `37` `643d4cb` Give the batched output head the Q8 tile
- `38` `891c005` Keep verify rows on the few-row matvec
- `39` `a154af7` Walk both rows of a grouped down simdgroup together
- `40` `3c0cfc3` Specialize the grouped MoE down pass on its pair count
- `41` `54a39d7` Specialize the grouped MoE mid pass on its pair count
- `42` `d36dd5d` Allocate the k-split scratch once
- `43` `8378039` Multiply Q8 decode batches on fp32 simdgroup matrices
- `44` `fe45751` Group a decode batch's routed experts and run its shared expert densely
- `45` `2c5b1aa` Keep F16 decode-batch projections on the fp32 tile
- `46` `4e01ef9` Take F32 decode-batch projections through the tiled GEMM
- `47` `3e068bb` Batch Qwen3.8 attention rows into one dispatch per stage
- `48` `6efb642` Report the logit distance when a batched verify fails
- `49` `ba3b0c2` Batch Qwen3.8 Flash Next decode natively across sessions
- `50` `147b263` Add a session-concurrency benchmark for the engine and the server
- `51` `d0b7434` Read a decode token's n-gram rows concurrently
- `147` `bbec200` Finish deferred Metal TP residuals before mixed-batch output heads
- `148` `5806d09` Batch Metal TP DSpark experts and reuse MXFP4 prefill activations
- `149` `be76012` Batch confident DSpark seeds on M5 and back off on weak drafts
- `150` `8d309ab` Capture partial cache frontiers for six-token DSpark verification

## Skip: CUDA and DSpark tuning

DGX Spark and CUDA decode work. Nothing here runs on the R9700.

- `128` `2803820` Use aligned Q8 vector kernels for small Spark batches
- `129` `45f54e9` Use aligned expert vector kernels for small Spark batches
- `131` `f6c169f` Retain all five CUDA DSpark partial prefixes
- `132` `df80a87` Batch DSpark seed tokens on resident DGX Spark
- `135` `b42682f` Reuse aligned Q8 scratch during CUDA decoding

## Skip: docs, QA, eval and test bookkeeping

Two of these carry information worth reading even though the commits themselves are
skippable. `31d1992` records an outstanding full GLM server stall, an open bug in a
model we serve. `6e4c285` records CUDA SSD prefill regression checks, which is a
template for ours.

- `1` `0aaea5a` Document physical Metal TP QA preflight and native batch coverage
- `3` `06f7135` Keep Metal-only Qwen batch tests out of the CUDA suite
- `6` `aafc65b` Bound loopback packet size in the tiny-buffer TCP regression
- `11` `25334fe` Test numeric reasoning strings at the API boundary
- `12` `ecf7a93` Enhance reasoning effort parsing logic
- `15` `62867e9` Make JPEG regression failure diagnostics safe
- `25` `d1620ba` Reject incomplete concurrency benchmarks and oversized requests
- `53` `bb6a748` Keep release QA focused on repeatable checks
- `54` `442982d` Update Qwen Spark performance references
- `58` `63b5dca` Add official Qwen quality references and validated API scoring
- `65` `15e88cf` Link CUDA regression tests with their shared helpers
- `66` `948307e` Keep Qwen steering capture compatible with Python 3.9
- `72` `6e4c285` Record CUDA SSD prefill gains and regression checks (read for the template)
- `79` `74fbe54` Add vision client and compaction recovery release checks
- `83` `f62ca29` README: intro section updated
- `88` `78c522e` Keep DSpark EOS fixture compatible with pre-fix source
- `90` `bff84ae` Adapt DSpark EOS regression fixture to current scheduler helper
- `94` `85a4f0e` tests: validate complete score tables and frontier logits
- `98` `90aa4b5` Document full GLM SSD append gains and validated coding QA
- `102` `31d1992` Record Metal SSD cache gains and the outstanding full GLM server stall
  (read for the open bug)
- `112` `54fc4f6` Document measured Metal SSD gains and scoped coding-session QA
- `116` `aa8fef4` Require real server coding sessions and prefix reuse checks in QA
- `117` `46e46e7` Extend tool-call QA with speculative boundaries and real coding clients
- `118` `42da390` Run frontend regressions without loading a model
- `127` `aa1e71c` Document DGX Spark DSpark speed and focused QA
- `133` `9ab7053` Record long GLM quality controls and Spark decode gains
- `136` `189ff0b` Add long GLM FP8 references with reasoning-aware scoring
- `137` `eb847ac` Fix official-score logprob parsing for null API responses
- `138` `f4d03f6` Record backend QA results and preserve failure diagnostics
- `143` `6c25bba` Put common getting-started instructions in the README
- `144` `2f0bc1e` Split README into platform setup and focused reference guides
- `153` `29aa726` Extend ds4-eval with hard capability suite

## Skip: agent and terminal UI

- `78` `6289c51` Give agent hints a distinct terminal label
- `80` `e33a5b2` Keep agent tasks running across context compaction
- `82` `6546ae5` agent: add opt-in programming hints
- `87` `d12898f` Initialize the terminal escape buffer for GCC dataflow checks
- `92` `b36e26e` Make agent file tools and background shell jobs reliable
- `93` `db19421` Fix agent terminal redraws, Unicode input and Markdown rendering

## Merge commits

No independent verdict. What matters is the content, which is listed above.

- `4067320` Merge remote-tracking branch 'pr-review/765'
- `0edbd9c` Merge upstream/main: vision live-state accounting in the reuse probe
- `f06d3eb` Merge Qwen3.8 Flash Next support from PR #991 (content is the skip cluster)
- `5ad935d` Merge PR #996: validate saved quality scores and frontier logits
- `909d656` Merge PR #992: ROCm gfx1151 prefill and DSpark improvements (content is the
  conditional group)
- `469ed2f` Merge upstream/main: fold multimodal live-KV hardening into the reuse probe

## Suggested intake order

1. Baseline: rebuild `make rdna4`, run the gates, record the numbers against
   `docs/PERFORMANCE_REFERENCE.md` (this is `docs/UPSTREAM.md` step 1).
2. Merge `origin/main` on an intake branch. Expect conflicts in `Makefile`, `ds4.c`,
   `ds4_server.c`, `rocm/ds4_rocm_runtime.cuh` and nowhere else.
3. Resolve the take groups first, `9d9e129` against our `c229363` and `3d53ea9` by hand.
4. gfx1151 is decided (skip, R9700-only). The v4.1 code merges as-is; the only open
   question is whether to port it to ROCm later, which is not part of this merge.
5. Gates, then deploy with the previous binary saved, as `docs/UPSTREAM.md` step 4 and 5
   describe.
