# Measured performance, live knobs and failure modes

Reference numbers for the R9700 streaming work, the environment variables that change its
behaviour, and the failure modes that were diagnosed along the way. Recovered from the
development transcripts of 2026-09-04 and 2026-09-17, both in
`~/.pi/agent/sessions/--media-NVME_DATA-MOUNTS-Models-ds4--/`, cross-checked against the tree.

What this file is not:

- **Not the chronological record.** `ROCM_STREAMING_PLAN.md` is that, including the accepted
  and rejected changes in order. This file is the cross-cutting reference those runs produced.
- **Not the deployment guide.** `DS4_ENGINE.md` covers the service, the RAM disk and tuning.
- **Not a claim about current defaults** unless a line says *live*, which means it was verified
  in the tree. Measurement numbers carry the session they came from.

Attribution matters here, so: everything below is **ds4** unless it is in the last section, and
the last section exists because the same transcripts also contain **qwen-flash** numbers that
read like ds4 numbers.

## Machine characterisation

Measured on this host (Sep 4), and not previously written down:

| Property | Value | Consequence measured |
|---|---|---|
| GPU | Radeon AI PRO R9700, 32 GiB, `gfx1201` | — |
| GPU host link | PCIe 3.0 x16 on the X299 root | transfer-bound streaming, hence the SDMA and prefetch work |
| ReBAR | 256 MiB | limits how much model can be mapped in one piece |
| XNACK | disabled, `BASE_PROFILE`, no coherent host access | host and device memory are genuinely separate |
| CPU | i9-7960X, 16 physical cores / 32 threads | 32 logical threads collapse decode to **1.99 t/s**; 16 physical threads is the correct setting |
| Host RAM | 128 GiB DDR4-2800 quad channel | why the 90 GiB tmpfs fits |
| Batch threads | 32 collapsed prefill | strict physical-core pinning raised repeat prefill **103.3 → 108.9 t/s** |
| CPU sampling cost | ~149 ms over 384 tokens against ~52.4 s of inference | sampling is not the bottleneck |

## The frontier at 262K allocated, RAM-backed, Vision-Exp

Steady decode and prefill against populated context (Sep 4/17, three paired runs where noted):

| Populated context | Prefill | Steady decode |
|---|---:|---:|
| 8K | 114.29 t/s | 8.60 t/s |
| 16K | 114.03 t/s | 8.05 t/s |
| 32K | 111.08 t/s | 6.74 t/s |
| 64K | 102.55 t/s | 7.08 t/s |
| 64K (second round) | 106.11 t/s | 7.41 t/s |
| 131K | 99.59 t/s | 7.06 t/s |

Decode variation between neighbouring contexts (6.74 at 32K against 7.08 at 64K) tracks the
expert route mix of the replay, not a smooth curve, so single runs at one context are not a
trend. A separate no-tuning round recorded 8K 114.88/8.61, 16K 113.32/8.36 and a short-prompt
smoke decode of 9.14.

## Accepted changes and what they measured

| Change | Measurement |
|---|---|
| Ram-backed prefill and first token (`35bacb1`) | 8K prefill 109.82 → **174.07 t/s**; 16K 108.70 → **171.06 t/s**; short-prompt TTFT 6.21 → **5.00 s** |
| Decode through pinned expert pointers (`78dce0e`) | 8K median 8.36 → **10.28 t/s** (+23.0%); 16K 7.90 → **9.68 t/s** (+22.5%); prefill flat at ~113 t/s |
| Double-buffered worker uploads | 8K median 10.19 → **10.30 t/s** (+1.1%) |
| Combined at the time | 8K 10.15/10.24/10.07 → 10.54/10.50/10.18; 16K 9.48/9.51/9.47 → 9.74/9.71/9.75 |
| Paired-row prefill and tmpfs bulk uploader | Prefill 109.13 → 113.83 t/s at 8K, 109.23 → 114.81 t/s at 16K, with saved top-20 logprob JSON equal at 12,583 tokens |
| Model-range cache accounting fix | No measured gain (10.18 t/s both); kept for correctness |

Prefill chunking was a real trade, not a free win: 4K chunks ran ~111 t/s while 8K chunks
reached ~153–155 t/s, but the larger chunk left less expert residency afterwards, so managed
KV with 4096-token chunks was kept.

## Configurations that were rejected, and the number that killed them

- **Model-span limits below 10 GiB** collapsed decode to **0.47–0.48 t/s**.
- **A 14.62 GiB expert cache (2218 experts) with 5.86 GiB of context buffers and a 3.38 GiB
  prefill reserve** produced a logged short decode of only **~1.09 t/s**; older smaller-context
  runs of the same kind sat at 2.2–2.9 t/s.
- **Re-enabling the one-token resident/missing split path** reached **~11.8 t/s** but produced
  highly nondeterministic logits, and synchronising it removed the gain. Rejected.
- **64K decode tuning by shrinking the cache** bought 4.73 → 4.86 t/s while expert residency
  fell 13.8 → 8.9 GiB. Rejected as a default.
- **Decode-only tuning** (22 GiB expert target, 4 read workers) reached ~9.87 t/s against
  ~12.1–12.2 t/s for the older deployed binary on the same workload, so it was not adopted as
  an improvement. Buffered versus direct I/O, and the GPU compute profile, both showed nothing.
- **A conservative reserve guard** rejected ubatch 512 in the earlier work; the later note that
  this was worth reconsidering (about +7.5% prefill) belongs to the qwen work — see below.

Cache defaults that were retained deliberately, because changing them traded away residency:
18 GiB expert target, 16 GiB model spans, 2 GiB free-VRAM reserve, four read workers.

## Known failure modes and their workarounds

1. **SDMA write into a trimmed expert slab** raises an HSA assertion. `HSA_ENABLE_SDMA=0`
   works around it at about **8.70 t/s** against 9.74 with SDMA. Any runtime fix should target
   stability while keeping 9.74, not attempt the ~10.8 that was hoped for.
2. **The decode-stage profiler must not be combined with selected pointers** for accepted
   performance numbers; that combination has a known failure.
3. **`DS4_ROCM_ENABLE_STREAMING_PREFILL_BATCH_SELECTED_LOAD` stays off** — the unsafe prefill
   path. Its `DISABLE_` twin exists because disable wins over enable.
4. **Cache sizes must stay above the model-span floor.** Decode does not degrade gracefully
   when the span budget drops under ~10 GiB; it collapses to under 0.5 t/s.
5. **Do not equate the 1,529-token replay with the original 8,920-token request** when quoting
   server numbers: the replay is a fixture, and warm HTTP behaviour is not what the CLI gains
   promise.

## Live knobs that change streaming behaviour

Verified in the tree; none of these were in the documentation before. `Getenv` names are given
with the default the code uses when unset.

| Variable | Default | Effect |
|---|---|---|
| `DS4_ROCM_STREAM_FREE_RESERVE_GB` | 16 GiB, but **2 GiB on the 32 GiB discrete profile** | free-VRAM reserve kept while growing the cache; the production unit sets `2` explicitly |
| `DS4_SSD_AUTO_CACHE_PCT` | 80 (accepted range 50–95) | expert cache size as a percentage of the working set; production runs `65` |
| `DS4_ROCM_STREAM_READ_WORKERS` | capped at the compile-time maximum `DS4_ROCM_N_EXPERT_USED * 3`; `0` means 1 | read/upload worker count; 24 was used in one transfer experiment |
| `DS4_ROCM_STREAM_MODEL_CACHE_GB` | derived | override the model-span cache budget in GiB |
| `DS4_ROCM_STREAM_Q8_F16_CACHE_GB` | derived | override the q8→f16 cache budget in GiB |
| `DS4_ROCM_STREAM_EVICT_PAST_LAYERS_FIRST` | off; any value other than `0` enables it | eviction order for the expert cache: default is LRU, this switches to preferring layers already passed in the forward pass |
| `DS4_ROCM_STREAM_CACHE_STATS`, `DS4_ROCM_STREAM_CACHE_LAYER_STATS` | off | cache hit/miss accounting, per layer for the latter |
| `DS4_ROCM_STREAM_READ_PROFILE` | off | read timings per job |
| `DS4_SERVER_DECODE_PROFILE` | off | server-side decode counters (sampling against evaluation time) |
| `DS4_ROCM_LAYER_STAGE_PROFILE`, `DS4_ROCM_DECODE_STAGE_PROFILE` | off | per-stage decode timing; see failure mode 2 before combining with selected pointers |
| `DS4_ROCM_ENABLE_* / DS4_ROCM_DISABLE_*` pairs | disable wins; `ENABLE=0` suppresses the automatic choice | the streaming sub-features (read pipeline, decode overlap, selected pointers, model-parallel read, q8 prefill pair) |
| `DS4_ROCM_DECODE_TRACE`, `_SKIP`, `_TOKENS` | off, 64, 8 | documented in `DS4_ENGINE.md` §3.7 |

Two experiments used variables that no longer exist in the tree, and are recorded here only so
that old logs make sense: `DS4_ROCM_STREAM_MODEL_ARENA_MB` (128/256/512/1024 sweep) and
`DS4_ROCM_ENABLE_DECODE_MODEL_RESET` / `DS4_ROCM_DISABLE_DECODE_MODEL_RESET` (the prefill
model-range reset that later landed as `c229363`).

## Evidence inventory, and why it is fragile

The raw evidence for these numbers is on disk, and almost none of it is versioned: `misc/` is
gitignored (`.gitignore:42`), so the whole 1.3 GB is invisible to git and one `git clean -x`
away from being lost.

| Path | Size | Contents |
|---|---:|---|
| `misc/rocm-vision-profile-20260917/` | 398M | vision profiling runs, the cache experiment patch and simulation |
| `misc/rocm-local-runtime/` | 311M | the first local ROCr build tree |
| `misc/rocm-timeline-20260918/` | 236M | timeline traces, `amd_blit_sdma` sources, PCIe topology |
| `misc/rocm-local-runtime-fixed-source/` | 192M | the source of the ROCr build that production loads |
| `misc/rocm-local-runtime{fixed-,-}build/`, `…/{fixed-,}prefix/` | 36M, 36M, 7.1M, 7.0M | build trees and the loadable prefixes |
| `misc/rocm-prefetch-20260919/` | 14M | prefetch experiments with per-run `results.json` |

Ephemeral but currently live: `/tmp/ds4-generation-perf/` (frozen baselines, `replay.py`, the
arena sweep), `/tmp/ds4-deeper-perf/`, `/tmp/ds4-next-perf/` (including `final-verified`),
`/tmp/ds4-perf-tests/`. These are the only copies of several baselines and diffs.

If any of this is meant to survive a reboot, the cheap subsets are the `results.json`,
`metadata.json`, `provenance.json` and small log files; the build trees are reproducible from
the scripts and only need their commit and flag list recorded.

## Numbers in the same transcripts that belong to qwen-flash, not ds4

Both sessions ran work on `../qwen-3.8-flash` in the same directory context, so these look like
ds4 numbers and are not:

- The "74 production requests" long-context cost model: 45,579 tokens → 8.4 min prefill at
  90.7 t/s and 15.44 t/s decode; 136,627 → 29.2 min at 77.9 t/s and 12.37 t/s; 205,386 →
  43.3 min at 79.0 t/s and 10.06 t/s; 209,480 → 46.3 min at 75.4 t/s and 9.52 t/s.
- The compaction cost: a 136,627-token prompt at 77.9 t/s for 29.2 min, then a 7,553-token
  summary at 12.4 t/s for 10.2 min, 39.4 min in total.
- Prompt caching: a second identical request reused 5,477 tokens in 0.19 s instead of 51.6 s,
  266× less re-prefill.
- `ubatch 512` (+7.5% prefill, 114.2 against 106.2 t/s, peak 30,294 MiB), the IQ4 placement
  sweep (32 CPU MoE layers 5.07 t/s … 22 layers 9.30 t/s … 477 tokens sustained 10.54 t/s) and
  the MTP work (depth 1, 18.3 t/s with MTP against 7.6 without).

Those belong in the qwen project's own `RESULTS.md`, `EVIDENCE.md` and `PROVENANCE.md`, which
already exist beside its scripts.

## Open items

- The expert-arena sweep was single-run, and the phase-reset experiments were validated only
  when they became `c229363`; treat the sweep numbers as screening, not as a recorded frontier.
- Model-span budget behaviour between 10 and 16 GiB was never mapped; only the collapse below
  10 GiB is measured.
- The vision numbers (wall 94 → 34 s, image prefill 3.07 → 10.60 t/s, generation 2.98 → 6.30
  t/s after moving from disk-backed to RAM-backed serving) come from one session and one image;
  they are a direction, not a benchmark.
