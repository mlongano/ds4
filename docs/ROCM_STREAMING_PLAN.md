# R9700 streaming improvement plan

Initial baseline: `78dce0e`, RAM-backed Vision-Exp IQ2_XXS gate / Q2_K down,
262144 allocated tokens, 4096-token prefill chunks, managed KV.
Keep the server stopped. Do not push or alter unrelated files.

## Acceptance gates

- Build with `make rdna4` and ROCm include/library paths.
- Compare complete saved top-20 logprob JSON, not just generated text.
- Test a coding prompt, a long prefill, and an image continuation.
- Run model processes sequentially. Record system-wide swap deltas and elapsed
  time. Compare first-token latency separately from steady decode.
- Test each optimization independently against a saved baseline binary.
- Keep an optimization only after correctness passes and repeated benchmarks
  show a benefit. Record rejected experiments rather than enabling them.

## Work

1. **Constrain direct-pointer policy.** Automatic selection must require
   gfx1201, discrete 30–36 GiB memory, DeepSeek streaming, and the tested
   IQ2_XXS/Q2_K quantization. Disable must override both enable variables.
   Add CPU-only policy tests.
2. **Add reproducible tests.** A sequential real-model runner will save logs,
   complete logprob comparisons, benchmark CSV rows, and swap counters outside
   the source tree. Include configurable long-prefill coverage.
3. **Avoid redundant cache eviction.** Account only for uncached model spans
   before deciding to flush. This is a bounded change; separating permanent
   dense arenas from temporary prefill arenas is a larger follow-up if cache
   traces show that duplicate accounting is not the main cost.
4. **Pipeline worker uploads.** Experiment with two pinned staging buffers per
   worker. Do not mark jobs complete until their uploads finish; drain every
   outstanding upload on error and shutdown. Compare separately with step 3.
5. **Overlap resident decode work.** Experiment only after upload lifetime
   checks. Keep pointer-table publication ordered and wait for missing data
   before its kernels. All-hit/all-miss cases must use the same pointer path.
6. **Prefill and residency follow-up.** Do not enable prefill overlap with
   decode-only pins. Batch pinning, phase-specific scratch ownership, and
   eviction policy changes need separate lifetime tests and measurements.

## Results

Steps 1–5 are implemented and tested. Step 3 fixes duplicate accounting but
showed no throughput gain on this workload; it does not separate dense and
prefill arenas. Step 6 remains follow-up work, not an enabled optimization.

### Measured performance

All rows use the same RAM-backed model, 262144 allocated context, 4096-token
prefill chunks, 128 generated tokens and sequential reference/candidate runs.
The reference binaries were saved from `78dce0e` before editing source.

| Experiment | 8K reference steady t/s | Candidate steady t/s | Notes |
|---|---:|---:|---|
| Cache accounting and narrowed policy | 10.18 | 10.18 | Median of 3, no measured gain |
| Double-buffered uploads alone | 10.19 | 10.30 | Median of 3 |
| Resident decode overlap alone | 10.15 | 10.28 | Median of 3; one candidate run had heavy swap activity and reached only 8.83 |
| Combined | 10.15 | 10.50 | Median of 3, +3.4% |
| Combined at 16K | 9.48 | 9.74 | Median of 3, +2.7% |

Combined 8K steady samples were 10.54, 10.50, 10.18; paired references were
10.15, 10.24, 10.07. Combined 16K samples were 9.74, 9.71, 9.75;
references were 9.48, 9.51, 9.47. No samples were excluded from these medians.
Prefill and first-token latency were effectively unchanged.

A final no-tuning-flags default check measured 10.43 versus 10.10 steady t/s.
An unsteered 512x507 image continuation measured 7.04 versus 6.92 generation
t/s, with identical output bytes and 9.18 prefill t/s on both paths. This is
one image measurement, not a repeated vision speed claim.

### Correctness and lifetime checks

- RDNA4 build and `tests/test_deepseek4_vision_image` pass.
- Each separate experiment, the combined experiment and the automatic-default
  build matched the complete top-20 logprob JSON for two short prompts and a
  12583-token prompt, each followed by 128 greedy decode steps.
- Two-turn CLI transcripts and image continuations matched baseline stdout;
  the runner also checks that the expected generations completed.
- CPU policy tests reject unvalidated hardware and all-Q2/IQ2-IQ2 arithmetic,
  and check disable precedence and explicit zero values.
- The production pipeline worker is compiled against a delayed-copy CPU model
  with AddressSanitizer and UBSan. It passes 1–4-worker tests with 1, 2, 3, 17
  and 257 jobs, profiling on/off, and injected read/enqueue/sync errors.
- Comparison-gate tests reject empty output, alternative-logit drift,
  truncation and nonfinite numbers.

The tests do not prove full-vocabulary equality, real GPU recovery after a
hardware fault, session cancellation, KV snapshot reload, or correctness on
other GPU families. Those remain explicit coverage gaps. CPU mock errors are
not a substitute for GPU fault-recovery tests.

### Defaults and diagnostics

Automatic transfer pipelining requires DeepSeek SSD streaming on discrete
gfx1201 with 30–36 GiB VRAM. Automatic direct pointers and resident decode
compute overlap additionally require IQ2_XXS gate / Q2_K down. Other arithmetic
paths remain compact. Prefill shared/upload overlap remains disabled.

Diagnostic switches, where value `1` disables the named optimization:

- `DS4_ROCM_DISABLE_STREAMING_DECODE_SELECTED_PTRS`
- `DS4_ROCM_DISABLE_STREAM_READ_PIPELINE`
- `DS4_ROCM_DISABLE_STREAMING_DECODE_OVERLAP`

Corresponding `ENABLE_...` switches support explicit experiments. Disable
wins when both are set; explicit enable value `0` suppresses automatic use.
The old `ENABLE_STREAMING_DECODE_SELECTED_SPLIT` remains a direct-pointer alias.

### Reproduction

CPU-only unit tests:

`make test-rocm-streaming-unit`

Real-model correctness, two-turn and vision tests against compact arithmetic:

`python3 tests/rocm_streaming_regression.py --reference-compact --skip-bench --interactive --vision gguf/DeepSeek-V4-Flash-Vision-Encoder.gguf`

Before editing source, save the reference CLI and benchmark binaries outside
this tree. Compare candidate changes with those binaries, not just with flags
that may leave a shared optimization enabled in both processes:

`python3 tests/rocm_streaming_regression.py --reference /tmp/ds4-next-perf/ds4-baseline --reference-bench /tmp/ds4-next-perf/ds4-bench-baseline --contexts 8192 16384 --repeats 3`

The runner creates a new `/tmp/ds4-regression-*` directory by default and saves
commands, binary hashes, experiment flags, logprob JSON, per-run CSV rows,
elapsed time and system-wide swap counters. Swap counters include other
processes, not just the model. Pipelined upload timing includes overlapped
reads and must not be added to read time as if the stages were serialized.
Local evidence for this implementation is under `/tmp/ds4-next-perf/` in
`cache-tests`, `pipeline-tests`, `overlap-tests`, `combined-tests`,
`default-tests`, `transitions`, and `final-verified`. The final rebuilt binary
passed all short/long logprob, two-turn and image checks in `final-verified`.
These temporary artifacts are not committed.

### Remaining work after the initial decode changes

Persistent dense allocations and temporary prefill ranges still share arenas.
Separate their ownership before changing arena eviction or scratch release.
Add batch-specific pins before enabling prefill overlap. Test cancellation,
recovery and long-context continuation beyond 16K before expanding the claims.
No new cache-size defaults were selected.

## Completed prefill and TTFT follow-up

Reference binaries for this phase were saved from `df7cc08` in
`/tmp/ds4-deeper-perf/` before editing. The RAM-backed Vision-Exp model,
262144 context allocation, managed KV, 4096-token chunks and cache budgets
remain unchanged. The service stayed stopped; no production settings changed.

### Accepted changes

- **Parallel model-span uploads.** Large prefill/static spans now use the
  existing four-worker read/upload pool in bounded batches of up to 64 jobs,
  each at most 8 MiB. Previously these reads used the single caller thread.
  Allocate the destination first, wait for every upload, then publish the
  range. Failed batches also drain before returning. Selected-expert uploads
  retain their separate uploader while they own the pool.
- **Paired-row Q8 prefill.** Two output rows reuse each LDS activation load.
  The selected shape is 16 tokens, 32 rows and 512 threads using 32 KiB LDS.
  The original per-lane FP32 accumulation and reduction order are retained.
  Smaller launch tiles alone were slower and were rejected. Exploratory
  tile/row/block switches were removed.

Automatic paired-row selection requires DeepSeek streaming, discrete gfx1201
with 30–36 GiB VRAM, and at least 256 tokens. Automatic bulk uploads use the
same hardware/model gates but are independent of token count. They require a
tmpfs model descriptor and a span of at least 16 MiB. Disk-backed
model-span I/O retains its existing default. Neither change enables prefill
shared/upload overlap or changes KV/cache ownership.

Independent tests before combining the changes:

| Experiment | 8K prefill reference → candidate t/s | 16K prefill reference → candidate t/s |
|---|---:|---:|
| Paired rows only, median of 3 | 109.13 → 113.83 | 109.23 → 114.81 |
| Parallel spans only, median of 3 | 109.19 → 161.76 | 107.97 → 157.89 |

An initial single parallel-span pair had 10.39 → 9.70 steady decode t/s.
That slowdown did not reproduce in six later isolated pairs or six combined
pairs. Its cause is not established; it is not attributed to swapping without
evidence. All runs remain in the artifacts.

### Final automatic-default results

Medians of three alternating reference/candidate pairs, 128 generated tokens:

| Measurement | Reference | Candidate |
|---|---:|---:|
| 8K prefill t/s | 109.82 | 174.07 |
| 16K prefill t/s | 108.70 | 171.06 |
| 8K first-decode ms | 2259.299 | 1092.376 |
| 16K first-decode ms | 2255.931 | 1096.503 |
| 8K steady decode t/s | 10.45 | 10.44 |
| 16K steady decode t/s | 9.75 | 9.78 |
| 8K overall generation t/s | 8.86 | 9.65 |
| 16K overall generation t/s | 8.35 | 9.09 |

Prefill improved 58.5% at 8K and 57.4% at 16K. There is no meaningful
steady-decode gain. Overall generation improves because first decode is shorter.

The separate `--ttft` test timestamps process launch through the first generated
stdout byte of the noninteractive `--nothink` CLI. It includes startup/model
loading and prefill, not just first decode. It is a cold-process TTFT proxy,
not a warm HTTP-server measurement or a cold-storage benchmark.

| Cold CLI first-output time, median of 3 | Reference | Candidate |
|---|---:|---:|
| Short binary-search prompt | 6.205 s | 4.996 s |
| 12583-token prompt | 135.997 s | 84.099 s |

Reductions are 19.5% and 38.2%. Every TTFT pair produced identical nonempty
stdout and completed generation. No timing samples were excluded. The final
suite recorded 2571 swap-in pages, 96855 swap-out pages and 3981 major faults
system-wide, using 4096-byte pages. Earlier noisy runs remain recorded: the
paired-row 78.50 t/s sample had 1325282 swap-in pages, and an isolated baseline
at 52.00 t/s had 6123609 swap-in pages. Tmpfs is not guaranteed physical residency.

### Final validation

- RDNA4 build, vision image unit test, policy tests and ASan/UBSan tests pass.
- Exact GPU output comparisons cover partial token/row/K tiles, 1–257 token
  boundaries and 4096-token production projection shapes. Outputs are poisoned
  before each kernel; every value must be finite and byte-identical.
- The actual bulk partitioning function is extracted verbatim for CPU tests.
  Tests cover byte boundaries, multiple batches, failed batches and missing
  upload completion. Existing delayed-copy worker tests cover read, enqueue
  and synchronization failures. These are not real GPU fault-recovery tests.
- Final defaults match all saved top-20 logprob JSON for the 42-, 25- and
  12583-token prompts, each followed by 128 greedy steps. Disabling both new
  optimizations also matches the saved baseline for 64-step continuations.
- Two-turn and image continuations match baseline stdout. Separate scale-3
  steering checks pass for coding logprobs and image continuation output.
- Nine Python tests cover comparison acceptance and first-output measurement,
  including empty output, delayed completion, large output, errors and timeout.

The decode stage profiler failed with a missing-compact-experts diagnostic
when used with selected-pointer execution. It was not used for performance
acceptance. The prefill stage trace remains useful, but synchronized stage
profiling is not an unprofiled end-to-end benchmark.

### Diagnostics and reproduction

Disable switches, with corresponding `ENABLE_...` overrides:

- `DS4_ROCM_DISABLE_Q8_PREFILL_PAIR`
- `DS4_ROCM_DISABLE_MODEL_PARALLEL_READ`

Nonzero disable wins; explicit enable `0` suppresses automatic selection.
Explicit enables allow controlled tests beyond automatic hardware/file gates.
Bulk uploads still cannot use the pool while selected uploads own it. The
experimental names without `ENABLE_` and the tile-sweep flags are no longer
supported.

CPU tests: `make test-rocm-streaming-unit`

GPU tests after an RDNA4 build: `LD_LIBRARY_PATH=/opt/rocm/lib LIBRARY_PATH=/opt/rocm/lib make test-rocm-q8-prefill`

Final correctness, vision, TTFT and performance suite:

`python3 tests/rocm_streaming_regression.py --reference /tmp/ds4-deeper-perf/ds4-baseline --reference-bench /tmp/ds4-deeper-perf/ds4-bench-baseline --contexts 8192 16384 --repeats 3 --interactive --vision gguf/DeepSeek-V4-Flash-Vision-Encoder.gguf --ttft`

Evidence is in `/tmp/ds4-deeper-perf/`: `pair-real`, `parallel-map-real`,
`parallel-repeated`, `final-combined`, `final-disabled`, `steering-final-v2`,
synthetic kernel CSVs and build/unit logs. Binary hashes, commands, per-run
measurements and swap deltas are retained. Temporary artifacts are not committed.

Further work is separate from these accepted changes: persistent dense arena
ownership, warm-server and KV-reuse TTFT, cancellation/snapshot recovery,
batch-prefill pins, and correctness/performance beyond the tested contexts.
Other GPU families and CUDA/Metal hardware were not tested in this phase.

## Sampled server generation follow-up

Baseline: `35bacb1`. The user prioritized generation throughput over TTFT.
The original service request used thinking, tools, vision residency and FFN
steering scale 3. Its approximately 7.3 t/s decode was not represented by the
older greedy CLI frontier benchmark.

### Diagnosis and accepted change

A fixed-seed server replay reproduced the slower generation. CPU sampling
accounted for only 149 ms out of 52.6 seconds. Only 6.10 GiB of experts stayed
resident despite the configured cache target; 48370 of 99072 selections missed.

The model-span cache can retain prefill weight ranges while adding decode
weights if their combined size fits its 16 GiB limit. Those prefill ranges
then occupy VRAM for the entire decode, displacing independently cached experts.
This depends on where the last prefill leaves the cache, so an 8K benchmark can
avoid the problem while a shorter prompt or tool-result suffix triggers it.

The accepted fix adds an explicit ROCm phase boundary after prefill and expert
seeding, before the output head. It synchronizes GPU work and retires model-span
allocations. Normal output/per-layer or static mapping then reloads decode
weights. It does not release KV, independent expert slabs or graph buffers.
If selected uploads or read-pool jobs are active, it leaves the cache untouched.
Automatic selection is limited to DeepSeek streaming on discrete gfx1201 with
30–36 GiB VRAM; Metal, CUDA and GLM do not take this path.

ROCm's static decode map is opt-in. The first experimental hook was mistakenly
placed only in that branch; the accepted hook covers both mapping modes.

In a profiled reset-only pair, expert residency rose to 13.83 GiB, misses fell
to 30694, and inference time fell to 42.14 seconds. Generated fields matched
exactly. Smaller model arenas alone offered about 5%, but added less than 1%
on top of the reset in screening. That tuning and its experimental environment
variable were removed. Allocation sizes and all cache budgets remain unchanged.

### Repeated server measurements

Three unprofiled alternating pairs, fresh process per sample, fixed seed 123,
temperature 1, top-p 1, 384 output tokens, thinking/tools enabled, encoder loaded,
steering scale 3, 262144 allocated context and managed KV:

| Measurement | Reference median | Candidate median |
|---|---:|---:|
| Overall server generation | 7.15 t/s | 8.99 t/s |
| Generation after first logged 50-token interval | 7.17 t/s | 9.19 t/s |
| Generation duration | 53.689 s | 42.699 s |

The overall gain is 25.7%; after the initial interval it is 28.0%. Individual
overall rates were 7.26/7.10/7.15 versus 8.99/9.09/8.90. No samples were
excluded. The six processes recorded 135 swap-in pages, 4 swap-out pages and
197 major faults system-wide, using 4096-byte pages.

The frozen replay contains the saved user/assistant/tool-result history and
representative system/tool definitions: 1529 input tokens. The original Pi
system prompt and schemas were not saved, so this is not an exact replay of its
8920-token request. A run that reread the growing live session instead sent
66796 tokens and failed the output gate; it is not comparable performance
evidence. Subsequent runs use the frozen request artifact exclusively.

All three pairs produced identical generated content, reasoning and tool
fields; transport fragmentation and random tool IDs are excluded. The requests
intentionally stop at the 384-token cap, not necessarily at a complete answer.
This does not establish full-vocabulary equality or a universal 10 t/s rate.

### Correctness and remaining limits

- The RDNA4 build and CPU sanitizer/policy tests pass. The actual reset body is
  tested for enabled/disabled hardware/model policy, active uploads and failed
  GPU synchronization; release must never precede a successful synchronization.
- Complete saved top-20 logprob JSON matches `35bacb1` for 42-, 25- and
  12583-token prompts, each followed by 128 greedy steps. The disable switch
  also matches for 64-step continuations.
- Two-turn chat, image continuation, and a separate steered image/text/image
  sequence match baseline stdout. The latter checks three completed generations
  and encoder use after earlier model-span retirement. The vision unit passes.
- A single 8K CLI pair measured 10.54 → 10.75 steady t/s; a single 16K pair
  measured 9.81 → 10.12. These are regression checks, not repeated speed claims.
- The service's reasoning/tool text buffering remains unchanged. Upstream
  `fe2d3b0` deliberately introduced it to handle repeated reasoning safely;
  deleting it would violate the existing streamed-reasoning regression test.
- Warm multi-request server performance, cancellation, GPU fault recovery,
  snapshot reload and other GPU families still need separate validation.

### Reproduction and diagnostics

`DS4_ROCM_DISABLE_DECODE_MODEL_RESET=1` restores the previous cache policy.
`DS4_ROCM_ENABLE_DECODE_MODEL_RESET=0` also disables automatic use; a nonzero
disable wins over enable. Explicit enable may select other hardware, never
active uploads or non-DeepSeek/non-streaming modes.

`DS4_SERVER_DECODE_PROFILE=1` reports sampling, inference, first inference and
other generation time. It is diagnostic; final performance pairs leave it off.
The existing ROCm cache/read profiles expose expert misses and transfer totals.

`make test-rocm-streaming-unit`

`python3 tests/rocm_server_regression.py --request /tmp/ds4-generation-perf/screen-v2/request.json --reference /tmp/ds4-generation-perf/ds4-server-baseline --vision gguf/DeepSeek-V4-Flash-Vision-Encoder.gguf --steering-file dir-steering/out/refusal_train400_en_it_plus20_nothink_ffn_out.f32 --tokens 384 --repeats 3`

The server runner binds localhost only, uses a new process per sample, never
starts/stops systemd units, and stores its working files under a new temporary
directory. Supply a frozen OpenAI request JSON; original private session data
and machine-specific replay fixtures are not committed. It saves binary hashes,
request JSON, SSE events, commands/environments, generated fields and swap
counters. Six parser/measurement unit tests cover fragmented output, tool fields,
missing output/usage, errors and generation timing.

Evidence: `/tmp/ds4-generation-perf/`, especially `screen-v2`, `reset-screen-v3`,
`final-server`, `final-cli`, `image-transitions`, `disabled` and build/unit logs.
The service was stopped with permission while holding the RAM model open, so
its stop hook could not unmount the RAM copy. Production settings are unchanged.

## Bounded long-context investigation, September 17 and 18, 2026

Reference: rebuilt `c229363`, with archived CLI, server and benchmark binaries
and hashes. No new inference optimization was accepted. Production settings,
managed KV, cache budgets and reasoning-stream buffering remain unchanged.
The bounded experiments below did not establish another 10–15% generation gain;
they do not prove that further optimization is impossible.

### Workloads and long-context baseline

The short replay was recovered from the original document-inspection session,
using its history as of the earlier screening run rather than later messages.
The server confirms 1529 prompt tokens. The original system prompt and schemas
were not saved; the same representative substitutes are retained.

A new frozen repository-review request supplies runtime/MoE source excerpts as
a tool result. It has 87441 actual prompt tokens, thinking/tools enabled,
temperature 1, top-p 1 and seed 123. This is a representative coding workload,
not an exact reproduction of the user's original conversation. Every server
sample loads the vision encoder and uses steering scale 3, 262144 allocated
context, managed KV, an 18 GiB streaming target and a 2 GiB free reserve.
The long request contains no image; encoder residency is not image validation.

Two fresh-process runs of the same baseline completed 1024 generated tokens
with identical content/reasoning/tool fields and no abort:

| Measurement | First run | Second run |
|---|---:|---:|
| Overall generation t/s | 7.91 | 7.95 |
| After first 50-token interval t/s | 8.002 | 8.043 |
| Request to first streamed output, seconds | 749.740 | 592.606 |
| System-wide swap-in pages | 3614825 | 12385 |
| System-wide swap-out pages | 217143 | 0 |
| System-wide major faults | 1080 | 12776 |

Pages are 4096 bytes. The first sample includes 13.79 GiB of system-wide
swap-in and is retained. Compilation also occurred during the baseline
prefills. Neither first-output sample is a clean latency comparison between
builds. Both generations reached the token cap while emitting reasoning;
ordinary answer text was not emitted. This validates repeatability over 1024
steps, not completion of an answer or freedom from longer-run runtime faults.

### Cost split

The existing stage profiler disables a selected-load path and cannot profile
this execution mode. A private diagnostic build under the evidence directory
instead preserves scheduling eligibility, fences the compute stream at stage
boundaries and adds GPU events around direct-pointer routed kernels. Its short
and long outputs match their baseline outputs exactly. It changes scheduling
and is not used for throughput acceptance.

| Diagnostic cost, ms/token | Short replay | 87441-token replay |
|---|---:|---:|
| Routed-MoE stage after first 50 positions | 65.16 | 56.54 |
| Host wait for missing experts, all positions | 52.35 | 44.00 |
| Routed gate/up plus down GPU events, all positions | 9.05 | 8.94 |
| Indexer-selection plus attention stages, after first 50 positions | 8.21 | 38.91 |
| Q projection, attention output and shared-expert stages, after first 50 positions | 18.60 | 18.67 |

These rows overlap and must not be summed. In particular, worker read/upload
elapsed times overlap each other and GPU work. Stage wall time also includes
host overhead. Expert data movement is the largest measured single cost;
long-context indexer/attention is a substantial second cost.

The short trace has 29415 misses out of 99072 selections, with 14.62 GiB of
resident experts. The long trace has 63490 misses out of 264192 selections,
with 14.57 GiB resident. Decode adds 202.06 and 426.67 GiB of read/upload jobs,
respectively, including decode-weight reloads. The retirement hook emits a
snapshot after each prefill chunk: subtract the **last** prefill snapshot,
not the first, to isolate decode. Long-context traffic is about 0.417 GiB per
generated token.

The GPU-facing link reports PCIe 5.0 x16, but its upstream bridge/root link
reports 8.0 GT/s x16, or PCIe 3.0 x16. This is a relevant transfer constraint,
not proof that the software already saturates the link.

### SDMA workaround measurement

Three unprofiled alternating pairs on the short replay, 384 output tokens,
compare explicit `HSA_ENABLE_SDMA=1` with `HSA_ENABLE_SDMA=0`. Generated fields
match in every pair.

| Median | SDMA on | SDMA off |
|---|---:|---:|
| Overall generation t/s | 9.54 | 8.44 |
| After first 50-token interval t/s | 9.742 | 8.699 |
| Request to first streamed output, seconds | 14.893 | 19.825 |

Disabling SDMA costs 11.5% overall and 10.7% after the first interval. Individual
steady rates are 9.746/9.742/9.723 versus 8.699/8.755/8.596. All samples remain
included, including 52188 swap-out pages in the final candidate. Totals across
the six processes are 124 swap-in pages, 52188 swap-out pages and 130 major
faults, system-wide.

ROCm remains 7.2.4. The known SDMA assertion did not recur in these runs, but
nothing here fixes it. SDMA-off was not subjected to the long-context/image
acceptance suite. No service environment was changed.

The server regression runner now accepts only `DS4_*` overrides and the added
`HSA_ENABLE_SDMA=0/1` override, validates them before launching processes, and
records the SDMA environment per sample. Four new unit tests cover inherited
settings, independent reference/candidate overrides, profiling defaults and
rejection of unrelated/invalid overrides.

### Rejected cache-policy trial

A separate private build tested frequency-based victim selection with counters
that survive eviction and halve every 128 tokens. Existing pin and
current-selection protections were retained. One short screening pair matched
output exactly but measured 9.742 versus 9.729 steady t/s. Misses fell only
from 29415 to 28882, or 1.8%. This is not a demonstrated speedup and was not
promoted to a long-context acceptance run.

A captured routing trace reproduces both measured miss counts in an offline
simulation at 2218 slots. Decay intervals of 0, 16, 32, 64, 128 and 256 tokens
yield 29805, 28411, 27370, 28234, 28882 and 29538 misses. The best is about 7%
below LRU on this one request, not a measured end-to-end improvement. No
frequency policy or tuning flag was added to the checkout.

Decode model arenas reserve 8.750 GiB and use 8.197 GiB in the short trial.
The 0.553 GiB difference does not support expecting several more GiB of expert
residency from smaller arenas alone.

### Validation and evidence

- RDNA4 rebuild, CPU streaming/policy/sanitizer tests and Q8 GPU exact tests pass.
- Long baseline A/A, both diagnostic comparisons and all SDMA pairs match
  generated fields. Server-runner unit coverage is now ten tests.
- No accepted engine change exists, so the full logprob/image-continuation
  acceptance suite was not rerun for the rejected cache experiment.
- No pushes, production configuration changes, model deletions or rebases.

Evidence is preserved in ignored `misc/rocm-vision-profile-20260917/`, copied
from `/tmp/ds4-vision-profile-c229363/`. It includes baseline binaries/hashes,
frozen requests/provenance, runner snapshots, logs, SSE/output JSON, swap
counters, private diagnostic/experimental source patches and build commands,
and the cache simulation. Private replay contents are not committed.

## Next bounded implementation pass

Status: bounded pass completed on 2026-09-18. One performance candidate reached
real-model screening and was rejected; no optimization was accepted. The second
candidate slot was left unused because attention and transfer measurements did
not show enough headroom. Profiling instrumentation does not count as a
performance candidate. The target remains a repeatable 10-15% steady generation
gain, not a promised outcome.

### 1. Asynchronous timeline

Implement an opt-in, bounded ROCm trace with GPU events on the existing compute
and upload streams and monotonic CPU spans for reads and waits. Do not insert
device/event synchronization, alter stream dependencies, or enable the old
stage-profiler flag. Resolve completed events after normal inference has
drained work. Report incomplete records, capacity exhaustion and trace errors;
never treat missing events as zero-duration work.

Keep disabled tracing inert. Restrict the hooks to ROCm streaming; preserve
selected-pointer execution, cache ownership and all other backend defaults.
Capture a bounded steady-decode window after the first 50 tokens. GPU events
share one GPU time origin; document uncertainty when aligning CPU timestamps.
Use synthetic tests for event ordering, incomplete/error paths, bounded storage
and disabled mode before real-model runs.

Done when: short and 87441-token traces preserve generated output and distinguish
GPU compute, uploads, CPU reads/waits and GPU idle intervals without additional
fences. Measure tracing overhead against an untraced run. Trace timings remain
diagnostic, not performance-acceptance numbers.

Implementation checkpoint (2026-09-18):

- Added `rocm/ds4_rocm_trace.cuh` and ROCm-only decode/read-worker hooks.
  The trace uses preallocated timing events, bounded host storage and the
  existing shutdown drain. It does not enable the old stage profiler or add
  a device/stream/event wait. Unset tracing allocates nothing and makes no
  GPU calls. Existing selected-pointer execution and upload scheduling remain.
- `DS4_ROCM_DECODE_TRACE=<new-file>` enables a private Chrome-trace JSON file.
  Existing files are refused. `DS4_ROCM_DECODE_TRACE_SKIP` defaults to 64
  decode steps; `DS4_ROCM_DECODE_TRACE_TOKENS` defaults to 8, with a maximum
  of 16. Storage is capped at 32768 records; exhaustion invalidates the trace
  rather than silently treating omitted work as free.
- `python3 tests/rocm_trace_summary.py <trace.json>` rejects incomplete,
  failed or truncated captures and summarizes intervals and upload bytes.
  GPU stream intervals include submission/queueing gaps. Uncovered intervals
  may contain uninstrumented work. CPU/GPU origins are only approximately
  aligned. **This is not yet an exact kernel-busy/device-idle timeline**, so
  the full completion criterion above remains open.
- RDNA4 build, Q8 GPU exact tests, existing streaming/server unit tests and
  new sanitizer-backed trace tests pass. New tests cover disabled mode,
  concurrent CPU spans, capacity exhaustion, event creation/record/query/
  timing errors, unfinished spans, nonfinite timings and existing-file safety.
- The 1529-token replay generated 384 identical tokens against archived
  `c229363`. Eight traced positions (1593-1600) produced 13560 records with
  zero drops, incomplete events or errors. Event-pool setup took 27.54 ms.
  Reference/traced steady decode was 9.338/9.375 t/s in this single pair;
  this does not establish window-level tracing overhead or a speedup.
- First output was 170.82/15.19 seconds. System-wide swap-in was approximately
  14.62 GiB/24.09 MiB, so these are not comparable first-output latencies.
  Raw swap-out counters and all timings are retained too.
- The 87441-token reference/traced pair completed 1024 generated tokens with
  identical generated fields, both at the token cap. Eight traced positions
  (87505-87512) produced 10144 records with zero drops, incomplete events or
  errors. Reference/traced steady decode was 7.984/8.062 t/s and first output
  was 589.96/590.60 seconds. System-wide swap-in was 27.54/3.91 MiB; swap-out
  was 0/16.38 MiB. These single-pair diagnostic results are not a gain claim.
- A tracing-disabled short pair also matched generated fields exactly, but
  measured 9.664/8.974 steady t/s (-7.1%). The candidate sample had 450.75 MiB
  system-wide swap-out versus none in the reference. Other CPU/VM workloads
  were active in a subsequent process snapshot. This does not establish either
  a code regression or its absence: retain the sample and repeat under quieter
  conditions before accepting disabled-mode throughput. Window-level tracing
  perturbation also remains unquantified.
- All model processes have finished. Production services remain stopped.
  Evidence and archived diagnostic binaries are under ignored, private
  `misc/rocm-timeline-20260918/`. No performance candidate has been selected.
  Full logprob, image-continuation and longer-generation acceptance is pending.

### 2. Long-context indexer and attention

Split the previously measured 38.91 ms/token into KV access, indexer projection,
score computation, top-k selection and attention. Compare short and long active
contexts while retaining 262144 allocated context, managed KV, vision residency
and steering scale 3. Distinguish memory movement from GPU compute before
choosing a kernel or scheduling change.

Done when: the expensive operation has measured timings and one specific,
correctness-preserving optimization hypothesis, or the measurements reject it.
Do not port unrelated model/backend changes or rebase to pursue this step.

First asynchronous breakdown (eight long-context decode steps, diagnostic
stream intervals in ms/token): indexer projection 1.29, score 19.08, top-k 1.87,
attention plus inverse RoPE 15.61. Short-context attention plus inverse RoPE
was 6.86; that short context does not enter indexer top-k selection. Do not sum
these with CPU wait/read timings or interpret them as exclusive GPU busy time.
KV movement versus compute has not yet been isolated.

Candidate 1, decode indexer scoring (screened, not accepted):

- A standalone ROCm benchmark reproduces the current scorer and compares every
  output bit before reporting timing. Simply widening its 128-thread block was
  rejected: at 21876 compressed rows, 256/512/1024-thread variants were 2.4%,
  13.3% and 27.7% slower by median runtime.
- A barrier-free mapping assigns one logical 32-lane wave to each compressed
  row. It retains each head's dot reduction and the exact ordered four-head
  accumulation. Volatile LDS contributions preserve the reference rounding
  boundary; without that boundary, about 80% of synthetic scores differed in
  their low bits and the variant was rejected before timing.
- The corrected wave-row kernel is bit-exact at 1024, 8192, 21876 and 65536
  rows. At 21876 rows, the best 128/256-thread block variants reduced median
  GPU-resident kernel time from 0.318 to 0.204/0.203 ms per indexed layer
  (about 36% less time). The old trace contains only 21 indexed layers, so the
  expected whole-token saving is about 2.4 ms, not 19 ms: traced GPU stream
  intervals included queue/submission gaps as warned.
- The private screening build used the opt-in selector
  `DS4_ROCM_INDEXER_WAVE_ROWS=1`; default behavior was unchanged. After
  rejection, the selector and candidate kernel were removed from the checkout
  and retained only in the private patch and archived binary. RDNA4,
  sanitizer/unit and Q8 GPU exact tests pass. The standalone benchmark target
  remains `make bench-indexer-score-rocm ROCM_ARCH=gfx1201`.
- Three alternating, unprofiled 87441-prompt/256-generation pairs matched all
  generated fields. Steady paired gains were +2.19%, +0.36% and -0.29%.
  Baseline/candidate mean was 8.056/8.116 t/s (+0.74%); median was
  8.077/8.112 t/s (+0.43%). System-wide swap movement was noisy: baseline
  swap-out was 446, 757 and 828 MiB; candidate swap-in/out also varied.
- This is not a repeatable 10-15% end-to-end gain, so candidate 1 does not
  advance to full logprob/image/long-generation acceptance and is not enabled
  by default. One candidate slot remains. Attention/KV or transfer work must
  show substantially more headroom before occupying it. Top-k is not the next
  target on current evidence.

A synthetic production-shape check also compared the exact-order indexed
attention decode kernel with the already-present multi-head online-softmax
kernel. The exact path took 0.178 ms/layer; 16/32-head online groups took
0.302/0.338 ms (+69%/+90%). Only 1932 of 32768 output floats were bit-identical
(max absolute delta `1.03e-8`). Small numerical drift is not the rejection
reason: both alternatives are much slower. They were rejected before model
screening, the temporary production selector was removed, and this does not
consume candidate 2. The private patch and benchmark are retained in evidence.

### 3. Targeted transfer benchmark

Measure the existing four-worker, double-buffered RAM-read/upload path using
production expert sizes and compare it with a copy-only measurement at the same
sizes. Record bytes, elapsed time, worker count, completion checks and the full
PCIe link path. Keep reads and overlapped uploads separate in the report.

Only select a transfer change if the trace or benchmark shows avoidable copies,
waits or unused transfer capacity. If bandwidth headroom is small, stop tuning
that path instead of repeating cache-size or eviction-policy sweeps.

The captured upload job sizes are 2162688 and 2752512 bytes. Requested bytes
per union of GPU upload intervals measured 11.86 GiB/s in the short window and
11.82 GiB/s in the long window. The eight-step windows uploaded 0.611/0.261 GiB
per token; do not extrapolate those route-dependent samples to whole-generation
averages.

A private same-size benchmark now covers four nonblocking streams, four workers
and two pinned staging banks per worker. Each measured run submitted 256 jobs
per worker (1024 jobs, 2.344 GiB total) and checked both final device buffers
for every worker. Five runs gave:

- copy-only median 11.83 GiB/s (median elapsed 198.05 ms);
- production-like `O_DIRECT` RAM read plus overlapped upload median 12.50 GiB/s
  (median elapsed 187.58 ms); worker read/wait sums overlap each other and wall
  time and are retained, not added;
- all completion checks passed and the SDMA assertion did not recur.

The read pipeline's higher result reflects staggered stream submissions, not
storage exceeding memory speed. It is only 5-6% above captured production
upload intervals. Even assigning all of that difference to avoidable transfer
cost saves about 1.2 ms/token in the long trace, under 1% of total decode time.
No transfer candidate is selected; stop tuning this path in this pass.

Sysfs reports the GPU and immediate bridge at 32.0 GT/s x16, then both upstream
links at 8.0 GT/s x16; the root port max is also 8.0 GT/s x16. Raw benchmark,
completion, source, hashes and topology are retained under
`misc/rocm-timeline-20260918/`.

### 4. SDMA runtime investigation

Check the installed ROCm runtime and upstream source for a confirmed fix to the
`BlitSdma<true>::PendingBytes()` ring-index assertion. Record the relevant source
and revision; do not infer a fix merely from a newer version number. If needed,
prepare a separately loaded runtime experiment without replacing system
libraries. Ask before changing the installed ROCm stack or production services.

SDMA-off remains a measured fallback with a 10.7% steady-throughput penalty on
the short replay, not a validated long-context solution. Any candidate runtime
must pass long-generation and image/continuation tests before being proposed
for production. If the crash recurs, retain the core and logs.

Read-only checkpoint (2026-09-18): installed `hsa-rocr` and `hip-runtime-amd`
are 7.2.4-1. The upstream source retrieved at
[`d1de02dae7804350300c45cdc5f894a6785d2ae6`](https://github.com/ROCm/rocm-systems/blob/d1de02dae7804350300c45cdc5f894a6785d2ae6/projects/rocr-runtime/runtime/hsa-runtime/core/inc/amd_blit_sdma.h)
still uses rounded-up eight-byte indexing into `bytes_written_`. For an
8 MiB ring, the hypothetical wrapped offset `0x7ffffc` still computes index
1048576 for a 1048576-entry vector. That reproduces the boundary arithmetic,
not the unavailable original offending offset. No confirmed upstream fix or
replacement runtime has been validated; installed libraries are unchanged.
Retrieved source, revision and file history are retained with this pass's
private evidence.

Diagnostic checkpoint (2026-09-19): the exact `rocm-7.2.4` tag at
`97f5574fe2fdc7bef44fb01545347912ee9f1779` is cloned under the ignored
`misc/rocm-local-runtime/` tree on branch
`ds4-sdma-pendingbytes-diagnostic`. An Arch-compatible assertion-enabled build
is installed only under `misc/rocm-local-runtime-prefix/`; its build ID is
`1.18.0-local-build-97f5574fe2-dirty`. The diagnostic reports the hardware read
pointer, wrapped offset, converted tracker index and size, commit/reserve
indices, queued bytes, first/final tracker values and wrap/clamp pending-byte
candidates, then preserves the original checked access and abort. It does not
change ring behavior. Loader inspection selects this library only when the
local prefix is explicitly prepended to `LD_LIBRARY_PATH`; the active production
server still maps `/opt/rocm/lib/libhsa-runtime64.so.1.18.0`.

Current upstream `develop` at comparison time still contains the same rounded
tracker indexing. A standalone deterministic boundary model covers offsets 0,
4, 8, `kQueueSize-8` and `kQueueSize-4`. It confirms that only the final
four-byte offset converts one past the vector and that clamping changes the
tracker's documented ceil-to-next-word meaning.

With explicit downtime approval, a loader smoke test confirmed the private
process mapped the local diagnostic library. A four-worker production-size run
then completed 4,000,000 copies (9,830,400,000,000 bytes, about 9.15 TiB) in
719.608 seconds at 12.723 GiB/s with exact final-buffer checks. It was followed
by a command-density run of 12,000,000 asynchronous 4-KiB copies in 96.735
seconds, also with exact completion. Neither run produced the diagnostic line
or assertion. The production-size run increased the system `pswpin` counter by
1662 pages and did not change `pswpout`. A second command-density run made
1,107,916,361 direct calls to `hsa_amd_memory_copy_engine_status()` while
completing another 12,000,000 copies in 97.509 seconds; it also did not reproduce
the assertion. These negative bounded runs show that this copy pattern does not
reach the failing ring state, not that the original failure is absent.

Arch debuginfod symbols and the retained cores then recovered the missing state
without another crash. In PID 508873, the failing boundary had commit
`0x2000007c`, uniquely implied read index `0x1ffffffc`, and 128 command bytes
pending; `bytes_queued - tracker[0]` was exactly 2162688 bytes. PID 3012217
independently had commit `0x2d80007c`, implied read index `0x2d7ffffc`, the same
128 command bytes pending, and exactly 2752512 bytes from tracker slot zero.
Those are the two production upload sizes. The final tracker slot was stale zero
in both cores, so clamping would have reported all 11.1/16.2 TB ever queued as
pending. This confirms that the ceil-converted final half-word denotes slot zero
in the next ring cycle; neither clamp nor an unmirrored sentinel is valid.

A clean local fix on branch `ds4-sdma-pendingbytes-wrap-fix`, commit
`1a2897e7ad2d26c631db52054c674f921252a9ce`, maps only
`convert(kQueueSize-4) == data_.size()` to slot zero and preserves the checked
access for every other invalid index. Six boundary-model tests pass. The
assertion-enabled fixed build is isolated under
`misc/rocm-local-runtime-fixed-prefix/`, has build ID
`1.18.0-local-build-1a2897e7ad`, and retains the system runtime's 262-symbol ABI,
SONAME and dependencies. A fixed-runtime stress run completed 12,000,000
asynchronous 4-KiB copies and 1,121,405,126 direct copy-engine status polls in
96.383 seconds with exact final-buffer checks and no errors. `pswpin` increased
by 641 pages and `pswpout` did not change.

Model-level comparison used the same current `ds4-server` binary with only the
ROCr loader path changed. Three alternating 1529-prompt/384-completion-token
pairs produced byte-identical complete generated fields in all six runs. Median
post-first-interval decode was 9.624 t/s on system ROCr and 9.613 t/s on fixed
ROCr; this is no material performance change. A separate 87441-prompt-token
pair generated 1024 continuation tokens per run with byte-identical complete
fields. Post-first-interval decode was 7.916 t/s on system ROCr and 7.966 t/s
on fixed ROCr; the single pair is a correctness/stability check, not a gain
claim. Per-run first-output, generation, swap and major-fault counters are
retained. The initial short-run attempt exited before GPU initialization because
the service's normal `ExecStopPost` had removed the RAM model; after recreating
it with `ram-disk-up.sh`, all intended runs passed.

The remaining acceptance checks also passed. Full top-20 logprob JSON matched
exactly for 128 greedy steps at 42, 25 and 12583 prompt tokens. Two-turn text,
single-image continuation and a steered image/text/image sequence completed
2, 1 and 3 generations per runtime respectively with byte-identical nonempty
stdout. A requested 2048-token short-prompt run stopped naturally at 413 tokens
and was retained as exactness evidence rather than misreported as long-run
coverage. The 87441-token replay was therefore rerun with a 1536-token cap; both
system and fixed runtimes reached the length cap with byte-identical complete
fields. Post-first-interval decode was 7.906 and 7.897 t/s respectively. Swap
and major-fault deltas are retained for every run.

The fixed runtime now passes the deterministic boundary model, bounded
concurrent-SDMA stress, repeated text exactness, full top-20 logprobs, image and
multi-turn continuation, and 87K-context generation beyond 1024 tokens without
a material performance regression. This is a pre-production acceptance pass.

After explicit deployment approval, `ds4-server-launch.sh` was changed to fail
closed if the validated project-local runtime is missing, prepend only
`misc/rocm-local-runtime-fixed-prefix/lib` to its loader path, and keep SDMA
explicitly enabled by default. No system ROCm file or other application was
changed. Service invocation `76e108da3dd54c2990a88c8f36197567`, PID 2332354,
maps the project-local `libhsa-runtime64.so.1.18.0`, reached its listening state,
and completed a 128-token production smoke request without an immediate fault.
The smoke added seven system swap-ins and no swap-outs. Production soak
monitoring is now in progress; remove the launcher override when a confirmed
upstream fix replaces local commit `1a2897e7ad`. Core derivation, logs, hashes,
stress source and validation summary are retained under
`misc/rocm-prefetch-20260919/`.

### Acceptance and stopping rules

- Test one candidate at a time against the archived `c229363` baseline.
- Screen at most two candidates. Record failures and rejected hypotheses.
- Require three alternating unprofiled pairs on representative workloads for a
  gain claim. Keep first-output/transition timings separate from steady decode.
- Retain noisy samples and system-wide swap counters. Do not compile or run a
  second model during acceptance measurements; respect the instance lock.
- Run the RDNA4 build, CPU streaming tests and Q8 GPU exact tests. Compare saved
  top-20 logprob JSON on short, medium and 12583-token prompts with greedy steps.
- Check two-turn and image stdout equality plus a steered image/text/image
  sequence. Recheck the 87441-token workload and test long generations beyond
  the existing 1024-token coverage. Record actual completion and token caps.
- Test each new optimization's disable switch. Preserve managed KV, vision,
  steering 3, reasoning-stream buffering and CUDA/Metal/CPU/distributed/SSD
  defaults. Ask before CUDA or distributed testing.
- A faster configuration that still aborts does not pass production acceptance.
  If this bounded pass finds no validated gain, report that result without
  claiming an absolute hardware/software ceiling.
- Preserve evidence in an ignored, private `misc/` directory. Do not push,
  delete models, change Pi configuration or silently restart production.

## Post-pass software feasibility checks

Read-only/offline follow-up on 2026-09-19 screened route prediction, additional
residency and transfer reduction. It did not change inference code or production
settings and did not start another acceptance candidate.

A 256-token route capture at about 87K context reproduced 15986 measured cache
misses when the simulator used the realized 2211 slots. Previous-token,
same-layer prediction repeated 31.7% of selections, but every repeated expert
was already resident and it covered no demand misses. Popularity, recency,
recurrence-period, route-transition and hybrid causal predictors also failed:
the best long-trace result recalled about 3.7% of misses at 5.9% precision,
increased traffic about 59%, and could ideally hide only 1.2 ms/token. The
short trace was also insufficient at about 5.1% recall, 9.8% precision, 47%
extra traffic and 2.1 ms/token ideal overlap.

Capacity curves reject arena reclamation as a standalone target. Another 83
slots (0.55 GiB) save about 3.2% of misses and ideally 1.1 ms/token. Reclaiming
the complete 512-slot/3.38-GiB prefill reserve saves about 15.7% of misses and
ideally 5.2 ms/token. Neither reaches the 11-16 ms/token gate.

The existing selected-load path already overlaps selected-ID read/load with the
shared gate/up and down branch, then executes resident routed gate/up while
missing uploads continue. In the long profile, missing routed gate/up totals
1.77 ms/token and all routed down compute totals 3.46 ms/token. Even granting
the impossible best case that all down work plus missing gate/up can overlap
transfers, staged gate/down readiness is bounded at 5.23 ms/token. Adding the
optimistic full 3.38-GiB cache-reclamation bound yields 10.4 ms/token, still
below the 11-16 ms gate before overlap conflicts and staging overhead. Partial
upload readiness and per-expert down scheduling were rejected without a
production implementation.

Replacement-policy simulation found theoretical but noncausal headroom. At
2211 slots, future-aware Belady replacement reduced misses from 15986 to 9596
(40.0%, ideally 13.2 ms/token). Causal frequency, LRU-K and segmented-LRU
variants did not approach that bound; the best reduced misses 1.7%. Admission
control using TinyLFU, reuse thresholds or interval estimates made misses worse.
A per-layer partition trained on the first 128 tokens also worsened held-out
misses from 7266 to 7502. Even a full-trace overfit partition saved only 25
held-out misses. The oracle gap depends on future routes that these causal
signals do not identify.

A deterministic 32-expert, 226492416-byte sample from the mixed IQ2_XXS
gate/up and Q2_K down streams compressed to 224559259 bytes with zstd level 1
(99.15% of input), 224683004 at level 3, and grew slightly with lz4. Lossless
compression cannot materially reduce PCIe traffic.

CPU miss offload also failed its optimistic timing gate. A standalone benchmark
used the existing scalar x86 IQ2_XXS/Q2_K dot routines and real sampled weights,
cycled through 216 MiB, and omitted SwiGLU, intermediate quantization,
orchestration, synchronization and result upload. With 16 physical threads it
took 1.65 ms for one missing expert and 3.38 ms for two; 32 SMT threads were
slower. One 6.75-MiB expert transfers in about 0.53 ms at 12.5 GiB/s, while the
measured missing-upload finish wait averages about 1.02 ms per MoE call. The CPU
lower bound is already slower before omitted production costs and would contend
for host memory bandwidth. A checksum-equivalent AVX2 rewrite improved the lower
bound only to 1.34/2.54 ms, still about 2.5 times the transfer time per expert on
a CPU without AVX-512 VNNI. Heterogeneous miss execution was rejected before
inference integration.

The main GGUF has no built-in speculative block. Vision-Exp DSpark requires a
separate checkpoint-specific support GGUF that is not installed, would violate
the no-new-model constraint, and is documented as unfavorable when target
verification re-streams routed experts. It was not downloaded or run.

Evidence and replay tools are retained under ignored
`misc/rocm-prefetch-20260919/`. These checks reject predictive prefetch,
cache-only growth, intra-layer down-transfer reordering, causal cache-policy
changes, lossless transfer compression and CPU miss offload for the current
10-15% target. The Belady result is an upper bound, not a demonstrated
optimization or an absolute hardware/software ceiling.
