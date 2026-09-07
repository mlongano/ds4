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
