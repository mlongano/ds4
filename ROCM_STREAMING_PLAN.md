# R9700 streaming improvement plan

Baseline: `78dce0e`, RAM-backed Vision-Exp IQ2_XXS gate / Q2_K down,
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

### Remaining work

The larger prefill/first-token opportunity is still open. Separate persistent
dense allocations from temporary prefill ranges before changing arena eviction
or scratch release. Add batch-specific pins before enabling prefill overlap.
Test cancellation/recovery and long-context continuation beyond 16K before
expanding the performance claims. Expert residency changes need new traces
with the revised transfer schedule; no new cache-size defaults were selected.
