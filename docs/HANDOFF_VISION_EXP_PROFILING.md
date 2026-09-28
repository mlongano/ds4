# Handoff: profile and speed up the existing DeepSeek V4 Flash Vision-Exp path (ROCm / RDNA4)

> **Rescued from `/tmp` on 2026-09-29.** It lived at `/tmp/handoff-ds4-vision-exp-profiling.md`,
> which a reboot clears. Read it as the task statement it was: the bounded profiling pass it asks
> for was attempted on September 17 and 18, and the answer is recorded in
> `ROCM_STREAMING_PLAN.md` under "Bounded long-context investigation, September 17 and 18, 2026" —
> no repeatable 10-15% generation gain was found and no optimization was accepted from it.
> What remains valuable here is the resumption protocol in the middle of the file: how to find out
> how a killed turn actually ended, and what to re-verify before trusting a handoff.

Date: 2026-09-17
Workspace: `/media/NVME_DATA/MOUNTS/Models/ds4`
Branch: `rocm-rdna4-streaming`, HEAD `c229363 perf(rocm): retire prefill model arenas before decode`
Tracked working tree: clean. Only pre-existing untracked files (`.deps/`, `bench-*.sh`, `build-rdna4.log`, `download-q2-imatrix.log`, `ds4-server.log`, `kv-cache/`, `smoke-rdna4-22gb.log`).

Record of what has already been done, with measurements and limits: `ROCM_STREAMING_PLAN.md`.
Do not restate that file here; read it first.

## Task for the next session

The user chose option "improve the existing Vision-Exp implementation" over porting a new model.
Scope is a **bounded profiling investigation at realistic operating conditions**, not a new model and not a rebase.

Question to answer: **is there another 10-15% repeatable generation gain left in the current implementation, and where?**

Acceptance target (not a promise): a repeatable 10-15% generation improvement on representative workloads, with vision, steering scale 3, 262144 allocated context, managed KV and output correctness preserved. A faster build that still aborts during long generations does not pass.

## Resuming after a provider usage limit

A turn can die with no output at all when the provider refuses the request. That is how the previous session ended: at entry 1811, `2026-09-16T20:45:02Z`, an assistant turn with empty `content`, `stopReason: "error"`, and `errorMessage: "Codex error: The usage limit has been reached"`. The user prompt that opened it, `go on`, is entry 1800 at `20:38:38Z`; the last real work was the tool result at entry 1810.

This is not a one-off. The same session history contains nine error turns, from websocket timeouts to two variants of the ChatGPT plan limit, one OpenRouter key limit, and one aborted request. So identify the **last** one rather than the first, and do not assume the interruption you are resuming from is the one you remember.

No build ran, no benchmark started, no file was written, so there is no partial state to repair. The risk is the opposite one: an agent resuming without checking may assume work happened, or re-run reconnaissance it already paid for.

Confirm how the previous turn ended before doing anything else:

```bash
python3 - <<'PY'
import json
p="/home/mauro/.pi/agent/sessions/--media-NVME_DATA-MOUNTS-Models-ds4--/2026-09-04T18-26-19-267Z_01a06dab-ee03-7c35-bbb0-2a194d4390d0.jsonl"
for i, l in enumerate(open(p, errors="replace"), 1):
    l = l.strip()
    if not l: continue
    r = json.loads(l); m = r.get("message") or {}
    if m.get("role") == "assistant" and m.get("stopReason") == "error":
        print(i, r.get("timestamp"), m.get("model"), repr(m.get("errorMessage")))
PY
```

To see what the killed turn had managed to do, list the entries between the matching user prompt and that error entry. Tool calls and results are appended as they complete, so the last tool result before the error marks the real stopping point:

```bash
python3 - <<'PY'
import json
p="/home/mauro/.pi/agent/sessions/--media-NVME_DATA-MOUNTS-Models-ds4--/2026-09-04T18-26-19-267Z_01a06dab-ee03-7c35-bbb0-2a194d4390d0.jsonl"
rows=[(i,json.loads(l)) for i,l in enumerate(open(p,errors="replace"),1) if l.strip()]
stop=[i for i,r in rows if (r.get("message") or {}).get("stopReason")=="error"]
last=stop[-1] if stop else len(rows)
start=max(i for i,r in rows if i<last and (r.get("message") or {}).get("role")=="user")
for i,r in rows:
    if start<=i<=last:
        m=r.get("message") or {}
        print(i, r.get("type"), m.get("role"), json.dumps(m.get("content"),ensure_ascii=False)[:160])
PY
```

Then resume in this order:

1. Trust the session record for reconnaissance already completed: files read, service and GPU state, profiler flag inventory, the binary and model listing. Reading them again wastes the run's budget, which is the constraint that killed the last one.
2. Re-check only the facts that drift. Service states, `/mnt/ram/ds4flash.gguf`, `/tmp` baseline artifacts, and `origin/main` all change between sessions. The exact commands used before are in the entries listed above; reuse them rather than inventing new ones.
3. Re-read `ROCM_STREAMING_PLAN.md` and this file instead of rebuilding the plan from memory. A killed turn commits nothing, and the conversation context that held the plan may not survive the resume.
4. Continue from the first incomplete step under "Measurements to take". Last time that was step 1, with nothing started.

Budget the run deliberately, since the limit that ended the previous session applies to the whole session, not to one turn. Do the expensive work, the build and the paired benchmark runs, in one uninterrupted stretch rather than leaving it until after more reading.

Done when: the last error entry is identified, the drifting facts are re-verified, the first incomplete measurement step is named, and expensive work is scheduled for early in the run.

## Guardrails

- Do not delete the on-disk 0731 model; deletion was never confirmed.
- Do not push. Keep commits local.
- Do not change production service files or Pi configuration as part of profiling.
- Preserve the behaviour of Metal, CUDA, CPU, distributed inference and SSD streaming.
- Hard rule from `AGENT.md`: whole-model SSD streaming, CUDA and Metal defaults must be unaffected by ROCm changes. Ask before touching distributed/CUDA.
- Record noisy samples and system-wide swap counters instead of excluding them.
- Treat reasoning-stream buffering as deliberate. A previous upstream change (`fe2d3b0`) added it to stop malformed second-reasoning leakage, and existing regression tests depend on it. Leave it alone.
- The decode-stage profiler fails when selected-pointer execution is active (`missing compact selected experts ... full expert table is not mapped`). Never use it for acceptance numbers.

## Current machine and service state

- GPU idle, about 62 MB VRAM in use. No ds4 process running.
- `ds4-server.service`: inactive.
- `pi-llama-router.service`: inactive (the Pi llama.cpp presets are not loaded).
- `pi-inference-manager.service`: active.
- `/mnt/ram/ds4flash.gguf`: **not present**. The tmpfs was unmounted after the crash. The NVMe original is intact and byte-for-byte verified.
- The last `git fetch` in this branch aborted. Re-fetch before making any claim about upstream.

Because both services are down, the earlier need to ask permission before stopping production no longer applies. Confirm with the user before starting or stopping anything anyway.

## Missing artifacts that must be rebuilt

`/tmp` does not currently hold the previous session's evidence or baselines:

- `/tmp/ds4-generation-perf/` is gone, including the saved `35bacb1` baseline binaries and the frozen replay request (`screen-v2/request.json`).
- Treat any other `/tmp/ds4-*` directory as absent until verified.

Consequences:

1. There is no saved reference binary to diff against. To isolate a new change, first build and archive a baseline from the current HEAD (`c229363`) the same way `35bacb1` was archived before.
2. There is no frozen request JSON for `tests/rocm_server_regression.py`. It must be recreated from the saved Pi session file below. That runner compares generated content between two binaries, so a fixture is mandatory before any A/B pair.

`/tmp/ds4.lock` exists. The instance lock is intentional; do not work around it.

## Bringing the model back

The RAM path needs `./ram-disk-up.sh` (uses `sudo mount` for a 90G tmpfs and copies about 81 GiB). Expect several minutes. It prints the RAM model path.

Alternative: run against the NVMe GGUF directly. This changes the I/O profile, so it is fine for correctness checks and for locating where time goes, but not for a speed claim about the RAM-backed production configuration.

## Measurements to take, in this order

Work from the real workload outward. Each step ends when its criterion is met.

### 1. Reproduce the long-context conditions

The user's real session reached roughly 87K active context; the strongest previous validation covered only 16K. Reproduce a long active context with steering 3, thinking and tools enabled.

Done when: a long-context server run completes with identical output between the two binaries being compared, and both first-output and steady-decode timings are recorded separately.

### 2. Split the decode cost

Use the existing committed diagnostics, unprofiled runs for the final numbers:

- `DS4_SERVER_DECODE_PROFILE=1` splits sampling, inference, first eval, other.
- `DS4_ROCM_STREAM_CACHE_STATS=1` and `_LAYER_STATS=1` give expert residency, hits and misses.
- `DS4_ROCM_STREAM_READ_PROFILE=1` gives read jobs, bytes and wait time.

Known baseline for calibration, from the frozen 1529-token replay: 99072 selections with 48370 misses and only 6.10 GiB resident before the model-span reset; after the accepted change, 13.83 GiB resident and 30694 misses. Sampling was 149 ms of 52.6 s, so sampling is not the bottleneck.

Done when: each of attention/KV, expert upload traffic, and resident GPU compute has a number, and the largest one is named with evidence.

### 3. Act on the largest cost only

Candidates, in rough order of expected value. Test one at a time.

- Expert upload traffic: bytes per token, miss rate, slab placement. The earlier cache tuning was sensitive to prefill-versus-decode VRAM competition, so check placement before changing budgets.
- Resident compute: Q8 projections, shared experts, routed-expert dequantization. Relevant because upstream's quantization recipe puts decode traffic in the 8-bit projections and shared experts rather than the routed experts.
- Upstream allocation and memory-budget changes (`9d9e129`, `0e9cc2d` among others). Inspect and test the specific change independently; a full rebase is not a prerequisite.
- `HSA_ENABLE_SDMA=0` as an SDMA workaround. Its throughput cost is unmeasured and it is not free to assume. Measure it before proposing it.

Done when: one change produces a repeatable gain on paired alternating runs, or the investigation is reported as exhausted with the numbers that show it.

## Validation required for any accepted change

- `CPATH=/opt/rocm/include LIBRARY_PATH=/opt/rocm/lib LD_LIBRARY_PATH=/opt/rocm/lib make rdna4 -j16`
- `make test-rocm-streaming-unit`
- `LD_LIBRARY_PATH=/opt/rocm/lib LIBRARY_PATH=/opt/rocm/lib make test-rocm-q8-prefill`
- Saved top-20 logprob JSON equality for the short, medium and 12583-token prompts, each followed by greedy steps.
- Two-turn and image continuation stdout equality, plus a steered image/text/image sequence, with steering scale 3.
- The disable switch for whatever is added.
- Report steady generation separately from first-token/transition time.

Read `ROCM_STREAMING_PLAN.md` for the exact runner invocations and the reasoning behind them rather than re-deriving them.

## Open problem that is not resolved

The service aborted twice on the same `libhsa-runtime64` assertion (`stl_vector.h:1253`, `__n < this->size()`), both mid-decode in the streaming upload path. The symbolized cause is `BlitSdma<true>::PendingBytes()`, whose ring index can reach 1048576 for an 8 MiB ring while the tracking vector holds exactly 1048576 entries. That is a bounds problem in ROCm 7.2.4, not in this codebase. Exact offending offset was optimized out and is not proven.

Any performance work has to coexist with this. Do not present a faster configuration as finished while long generations can still abort.

## Suggested skills

- `unslop` (`/home/mauro/.agents/skills/unslop/SKILL.md`): apply to all written output.
- `diagnose-crash` (`/home/mauro/.pi/agent/skills/diagnose-crash/SKILL.md`): if the SDMA abort recurs, gather the core and symbolize from `coredumpctl`; the earlier analysis of frames and locals is in the conversation record.
- `conventional-commit` (`/home/mauro/.pi/agent/skills/conventional-commit/SKILL.md`): for any local commit.
- `debug-helper` (`/home/mauro/.pi/agent/npm/node_modules/@ifi/oh-pi-skills/skills/debug-helper/SKILL.md`): general measure/profile/verify loop.

## Pointers

- Plan and prior measurements: `ROCM_STREAMING_PLAN.md`
- Rules for the codebase: `AGENT.md`
- Server A/B runner: `tests/rocm_server_regression.py`
- CLI A/B, TTFT and vision runner: `tests/rocm_streaming_regression.py`
- CPU sanitizer/policy tests: `tests/test_rocm_stream_pipeline.py`, `tests/test_rocm_stream_policy.c`, `tests/test_rocm_model_upload.c`, `tests/test_rocm_decode_model_reset.c`
- GPU prefill correctness test: `tests/test_rocm_q8_prefill.c`
- Streaming runtime and arenas: `rocm/ds4_rocm_runtime.cuh`
- Decode MoE, overlap and stage profiler: `rocm/ds4_rocm_moe_launch.cuh`
- Phase-boundary hook: `ds4.c` (search `ds4_rocm_release_prefill_model_ranges`)
- Saved session used for the replay fixture: `/home/mauro/.pi/agent/sessions/--media-NVME_DATA-MOUNTS-Models-ds4--/2026-09-04T18-26-19-267Z_01a06dab-ee03-7c35-bbb0-2a194d4390d0.jsonl`
