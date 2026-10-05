# State of the codebase, 2026-10-05

Written by the MiMo-V2.6-Pro session of 2026-10-05, after reading the tree, the
benchmark corpus and the fetched upstream. Companion file with per-commit
upstream verdicts: `docs/UPSTREAM-INTAKE-2026-10-05.md`. Upstream procedure:
`docs/UPSTREAM.md`. Measurement history: `docs/PERFORMANCE_REFERENCE.md` and
`docs/ROCM_STREAMING_PLAN.md`.

## What this is

A fork of `antirez/ds4`, a deliberately narrow inference engine for DeepSeek V4
Flash (including the experimental vision model) and GLM 5.2/5.3. The fork adds
one thing upstream does not have: it runs on a discrete AMD card. The ROCm
backend upstream targets Strix Halo integrated graphics; this branch makes it
run on a Radeon AI PRO R9700, 32 GiB, `gfx1201`, and adds a RAM-disk streaming
mode so the 86 GB weights live in a 90 GiB tmpfs and stream into VRAM on demand.

That port is deployed as a service on this machine and used daily. It is not an
experiment branch. That fact should govern every merge decision made from here.

## Where the branch stands

Branch `rocm-rdna4-streaming`, head `42e19db` (2026-10-05), pushed to
`mine/rocm-rdna4-streaming`. `origin` is `antirez/ds4` and is fetch-only by
design; pushes go to `mlongano/ds4` (`mine`). Merge base with `origin/main` is
`b0a147a` ("Add conversation prefix conditioning", 2026-09-02). As of the fetch
today: 29 commits ahead, 156 behind.

The 29 local commits, by area:

| Area | Commits | Where |
|---|---|---|
| ROCm/RDNA4 port (WMMA fallback, VRAM reserve knob) | `85ec136` | `rocm/`, `ds4_rocm.*`, `Makefile` |
| RAM-disk SSD streaming | `318b3d5`, `d98551b`, `45eafee` | `ram-disk-*.sh`, `ds4_ssd.*` |
| ROCm performance chain | `3d53ea9`, `cc0ef07`, `78dce0e`, `df7cc08`, `35bacb1`, `c229363` | `ds4.c`, `rocm/` |
| Decode tracing and microbenchmarks | `e91fdaf`, `9101e0c`, `6557dd8`, `613c00b` | `rocm/`, `tests/`, `Makefile` |
| Metal TP portability | `575a4c7` | `ds4.c`, `metal/` |
| Local ROCr runtime launch fix | `a825b94` | `ds4-server-launch.sh`, `misc/` |
| Bench sweeps (added today) | `42e19db` | `bench-*.sh` |
| Documentation and rescued evidence | 12 `docs:` commits | `docs/`, `evidence/` |

## What is running right now

`pi-inference status` reports mode `ds4`, service `ds4` active. The user unit
`~/.config/systemd/user/ds4-server.service` runs
`ds4-server-launch.sh` out of this directory with:

```
--ctx 262144 --port 8000 --ssd-streaming --ssd-streaming-cold
--ssd-streaming-cache-experts 18GB
--vision gguf/DeepSeek-V4-Flash-Vision-Encoder.gguf
--dir-steering-file dir-steering/out/refusal_train400_en_it_plus20_nothink_ffn_out.f32
--dir-steering-ffn 3
```

plus `DS4_ROCM_STREAM_FREE_RESERVE_GB=2` and `DS4_SSD_AUTO_CACHE_PCT=65`, and
`ram-disk-down.sh` on stop. The binary is the one built in this tree on Sep 19
03:41 (`ds4-server`, md5 `ba28201a6c3ff1c6a677739ad3c2259b`). No source file is
newer than the binaries; every commit since the build is documentation. So the
deployed code matches the head of the branch.

## Model weights

`gguf/` holds about 162 GiB, three files:

| File | Size | Role |
|---|---:|---|
| `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf` | 86.7 GB | text model |
| `DeepSeek-V4-Flash-Vision-Exp-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8.gguf` | 86.7 GB | vision model, the `ds4flash.gguf` symlink target, the one the service loads |
| `DeepSeek-V4-Flash-Vision-Encoder.gguf` | 0.93 GB | vision encoder, passed as `--vision` |

Only the Q2 XXS tier is on this machine. `download_model.sh` knows about Q4,
MXFP4, Vision Q2+Q4, DSpark, GLM 5.2/5.3, GLM 5.3 full and DeepSeek PRO
variants; none of those are downloaded here. If upstream lands DeepSeek v4.1
support and the weights follow, the intake question becomes whether this
machine should also host v4.1.

## The refusal steering vector

This is not a code patch, and the distinction matters. `dir-steering/` implements
a runtime activation edit: `y = y - scale * direction[layer] *
dot(direction[layer], y)`, applied after FFN outputs. Positive scale removes the
direction from the activations, negative amplifies it.

The deployed vector is
`dir-steering/out/refusal_train400_en_it_plus20_nothink_ffn_out.f32`, a 43 x 4096
f32 matrix (704,512 bytes, the DeepSeek V4 Flash shape). It was downloaded from
`https://antirez.com/misc/refusal_train400_en_it_plus20_nothink_ffn_out.f32`,
sha256 `cf84bb41c44105b53cb12566234e3e30470c9c62cf1450bcad3156dd3a28ee42`,
fetched again on 2026-09-29 and byte-identical.

The provenance JSON beside it (`dir-steering/out/refusal_...json`) is honest
about the gaps, and I recommend keeping that honesty. The training set is not
published; the filename is decoded by inference; the file is a global FFN
activation edit and not a refusal switch. It is recorded as a possible
contributor to planning repetition, unproven either way. Scale 3 was chosen by a
boundary sweep on two prompts that normally refuse, 24 tokens greedy: scale 0
refuses, 2 refuses, 3 complies, 4 complies with unnecessary distortion, and
control prompts stay coherent at 3.

GLM 5.3 would use a 45 x 4096 direction; GLM 5.2 steering is not implemented. If
the model is ever replaced, this vector is dead weight and must be rebuilt, not
copied.

## VRAM and RAM, ours against antirez's

The two designs answer different machines, and both are right for their target.

Antirez upstream (`9d9e129` and friends) assumes unified memory. On Strix Halo
the GPU eats host RAM, so the enemy is starving the OS. His fix is admission
control: `ds4_rocm_memory.h` counts only usable host memory (CMA pages excluded),
keeps a 2 GiB OS reserve, refuses allocations that would breach it, clamps
`hipMemGetInfo` to what the host can actually give, tightens arena chunking so
large tensor spans stop stranding hundreds of MiB, and treats the Q8 to F16
weight expansion as optional acceleration that must leave 8 GiB for workspaces.

This fork assumes separation. On the R9700, XNACK is off and host and device
memory are genuinely distinct pools joined by a PCIe 3.0 x16 link, so the enemy
is the transfer, not the OS. The design here splits the world into a 90 GiB tmpfs
holding the weights and 32 GiB of VRAM acting as a streaming cache for experts
and KV. The work then goes into making the two cooperate:

- `85ec136` makes the free-VRAM reserve configurable (`DS4_ROCM_FREE_RESERVE_MB`,
  later `DS4_ROCM_STREAM_FREE_RESERVE_GB`), because upstream's 16 GiB default
  suits APUs and leaves half of a 32 GiB card idle.
- `3d53ea9` shares VRAM between KV and expert caches: resident expert slabs are
  reclaimed before transient arena growth, and large streaming contexts get
  managed KV pages.
- `78dce0e` and `df7cc08` pin expert pointers and pipeline host to device
  transfers so loading overlaps decode.
- `35bacb1` runs prefill on the host side (8K prefill 109.82 to 174.07 t/s) and
  `c229363` retires those prefill arenas before decode starts.

Where the two meet: upstream's admission layer is a no-op on discrete GPUs
(`ds4_rocm_allocation_fits` returns true unless the device is integrated or the
allocation is host memory), so the philosophies do not fight. The code still
collides, because `9d9e129` reroutes every `cudaMalloc`, `cudaMallocHost` and
`cudaMemGetInfo` through wrappers, replaces `cuda_model_arena_chunk_bytes`
(the DeepSeek Q4 span logic our `c229363` builds on) with
`ds4_rocm_model_arena_bytes`, and rewrites the Q8 reserve function, all inside
`rocm/ds4_rocm_runtime.cuh`, which is also the file our perf chain rewrote
(about 640 changed lines). That single file is the merge that needs a human.

## What the benchmarks proved

The corpus is unusually good, mostly because it keeps its negative results.

Measured and accepted (full tables in `docs/PERFORMANCE_REFERENCE.md`):
steady decode 8.60 t/s at 8K populated context down to 7.06 t/s at 131K, prefill
114.29 t/s at 8K down to 99.59 at 131K, RAM-backed Vision-Exp, 262K allocated.
The per-change wins are recorded with their A/B method: ram-backed prefill
(109.82 to 174.07 t/s at 8K, TTFT 6.21 to 5.00 s), pinned expert pointers (8K
median 8.36 to 10.28, +23.0%), paired-row prefill with the tmpfs bulk uploader
(113.83 t/s at 8K with saved logprobs equal at 12,583 tokens).

Rejected, with the number that killed each: model-span budgets under 10 GiB
collapse decode to 0.47 to 0.48 t/s; the one-token resident/missing split path
hit 11.8 t/s but produced nondeterministic logits; shrinking the cache for 64K
tuning bought 4.73 to 4.86 t/s and lost 4.9 GiB of expert residency; a
decode-only tuning regressed against the deployed binary. Known trap: SDMA
writes into trimmed expert slabs raise an HSA assertion, `HSA_ENABLE_SDMA=0`
gives 8.70 t/s against 9.74 with SDMA, and any runtime fix should target
stability at 9.74 rather than the hoped 10.8.

Tooling worth knowing about: `speed-bench/metal_decode_schedule_bench.c` and
`metal_prefill_variant_bench.c` demand bit-identical full-vocabulary logits
between variants and abort otherwise, with ABBA/BAAB alternation and rollback
controls via `--candidate-env`. That is the bar any future ROCm A/B should match.
`speed-bench/gfx1151-prefill-results.md` records the Strix Halo tuning (warm 4K
prefill 187.26 to 292.86 t/s) with byte-identical frontier validation.

Evidence fragility is documented and partly real: `misc/` (1.3 GB of run
directories, ROCr build trees) is gitignored, and `/tmp` already swallowed the
frozen reference binaries and the `replay.py` harness. The small artifacts are
rescued under `evidence/` (copied 2026-09-29), including the rejected patches by
name. One thing I would call a genuine risk: the patched ROCr runtime the
service loads comes from `misc/rocm-local-runtime-fixed-prefix/lib`, built from a
six-line local commit (`1a2897e7ad`, `amd_blit_sdma.h`) that exists on no
remote. The tree says this clearly, but a rebuild on a fresh machine is not
possible today.

## Standing against origin/main

156 commits behind. The intake verdict for every one of them is in
`docs/UPSTREAM-INTAKE-2026-10-05.md`; the summary of what actually matters to
this port:

Must take, and read before merging:

- `9d9e129` ROCm allocation budgeting against pinned memory. Overlaps our VRAM
  work in `rocm/ds4_rocm_runtime.cuh` and `ds4.c`. Its tests are Strix-focused;
  ours would need a discrete-GPU counterpart.
- `0e9cc2d` GLM attention masking and reductions across backends. A correctness
  fix in a model this deployment serves, in `rocm/ds4_rocm_glm.cuh`, and it
  unlocked 21 to 68 t/s continued prefill on Strix. Correctness before speed,
  take it.
- `233eeb8` session rollback, TP restore and GLM memory accounting, plus
  `ade4d62` (Linux builds after Metal TP changes) and `932896f`. These land in
  the same area as our `575a4c7` portability commit and reduce its patch
  surface.

Take, cheap and in our paths: `85837d0` (the only upstream commit touching
`ds4_ssd.*`, the file our RAM-disk mode extends), `b6af0ad` (GLM SSD cache
priorities, 0.48 to 4.38 t/s on a real bug), `96f547a` and `4a14eb8` (dead ROCm
kernel removal; I checked that nothing in our tree calls the deleted symbols and
that the helpers our added Q8 kernel uses survive), `6687c6f`, `49ec955` (ROCm
build hygiene), and the session/server correctness set (`c0a6119`, `4d7028e`,
`ada805a`, `5994f06`, `81a7658`).

Skip: the gfx1151 tuning set (`1ef9bba`, `e83b106`, `64e490b`, `394865c`, merge
`909d656`). Decided on 2026-10-05: this fork is R9700-only from here on, so
Strix tuning is not intent we defend in conflicts.

DeepSeek v4.1 Flash support (`bd66c40`, `a04f46f`, `d9bc45d`, `6c00e2d`) is
taken as-is, with one thing understood. Upstream wrote v4.1 for Metal and CUDA
only, and nothing in the queue ports it to the ROCm backend. After the merge the
code is inert on the R9700 until someone does that port. Until then a v4.1 model
cannot be served here. If that day comes it is its own project, its own branch,
its own weights download, and it must rebuild the refusal vector, because the
deployed 43 x 4096 direction is bound to V4 Flash and cannot be copied to
another model.

Skip: the Qwen3.8 Flash Next cluster (about 45 commits), the Metal batched
decode and speculative batch work (about 25), CUDA/DSpark tuning, agent and
terminal UI work, docs and QA bookkeeping. Skip means the intent need not
survive a conflict. It does not mean the commit is dropped; a merge of
`origin/main` brings all of it in anyway.

Conflict surface, from a dry-run merge (no working tree touched): `Makefile`,
`ds4.c`, `ds4_server.c`, `rocm/ds4_rocm_runtime.cuh`. Everything else merges
clean. The conflict policy in `docs/UPSTREAM.md` still applies: local wins in
`rocm/`, `metal/`, `ram-disk-*.sh`, `ds4_ssd.*`, upstream wins in `tests/`,
`third_party/`, and `ds4.c` hunks are resolved by re-applying our perf intent on
top of upstream's version.

## Repository hygiene

Loose ends I found, in rough order of annoyance:

1. `docs/UPSTREAM.md` is stale. It says "11 local commits ahead, 131 behind" and
   its delta table lists 11 commits; the real numbers today are 29 and 156, and
   the table predates the test and documentation commits. The file itself warns
   to distrust its numbers, and it is right to.
2. Untracked and not ignored: `build-rdna4.log`, `smoke-rdna4-22gb.log`,
   `ds4-server.log`, `download-q2-imatrix.log`, `kv-cache/`, `.deps/`. Either
   commit the logs worth keeping (the smoke log backs a documented number) or
   add them to `.gitignore`, because `git status` is what intake step 1 gates on.
3. Stale branches: `pr-558`, `pr-558-rebase-main`, `pr-558-rebase-main-2`,
   `fix/rocm-elastic-expert-slabs` are earlier rebases of this same work, fully
   superseded. `perf/gfx1201-bench` is parked in a pi worktree at `575a4c7`.
   Local `main` is 465 behind origin. I have not deleted anything.
4. The evidence inventory in `docs/PERFORMANCE_REFERENCE.md` lists open items
   that are still open: the expert-arena sweep was single-run, the 10 to 16 GiB
   model-span range is unmapped, the vision numbers come from one session and
   one image.

## Open items and risks

- The merge with `origin/main` is not optional forever. The queue already
  contains two correctness fixes in models we serve (GLM attention, GLM SSD
  cache priorities) and the memory-budgeting work our config silently overlaps.
  Delaying costs conflict resolution time, since upstream rewrites the same
  files we rewrote.
- `9d9e129` needs a discrete-GPU review. Its admission layer does nothing on the
  R9700, but its arena chunking and Q8 reserve changes do. Our 32 GiB defaults
  (`cc0ef07`) and the shared KV/expert budget (`3d53ea9`) were tuned against the
  old chunking.
- The patched ROCr runtime is single-homed in gitignored `misc/`. The patch
  (`rocm/patches/`) is tracked now, which is good; the build is reproducible but
  tedious.
- `docs/ROCM_STREAMING_PLAN.md` ends with a planned "next bounded
  implementation pass" (asynchronous timeline, long-context indexer and
  attention, targeted transfer benchmark, SDMA runtime investigation) that has
  not been started. The SDMA assertion is still only worked around by disabling
  SDMA, at a 10% decode cost.
- If DeepSeek v4.1 arrives in `gguf/`, the refusal vector, the vision encoder
  pairing and the expert cache defaults all need re-checking before it goes near
  the service.

## Method note

Everything stated here was checked against the tree today: branch and fetch
state with git, deployment with `pi-inference status` and the systemd unit,
weights with `ls` and `download_model.sh`, the steering vector's JSON sidecar,
upstream commits with `git show`, and the merge surface with a read-only
`git merge-tree` simulation. No merge, rebase, checkout or delete was performed.
