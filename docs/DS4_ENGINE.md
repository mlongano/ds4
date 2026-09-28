# ds4.c — DeepSeek V4 Flash Native Inference Engine

A separate inference stack from the `pi-llama-router` / Unsloth Studio setup described in `ECOSYSTEM.md`. Runs one large MoE model — DeepSeek V4 Flash — that doesn't fit the R9700's 32 GB VRAM even at low quantization, by streaming routed expert weights from storage instead of holding the whole model resident. All three (router, Studio, ds4) share that one GPU and are now coordinated through `pi-inference` (§4) rather than manual `systemctl` juggling.

Repo: `/media/NVME_DATA/MOUNTS/Models/ds4/` (fork of [antirez/ds4](https://github.com/antirez/ds4), branch `pr-558-rebase-main`)

---

## 1. Foundation: what ds4.c is and how it fits the hardware

ds4.c is antirez's standalone C inference engine for DeepSeek V4 and GLM 5.2 — not llama.cpp, no shared code with the router stack. It ships CPU, Metal, CUDA, and ROCm backends from one codebase, built with a plain `Makefile` (no CMake, no external ML framework).

The model in use, DeepSeek V4 Flash at `IQ2XXS` quantization, is an 86,720,111,488-byte (80.76 GiB) GGUF — larger than the R9700's 32 GB VRAM and larger than most of a typical workstation's VRAM at any realistic quant. ds4.c makes this fit by never loading the full model into VRAM at once:

- **Non-routed weights** (attention, shared experts, embeddings, output head — ~8.2 GiB for this model) load once and stay resident on the GPU.
- **Routed experts** (the MoE-specific weights selected per-token by the router, 6.75 MiB each) are *not* preloaded. They're read from the backing file on demand as the model's router selects them, uploaded to the GPU, and kept in a bounded **resident expert cache** sized as a percentage of the GPU's recommended working set (default 80%, tunable via `DS4_SSD_AUTO_CACHE_PCT`, see `ds4_ssd.c`; this deployment runs 65% — see §6).
- When the cache is full, eviction prefers **older layers first** (`cuda_stream_evict_past_layers_first`), falling back to plain LRU by `last_used` timestamp — the intuition being that during autoregressive decode, layers already passed in this forward pass are less likely to be needed again soon than layers still ahead.
- Reads use **`O_DIRECT`** by default to skip the page cache — avoids an extra copy and page-cache churn for every streamed expert, since each read is used once and discarded. Any read that fails direct I/O silently falls back to buffered `pread` for that job only (`cuda_stream_read_job_run` in `rocm/ds4_rocm_runtime.cuh`); it can also be disabled globally with `DS4_ROCM_STREAM_NO_DIRECT=1`.
- A configurable **free-VRAM reserve** (`DS4_ROCM_STREAM_FREE_RESERVE_GB`, 2–64 GiB, default 16) is kept free while growing the cache, to cover decode scratch and transient graph buffers without tripping an allocation failure.

This is why the model — 81 GB on an NVMe drive — can serve requests through a GPU with 32 GB of VRAM at all: only the working set of *currently relevant* expert weights needs to be resident at any moment, and the rest streams in from wherever the backing file lives.

That last point — "wherever the backing file lives" — is the hook the RAM-disk feature (§3) uses: the streaming code doesn't care if the backing file is on an NVMe SSD or in a tmpfs RAM disk, only that it's a regular file it can `pread()` from.

### 1.1 Two ways to run it

| Mode | Binary | Use case |
|---|---|---|
| Server | `ds4-server` | Long-running HTTP server (OpenAI/Anthropic/Responses-compatible API), warm cache persists across requests |
| CLI | `ds4` | One-shot (`ds4 -p "..."`) or **interactive chat REPL** (`ds4` with no args) — no server process, no HTTP, talks to the terminal directly |

Both link the same core streaming/backend code; the CLI is not a client of the server, it's a second front-end over the same engine.

---

## 2. How this fork's ROCm/RDNA4 support was built

### 2.1 The upstream gap

antirez's ROCm backend targets Strix Halo (`gfx1151`), an APU with unified memory. The R9700 is a discrete RDNA4 card (`gfx1201`) with dedicated 32 GB VRAM — a different architecture generation with two consequences:

1. **WMMA intrinsic mismatch.** The hand-written Q8 batch GEMM kernel (`rocm/ds4_rocm_q8.cuh`) uses the gfx11-style intrinsic `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`. RDNA4 changed the WMMA intrinsic signatures, so this fails to compile on gfx1201 (`Cannot select: intrinsic llvm.amdgcn.wmma.f32.16x16x16.f16`).
2. **Reserve sizing assumption.** The free-VRAM reserve was hardcoded to 16 GiB, sized for a unified-memory APU where the OS shares the same pool. On a 32 GB discrete card that reserve alone eats half the VRAM, and SSD streaming fails outright once the expert cache needs to grow past ~12 GB (`cannot keep ... while preserving 16.00 GiB free`).

### 2.2 The fix (PR #558, `cm999club`)

Two small, additive changes, tested on an R9700 32GB (gfx1201), Ubuntu 26.04, ROCm 7.2.2, rocWMMA 2.2.0:

1. **`DS4_ROCM_NO_WMMA`** compile-time guard, added to the `#if` conditions gating the WMMA path in both `rocm/ds4_rocm_matmul.cuh` and `rocm/ds4_rocm_q8.cuh`:
   ```c
   #if (defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)) && !defined(DS4_ROCM_NO_WMMA)
   ```
   When defined, host code falls back to the pre-existing "sharedx" batch kernel instead of the WMMA path. Default builds (gfx1151) are unaffected.
2. **`DS4_ROCM_FREE_RESERVE_MB`** env override for the reserve (superseded — see §2.4).
3. A `make rdna4` convenience target: `strix-halo` with `ROCM_ARCH=gfx1201` and the WMMA guard baked in.

### 2.3 Rebasing onto upstream main

By the time this fork picked the PR up, `origin/main` had moved 5 commits past the PR's original base, including an independent ROCm rewrite (`ef8d923`, "Add ROCm GLM 5.2 support") that touched the same functions. The rebase was done onto a **new local branch** (`pr-558-rebase-main`), leaving the original `pr-558` branch untouched as a fallback — nothing was pushed anywhere.

Two real conflicts, both resolved by hand (a stale `git rerere` cache tried to auto-apply a resolution from an unrelated earlier attempt that silently deleted unrelated upstream features — DSpark support, TP, layer packing, sampling tests — that resolution was rejected and rerere disabled for this rebase):

1. **`Makefile`** — trivial additive merge: PR #558's `rdna4` target plus upstream's newer `.PHONY` list and a `ds4_gpu_args.o` dependency upstream had added to the `ds4` build target.
2. **`rocm/ds4_rocm_runtime.cuh`** — upstream's `ef8d923` had *independently* implemented the same fix as PR #558's reserve override, under a different name and unit: `DS4_ROCM_STREAM_FREE_RESERVE_GB` (GiB granularity, clamped 2–64) instead of `DS4_ROCM_FREE_RESERVE_MB` (MiB, unclamped). Upstream's version was already wired into several other call sites rewritten by later commits (the slab allocator, the q8 fp16 cache release path), so keeping PR #558's older version wasn't viable — the surrounding code had moved on. Upstream's version was kept as-is.

### 2.4 What survived the rebase

Because the reserve feature was fully superseded by upstream's equivalent, this fork's actual surviving contribution is just the WMMA guard — **3 files, 8 insertions, 4 deletions**:

- `Makefile`: `.PHONY` entry + `rdna4` target
- `rocm/ds4_rocm_matmul.cuh`: `!defined(DS4_ROCM_NO_WMMA)` guard (1 line)
- `rocm/ds4_rocm_q8.cuh`: same guard (1 line)

`2048 MB` (the value this fork's `ds4-server.service` used) maps exactly onto `2 GB` under upstream's variable, so no behavioral change — just a rename. All configs that referenced `DS4_ROCM_FREE_RESERVE_MB=2048` were migrated to `DS4_ROCM_STREAM_FREE_RESERVE_GB=2` (systemd unit, `bench-ctx-size.sh`). One exception: `bench-ssd-streaming.sh` originally used `=0` (no clamp existed under the old code, letting that script deliberately push a true zero-reserve worst case); upstream's clamp makes `0` inexpressible, so it was mapped to `2` (the new minimum) — this changes what that specific "cold" benchmark row measures versus older runs.

### 2.5 The build

```bash
cd /media/NVME_DATA/MOUNTS/Models/ds4
make rdna4
```
Produces `ds4`, `ds4-server`, `ds4-bench`, `ds4-eval`, `ds4-agent` linked against `--offload-arch=gfx1201 -DDS4_ROCM_NO_WMMA`.

---

## 3. The RAM-disk on-demand system

### 3.1 Motivation

SSD streaming works, but every cache-miss expert read pays NVMe latency (measured ~4 ms per streamed read under `--ssd-streaming-cold`). System RAM on this machine (125 GB total, typically 70–110 GB free) is roughly 22× faster per the back-of-envelope bandwidth comparison that prompted this — and large enough to hold the entire 80.76 GiB model resident. ds4.c's streaming code has no notion of "SSD" specifically; it just `pread()`s a backing file. Pointing that file at a RAM-backed filesystem (tmpfs) instead of the NVMe partition gets the same streaming architecture with RAM-speed cache misses, no code changes required.

### 3.2 Feasibility result

An earlier attempt (before this system existed) tried exactly this and appeared to fail — the model loaded but prefill errored out. Investigation traced this to a **different, pre-existing bug**: `--ssd-streaming-cold` in CLI (`ds4 -p`) mode was broken for prefill regardless of backing storage; server mode (`ds4-server`) was never affected and worked the whole time. One of the commits pulled in by the §2.3 rebase fixed this — the CLI path now works cleanly against both SSD and RAM-backed files.

Head-to-head, same rebased binary, same prompt, page cache dropped before the SSD run to guarantee a genuinely cold baseline (tmpfs data isn't affected by `drop_caches` — it's not reclaimable page cache, it *is* the storage):

| Metric | SSD (cold) | RAM disk | Speedup |
|---|---|---|---|
| Prefill (27 tokens) | 72.3 s | 23.1 s | ~3.1× |
| Steady-state decode | 4.2–4.8 t/s | 9.4–10.1 t/s | ~2.2× |
| Total (400 tokens generated) | 167.1 s | 68.2 s | ~2.4× |

No errors or crashes in either configuration across sustained generation.

### 3.3 Why `/mnt/ram`, not `/dev/shm`

`/dev/shm` is tmpfs and looks like the obvious choice — `df -h` reports 95 GB available — but it carries an **undocumented per-user quota** on this system (`usrquota` mount option, no `/etc/fstab` entry, no quota tooling installed to introspect or change it). Empirically measured by filling it with `dd` until it failed: the quota caps out at exactly `53,869,187,072` bytes (~50.16 GiB), well under the 80.76 GiB model. `/dev/shm` is a dead end for this use case without root-level quota surgery.

The working solution is a **separate tmpfs mounted at `/mnt/ram`** with an explicit size and no quota:
```bash
sudo mount -t tmpfs -o size=90G,mode=1777 tmpfs /mnt/ram
```
This is not persistent — tmpfs is wiped on unmount or reboot — which is fine, since the whole point of this system is to allocate it on demand and release it when done.

### 3.4 Components

```
ram-disk-up.sh          — allocate: mount tmpfs + copy model in (idempotent, self-cleaning on failure)
ram-disk-down.sh        — release: unmount tmpfs, freeing RAM (idempotent, refuses if in use)
ds4-server-launch.sh    — RAM-first launcher: tries ram-disk-up.sh, falls back to SSD with a warning
ds4-server.service      — the one systemd unit; ExecStart runs the launcher, ExecStopPost releases RAM
ds4-chat                — interactive CLI chat, always RAM-backed, coordinates via pi-inference (§4)
```
The scripts and `ds4-server.service` live in `/media/NVME_DATA/MOUNTS/Models/ds4/` and `~/.config/systemd/user/`. `ds4-chat` lives in `~/.local/bin/` (on `PATH`) since it's a user-facing entry point, not project code.

There used to be a second unit, `ds4-ram-server.service`, coordinated with the SSD-backed one via `Conflicts=`. That's been retired: RAM-vs-SSD is now decided *inside* `ds4-server-launch.sh` on every start, so there's only one systemd identity (`ds4-server.service`) to start, stop, or query, regardless of which backing store ends up serving.

#### `ram-disk-up.sh`

- No-ops the mount step if `/mnt/ram` is already a mountpoint.
- Resolves the model path via `readlink -f ds4flash.gguf` — the repo keeps this as a symlink to whichever quantized GGUF is "current", so the script never hardcodes a filename.
- Copies to `/mnt/ram/ds4flash.gguf.tmp` first, then atomically `mv`s to the final name — a copy that's interrupted partway never appears as a "ready" file.
- Skips the copy entirely if the destination already exists **and** matches the source's exact byte size — this is what makes repeated `start`/`stop` cycles fast after the first allocation in a session, and also what catches a truncated previous copy (wrong size ≠ skip).
- **On any failure after mounting** (copy fails, disk/RAM exhausted mid-copy), a trap removes the partial `.tmp` file and unmounts the tmpfs — but only if *this run* was the one that mounted it; a tmpfs someone else already had mounted is left alone. This matters because the launcher below now calls this unconditionally on every server start: a failed attempt must not leave a half-populated, pointlessly-mounted tmpfs sitting around eating RAM.
- Prints the resolved RAM-backed path to stdout, so callers can do `MODEL=$(./ram-disk-up.sh)`.

#### `ram-disk-down.sh`

- No-ops if `/mnt/ram` isn't mounted.
- Runs `sudo fuser -m /mnt/ram` first — if **any** process still has an open file on that mount (the server, an interactive `ds4` chat session, anything), it refuses and exits 1 rather than yanking the model out from under a live process.
- Otherwise `sudo umount /mnt/ram`, which frees all resident tmpfs pages immediately (not gradually) — measured: system free memory went from ~30 GiB to ~106 GiB the instant the unmount completed.

#### `ds4-server-launch.sh`

The `ExecStart` target for `ds4-server.service`. Tries `ram-disk-up.sh` first; on success, serves from the RAM copy. On **any** failure (out of RAM, mount failure, copy interrupted), it logs a `WARNING:` line to the journal and falls back to the SSD-backed `ds4flash.gguf` instead of refusing to start — a degraded-but-working server beats an outage. CLI flags (`--ctx`, `--port`, `--ssd-streaming`, ...) live in the systemd unit, not the script, and are forwarded straight through to `ds4-server`.

#### `ds4-server.service`

```ini
[Service]
Environment=DS4_ROCM_STREAM_FREE_RESERVE_GB=2
Environment=DS4_SSD_AUTO_CACHE_PCT=65
ExecStart=/media/NVME_DATA/MOUNTS/Models/ds4/ds4-server-launch.sh \
  --ctx 262144 --port 8000 --ssd-streaming --ssd-streaming-cold
ExecStopPost=/media/NVME_DATA/MOUNTS/Models/ds4/ram-disk-down.sh
Restart=on-failure
```
`--ctx` and `DS4_SSD_AUTO_CACHE_PCT` are tuned together, not independently — see §6 for why the second one has to move whenever the first does.
`ExecStopPost` releases the RAM disk on **every** stop, even on a run where the SSD fallback ended up serving instead (`ram-disk-down.sh` no-ops harmlessly when nothing's mounted). One consequence: a plain `systemctl restart` re-copies the full 81 GB on the way back up rather than reusing a still-warm disk, since stop always tears it down first.

#### `ds4-chat`

Wraps the CLI (`ds4`, no server) for interactive use, always RAM-backed. Before touching the GPU it runs `pi-inference stop` (§4) — which now correctly frees the router, Studio, *and* `ds4-server.service` together, and fails closed (refuses to proceed) if a lease is currently held by someone else, rather than an earlier version of this script that only stopped `ds4-server.service` directly and left the router/Studio question unhandled. It then runs `ram-disk-up.sh` unconditionally; unlike the server launcher, it does **not** fall back to SSD on a RAM-allocation failure — an interactive session is a fine place to just surface that error directly rather than silently degrade.

### 3.5 Usage

**On-demand server**, coordinated with the router/Studio (recommended — see §4):
```bash
pi-inference ds4      # stops router+Studio, allocates RAM (or falls back to SSD), starts serving on :8000
pi-inference stop      # stops it (and the others) again, releasing the RAM disk
```

**Direct systemctl** also works, but doesn't stop the router/Studio for you — only use this if you've already freed the GPU some other way:
```bash
systemctl --user start ds4-server.service   # RAM-first (~40s cold, instant if already warm), SSD fallback with a WARNING on failure
systemctl --user stop  ds4-server.service   # drain + stop + release the RAM disk
systemctl --user status ds4-server.service
journalctl --user -u ds4-server.service -f
```

**Interactive chat, no server:**
```bash
ds4-chat                    # on PATH; interactive REPL
ds4-chat -p "..." -n 200    # one-shot, extra ds4 flags pass through
```

**Manual RAM-disk allocation** (e.g. to warm it before a benchmark, independent of the service):
```bash
cd /media/NVME_DATA/MOUNTS/Models/ds4
./ram-disk-up.sh      # idempotent
./ram-disk-down.sh    # idempotent, refuses if something's still using it
```

### 3.6 Operational notes

- **RAM cost while allocated**: ~82 GB resident (the model) plus whatever the running process needs on top (~19 GB VRAM-side for KV cache, buffers, and the streaming expert cache at the current 262K-context config, per the server's own startup log — see §6) — leaves comfortable headroom on a 125 GB machine, but don't run this alongside another RAM-heavy workload (e.g. a large fine-tuning job) without checking `free -h` first.
- **Not persistent by design.** A reboot or an explicit `ram-disk-down.sh` fully releases the RAM; the next `up` re-copies from the NVMe source (~40 s at ~3 GB/s observed).
- **Same GGUF, same correctness.** This is purely a change of backing storage for the identical file the SSD path already streams from — no quantization, precision, or model-behavior difference between the two modes.

---

## 4. Coordinating with the router/Studio via `pi-inference`

ds4-server, `pi-llama-router`, and Unsloth Studio all share the one GPU. Nothing used to stop them from fighting over VRAM at the same time — for a while the only fix was manually `systemctl --user stop`-ing whichever one wasn't wanted before starting ds4. `pi-inference` (full control-plane docs: `PI_INFERENCE_CONTROL_PLANE.md` in dotfiles) now treats `ds4` as a third fully coordinated mode alongside `team` and `studio`:

```bash
pi-inference ds4      # stops router + Studio, starts ds4-server.service, waits for it to actually respond
pi-inference team     # stops Studio + ds4-server.service, starts the router
pi-inference studio   # stops router + ds4-server.service, starts Unsloth Studio
pi-inference stop     # stops all three
pi-inference status   # {"mode": ..., "lease": ..., "services": {"router": ..., "studio": ..., "ds4": ...}}
```

`ds4` is a leasable mode exactly like `team` (`pi-inference acquire --mode ds4 --owner ... --ttl ...`): a remote client's held lease now blocks a local `pi-inference ds4` with a 409 conflict, the same protection `team` already had, instead of a local switch silently evicting them. `ds4-chat` (§3.4) goes through `pi-inference stop` for exactly this reason before grabbing the GPU for its own direct `./ds4` process.

Mode switches can take up to ~90s (a cold ds4 RAM-disk allocation), so `pi-inference` streams live progress to stderr instead of blocking silently:

```
$ pi-inference ds4
pi-inference: switching to ds4 mode...
pi-inference:   waiting for ds4 to become ready...
Inference mode is ds4
```

Endpoint list, auth model, and the lock-free progress mechanism: `PI_INFERENCE_CONTROL_PLANE.md`.

## 5. Public endpoint: `ds4.malo.tn.it`

Mirrors the router's public endpoint (`llm.malo.tn.it`): same wildcard TLS cert, same OpenAI-compatible `/v1` surface, reachable once the host is in `ds4` mode. Unlike `llama-server` (which validates its own `--api-key-file`), `ds4-server` has no built-in authentication — so the nginx vhost gates access itself via `auth_request` against a `pi-inference-manager` endpoint that validates the *same* model API key already used for `llm.malo.tn.it`. No new credential to manage; if the host isn't in `ds4` mode, requests get a clear `503` rather than a hung connection. Full nginx config and deployment steps: `PI_INFERENCE_CONTROL_PLANE.md`.

---

## 6. Context-size tuning

### 6.1 There was real headroom

`./ds4 --inspect` reports the model's **trained context is 1,048,576 tokens (1M)**. The original `--ctx 65536` deployment was using 6.25% of that — the ceiling is VRAM, not the architecture.

DeepSeek's MLA-style attention keeps this cheaper than a naive KV cache would be: the "raw" (uncompressed) window is capped at a fixed 4,352 rows regardless of `--ctx` — it stops growing once `--ctx` exceeds it. Only the *compressed* KV rows grow with context (`ctx/4` for 21 of the model's 43 layers, `ctx/128` for 20 of them), which is why quadrupling `--ctx` didn't quadruple total memory.

### 6.2 The startup memory estimate is optimistic — verify with a real request, not just the planner

Doubling and quadrupling `--ctx` looked safe on paper (the server's own startup "planned" line showed 67% and 75% of total VRAM respectively) but **both crashed with a real out-of-memory error the moment a request came in** — one during prefill, one during decode, zero tokens generated either time. The static plan doesn't account for the dynamic tug-of-war between the expert cache and the growing KV/context-buffer footprint at runtime; only an actual generation request under load surfaces that.

Measured (RAM-backed, same essay-writing prompt, 400 tokens generated each run):

| `--ctx` | `DS4_SSD_AUTO_CACHE_PCT` | Expert cache | Result | Decode speed |
|---|---|---|---|---|
| 65,536 | 80 (old default) | 13.92 GiB | OK | 6.51 t/s |
| 131,072 | 80 (old default) | 13.92 GiB | **OOM, 0 tokens** | — |
| 131,072 | 70 | 10.72 GiB | OK | 6.51 t/s |
| 262,144 | 80 (old default) | 13.92 GiB | **OOM, 0 tokens** | — |
| 262,144 | 65 | 9.14 GiB | OK | 6.40 t/s |
| 262,144 | 55 | 5.95 GiB | OK | 5.70 t/s |

`--ctx` and `DS4_SSD_AUTO_CACHE_PCT` are coupled: raising context without shrinking the cache split reliably OOMs on this hardware, not just theoretically.

### 6.3 Current deployment

`--ctx 262144` + `DS4_SSD_AUTO_CACHE_PCT=65` — 4× the original context, essentially the same decode throughput (6.40 vs 6.51 t/s) as the un-tuned 65,536 baseline, verified through a real `pi-inference ds4` start and chat request, not just the benchmark harness.

Pushing further (524,288+, or the full 1,048,576 native context) would need a much smaller expert cache still and hasn't been tested — extrapolating from how steeply `pct` had to drop for the two steps already measured, it's likely to cost real throughput rather than being free like this jump was. Treat it as a hypothesis to benchmark the same way, not an assumption.
