# Evidence

The small artifacts behind the measurements quoted in `ROCM_STREAMING_PLAN.md`,
`docs/PERFORMANCE_REFERENCE.md` and `docs/DS4_ENGINE.md`. They are copied verbatim, with their
paths preserved, from run directories that git does not track: the trees under `misc/` are
excluded by `.gitignore:42`, and the `/tmp/ds4-*` scaffolding is ephemeral. The `/tmp` half is not
hypothetical: `/tmp/ds4-{generation,deeper,next}-perf` and `/tmp/ds4-perf-tests`, which held the
frozen reference binaries and the `replay.py` harness, were already gone when this directory was
created.

Copied on 2026-09-29. What is here is `*.json`, logs up to 64 KB, and experiment `*.patch` files.
What is deliberately not here: the multi-gigabyte ROCr build and source trees, logs above 64 KB,
the model files, and the lost `/tmp` baselines.

## What each directory backs

| Path | Backs |
|---|---|
| `misc/rocm-vision-profile-20260917/` | The vision profiling session: `short-profile/`, `long-profile/`, `long-aa/`, `sdma-short/`, the cache-frequency screen, and the cache experiment patch. Also `short-replay/` with the frozen `request.json`, `metadata.json` and `provenance.json`, which is what makes a replay reproducible. |
| `misc/rocm-timeline-20260918/` | The SDMA timeline work: `short-trace/` results, `pcie-topology.txt`, `amd_blit_sdma` sources under review, plus `implementation.patch`, `final-checkout.patch` and the two patches marked rejected (`rejected-online-attention.patch`, `rejected-indexer-wave-rows.patch`) — the negative results, which are usually the ones lost. |
| `misc/rocm-prefetch-20260919/` | The prefetch experiments: four `fixed-*` runs with their `results.json`, `long-layer-partition.json`, and `draft-sdma-pendingbytes.patch`, an earlier draft of the ROCr patch that now lives in `rocm/patches/`. |
| `misc/ab-server-prefill-20261006/` | The merged-versus-baseline server prefill A/B behind `ROCM_STREAMING_PLAN.md`'s server-path gap section: identical request and flags, both response bodies, both server logs. |
| `rescued-from-tmp/` | The crash-analysis tooling, which existed only in `/tmp`: five gdb scripts used to symbolise the ROCr abort from the core dump, the two ring-state logs from the SDMA investigation, and the ROCr ABI symbol dumps (`local.syms`, `system.syms`) that compare the project-local prefix against `/opt/rocm`. |

## Reading a run

Every run directory carries the same shape, so a number can be traced back to its conditions:

- `metadata.json` — what ran, on which binary and model, with which environment
- `provenance.json` — the binaries and flags, sometimes hashes
- `results.json` — the measurements themselves
- `server.log` — the server's own view, when the run used one

When adding a measurement, copy those files here rather than the whole tree, and leave the run
directory in `misc/` where it is. If a result matters and its only copy is under `/tmp`, it does
not exist yet.
