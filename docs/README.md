# ds4 documentation

Documentation for this fork: the RDNA4/ROCm port, how the engine is deployed on the machine
that runs it, and how upstream intake is handled. Everything here is maintained by the fork
owner. Upstream's own documentation stays at the repository root for now (`README.md`,
`AGENT.md`, `CONTRIBUTING.md`, `MODEL_CARD.md`, `QA_BEFORE_RELEASES.md`, `STRIXHALO.md`).

Upstream `main` has already moved its copies into `docs/` (`docs/SSD_STREAMING.md`,
`docs/STRIX_HALO.md`, `docs/TESTING.md`, `docs/PERFORMANCE.md`, `docs/SERVER.md` and others).
This branch predates that move. The three files below deliberately avoid those names, so the
next intake merge leaves both sets side by side with no collisions.

| Document | Owns |
|---|---|
| [`ROCM_STREAMING_PLAN.md`](ROCM_STREAMING_PLAN.md) | The measurement record for the streaming work: long-context baseline, cost split, the SDMA workaround, the rejected cache-policy trial, the decode trace, and the next bounded pass with its acceptance and stopping rules |
| [`UPSTREAM.md`](UPSTREAM.md) | What the local delta is, how to judge what upstream is worth taking, and the refresh-first intake procedure with its completion criteria |
| [`DS4_ENGINE.md`](DS4_ENGINE.md) | What the engine is, how the R9700 port was done, the on-demand RAM disk, coordination through `pi-inference`, the public endpoint, and context/cache tuning |
| [`PERFORMANCE_REFERENCE.md`](PERFORMANCE_REFERENCE.md) | The measured landscape: hardware characterisation, the prefill/decode frontier, accepted and rejected tuning with the numbers that decided them, known failure modes and their workarounds, the live environment knobs, and the inventory of evidence that is not in git |
| [`SESSION_SUMMARY_2026-09-04_2026-09-17.md`](SESSION_SUMMARY_2026-09-04_2026-09-17.md) | Narrative of the first two weeks: the rebase, the performance work one variable at a time, why the server was slower than the CLI, the September 7 crash, the thinking loops, the upstream and model investigations |
| [`HANDOFF_VISION_EXP_PROFILING.md`](HANDOFF_VISION_EXP_PROFILING.md) | The September 17 task statement for the bounded profiling pass, and the protocol for resuming after a killed turn without redoing reconnaissance |

Two directories outside `docs/` hold what those documents cite:

- [`../evidence/`](../evidence/README.md) — the small artifacts (results, metadata, provenance,
  logs, experiment patches) copied out of the untracked run directories under `misc/`.
- [`../rocm/patches/`](../rocm/patches/README.md) — the local patch to ROCr that the loaded
  runtime was built from, its recipe, and its checksums.

Paths quoted in these documents are relative to the repository root (`rocm/...`, `ds4.c`,
`ram-disk-up.sh`, `tests/...`) unless a document says otherwise. Absolute paths that begin
with `/media/NVME_DATA/MOUNTS/Models/ds4` refer to the checkout on the machine this fork is
developed on.

## Documentation that lives outside this repository

- `~/Develop/MACHINE_LEARNING/local-models/` holds the host and ecosystem documentation:
  `PI_INFERENCE_CONTROL_PLANE.md` (the manager, modes, leases and endpoints),
  `ECOSYSTEM.md` (how the router, Studio, ds4 and the panel share one GPU),
  `QWEN_FLASH_CONTROL_PLANE.md`, `CONTROL_WEBAPP_PLAN.md`. These describe the machine around
  the engine rather than the engine, and they link back to the files above.
- `AGENT.md` at the repository root is the working-rules file and stays there by convention;
  it points at `docs/UPSTREAM.md`.
