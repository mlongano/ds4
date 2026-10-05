# Upstream intake

How to track `antirez/ds4`, judge whether upstream has anything worth taking, and land it
without damaging the local work. Code philosophy and per-change test gates: `AGENT.md`.
Release gates: `QA_BEFORE_RELEASES.md`. This file is only about upstream.

## Where we stand

- Upstream: <https://github.com/antirez/ds4>, branch `main`. This repo is a fork whose
  delta is the ROCm/RDNA4 port and the RAM-disk SSD streaming mode.
- At last writing: 29 local commits ahead, 156 behind (merge base `b0a147a`, 2026-09-02;
  refreshed 2026-10-05). Upstream moves fast; distrust these numbers and refresh first:

  `git fetch origin && git rev-list --count origin/main..HEAD && git rev-list --count HEAD..origin/main`

## What we carry (the local delta)

Every commit below must survive every merge. `git log --oneline origin/main..HEAD` is the
live inventory; at last writing:

| Area | Commits | Files |
|---|---|---|
| ROCm/RDNA4 port (WMMA fallback, VRAM reserve) | `85ec136` | `rocm/`, `ds4_rocm.*`, `Makefile` |
| RAM-disk SSD streaming | `318b3d5`, `d98551b`, `45eafee` | `ram-disk-*.sh`, `ds4_ssd.*` |
| ROCm perf (prefill arenas, expert transfer pipeline, pinned expert pointers, shared KV/expert VRAM) | `3d53ea9`, `cc0ef07`, `78dce0e`, `df7cc08`, `35bacb1`, `c229363` | `ds4.c`, `rocm/` |
| Decode tracing and microbenchmarks | `e91fdaf`, `9101e0c`, `6557dd8`, `613c00b` | `rocm/`, `tests/`, `Makefile` |
| Metal TP portability | `575a4c7` | `ds4.c`, `metal/` |
| Local ROCr runtime launch fix | `a825b94` | `ds4-server-launch.sh` |
| Bench sweeps | `42e19db` | `bench-*.sh` |
| Documentation and rescued evidence | 12 `docs:` commits | `docs/`, `evidence/` |

- The tree may also carry uncommitted ROCm work. Intake starts from a clean tree:
  `git status --porcelain` must print nothing, or stop and get it committed first.
- `misc/rocm-local-runtime-fixed-prefix/lib` is a **published interface**: qwen-flash loads
  its ROCr from there (`../qwen-3.8-flash/run-mtp-pr-test.sh`). Never move or delete it in
  the same change that does not move that launcher.

## Deciding whether to take something

Take a change when it is at least one of:

- a fix for a bug or misbehaviour we have hit or that sits in our paths (loading, KV
  lifetime, SSD streaming, session/disk-cache policy),
- a correctness improvement (AGENT.md: correctness before speed),
- a measurable win on the R9700 paths, proven by the comparison below.

Skip a change that is only a refactor of code our delta already rewrites (it buys conflicts,
not value), or that contradicts the Goals/Quality Rules in `AGENT.md` (no C++, no permanent
semantic flags, no slop).

Completion criterion: a written take/skip verdict, with reason, for **every** commit in
`git log --oneline HEAD..origin/main` — not for the summary the agent remembers.

## Procedure

1. **Baseline the current tree.** Build (`make rdna4`), run the gates in step 4, and record
   speed numbers. → complete when: gates pass and numbers are written down, so any failure
   after the merge is the merge's fault.
2. **Triage.** `git fetch origin`, then the verdict per commit above. The verdict file
   for the 2026-10-05 intake is `docs/UPSTREAM-INTAKE-2026-10-05.md`; the reasoning
   behind it is in `docs/STATE_OF_THE_CODEBASE_MiMo-V2.6-Pro_2026-10-05.md`.
   → complete when: every incoming commit has a verdict.
3. **Merge on a branch.** `git switch -c intake-$(date +%Y%m%d) && git merge origin/main`.
   Never rebase the local history: the 11 commits are referenced by `local-models` handoff
   docs and qwen-flash's ROCr prefix. Conflict policy: local wins in `rocm/`, `metal/`,
   `ram-disk-*.sh`, `ds4_ssd.*`; upstream wins in `tests/`, `third_party/`; `ds4.c` hunks are
   resolved by re-applying the perf intent of the matching local commit on top of upstream's
   version. → complete when: merge commit exists and `git status --porcelain` is empty.
4. **Gates.** `make rdna4`, then `make test-rocm`, `make test-glm53-kda-rocm`,
   `make dspark-acceptance`, `python3 tests/test_rocm_server_regression.py`, and the speed
   regression checks in `CONTRIBUTING.md`. → complete when: all exit 0 and speed is at or
   above the step-1 numbers; an unexplained regression is a failure, same as unexplained
   drift in `AGENT.md`.
5. **Deploy.** Save the previous binary (`cp ds4-server ds4-server.prev-$(date +%Y%m%d)`),
   then `pi-inference ds4`. → complete when: `pi-inference status` shows mode `ds4` and a
   smoke request through `127.0.0.1:8000/v1` answers.
6. **Record.** Numbers, verdicts, and the merge reference in the merge commit message and,
   if a gate changed meaning, in `QA_BEFORE_RELEASES.md`. → complete when: a reader of the
   commit message alone knows what was taken, skipped, and measured.

## Rollback

Restore `ds4-server.prev-*` over `ds4-server` and `pi-inference ds4` again, or rebuild from
the pre-intake commit. The model files are untouched by any of this.

## When to stop and ask

- Validation that needs distributed inference or the CUDA machine (`AGENT.md`, Testing).
- A conflict resolution that would silently drop one of the local commits' intent.
