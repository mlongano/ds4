# Patches to external projects this fork depends on

A patch here exists because the fork cannot run without it. It is kept as a file, in git, because
the source tree it was produced in does not live in git: the build trees for these projects sit
under `misc/`, which `.gitignore:42` excludes, so they have no history and no remote copy.

## `0001-rocr-wrap-final-sdma-tracker-half-word.patch`

Applies to `ROCm/rocm-systems`, `projects/rocr-runtime/runtime/hsa-runtime/core/inc/amd_blit_sdma.h`
(six insertions).

**What it fixes.** On Sep 7 2026 `ds4-server` aborted twice mid-decode in the streaming upload
path, with `stl_vector.h:1253`, `__n < this->size()`, inside `libhsa-runtime64`. Symbolised from
the core: `rocr::AMD::BlitSdma<true>::PendingBytes()` at `amd_blit_sdma.cpp:1183`. The ring index
is masked into an 8 MiB ring and then rounded up to a `uint64_t` index, so an offset of `0x7ffffc`
yields index `1048576`, exactly one past the end of a vector holding 1048576 entries. The patch
maps that ceil-converted final four-byte offset to slot zero, which is the first aligned word of
the next ring cycle, and leaves the checked access in place for every other invalid index.

The assertion speaks at all because this build carries the distribution's `-D_GLIBCXX_ASSERTIONS`
(see the flags below). Without it the same index would have read out of bounds silently.

**Provenance.** Clone of <https://github.com/ROCm/rocm-systems.git>, branch
`ds4-sdma-pendingbytes-wrap-fix` (local name, pushed nowhere), based on the upstream branch
`rocm-7.2.4`. The patch commit is `1a2897e7ad2d26c631db52054c674f921252a9ce`, and it is on no
remote branch and not an ancestor of `rocm-7.2.4`. A second checkout under
`misc/rocm-local-runtime` carries the same commit, which duplicates the exposure rather than
insuring it.

**How the loaded library is built.** A plain CMake build of that tree, configured with the
distribution's compiler and flags as recorded in `misc/rocm-local-runtime-fixed-build/CMakeCache.txt`:

```bash
cmake -S misc/rocm-local-runtime-fixed-source -B misc/rocm-local-runtime-fixed-build \
  -DCMAKE_INSTALL_PREFIX=/media/NVME_DATA/MOUNTS/Models/ds4/misc/rocm-local-runtime-fixed-prefix \
  -DBUILD_SHARED_LIBS=ON -DIMAGE_SUPPORT=ON -DCMAKE_BUILD_TYPE=None
cmake --build misc/rocm-local-runtime-fixed-build --target install
```

`CMAKE_BUILD_TYPE=None` is deliberate: it keeps the distribution `CXXFLAGS`, which are
`-march=x86-64 -mtune=generic -O2 -pipe -fno-plt -fexceptions -Wp,-D_FORTIFY_SOURCE=3
-Wformat -Werror=format-security -fstack-clash-protection -fcf-protection
-fno-omit-frame-pointer -mno-omit-leaf-frame-pointer -Wp,-D_GLIBCXX_ASSERTIONS -DNDEBUG`.
The install manifest has 37 entries: `include/hsa/`, `include/hsakmt/`, `lib/libhsa-runtime64.so*`,
`lib/libhsakmt.a` and CMake package files. Only `lib/` matters at runtime.

**Rebuilding it.** The patch is the only irreplaceable part and it is in git now; the library itself
rebuilds from upstream plus that patch:

```bash
git clone --filter=blob:none --branch rocm-7.2.4 https://github.com/ROCm/rocm-systems.git rocm-systems
git -C rocm-systems apply ds4/rocm/patches/0001-rocr-wrap-final-sdma-tracker-half-word.patch
cmake -S rocm-systems -B rocm-systems-build \
  -DCMAKE_INSTALL_PREFIX=<prefix> \
  -DBUILD_SHARED_LIBS=ON -DIMAGE_SUPPORT=ON -DCMAKE_BUILD_TYPE=None
cmake --build rocm-systems-build --target install
```

`rocm-7.2.4` is an upstream tag (`b12dffd90a41dc9486aadfa039b635a4fb19a763`), and the patch applies
cleanly to it: checked 2026-09-29 by applying it to that tag's `amd_blit_sdma.h`, which reproduced
`misc/rocm-local-runtime-fixed-source`'s file byte for byte.

The loaded binary was built with **GCC 16.2.1** (`/usr/bin/c++`), a detail that lives only in the
gitignored `misc/rocm-local-runtime-fixed-build/CMakeCache.txt`. A rebuild on another compiler or
with different flags still gives a working library, but it will not hash to `1009e6f5…`; re-record
the new SHA-256 wherever the old one is quoted (`~/.local/opt/rocr-r9700/README.md`, the qwen
project's `PROVENANCE.md`) and re-run the ROCm smoke tests before trusting it.

**What loads it.** `ds4-server-launch.sh` puts `misc/rocm-local-runtime-fixed-prefix/lib` first on
`LD_LIBRARY_PATH` and **exits with an error if the prefix is missing**, and qwen-flash's
`../qwen-3.8-flash/run-mtp-pr-test.sh` and `sweep-longcontext.sh` load the same prefix, so both
servers run one ROCr build. `UPSTREAM.md` calls the prefix a published interface for that reason.

**Checksums and the backup copy.**

```
1009e6f51ba351bf632867effc63bb0a30de2b17539b46a6150de781d0c3ec9f  lib/libhsa-runtime64.so.1.18.0
32e2b7683a5ccb8f4529315f17b6c8fa69aa84481772b7bdc1ef1ad58aaef90a  this patch
```

A copy of the built prefix, this patch and the checksums is kept outside the repository at
`~/.local/opt/rocr-r9700/`, so `git clean -xdf` in the repository cannot take the runtime with it.
`ds4-server-launch.sh` still points at the repository prefix; qwen-flash's `run-mtp-pr-test.sh` and
`sweep-longcontext.sh` resolve `$FIXED_ROCR`, then the repository prefix, then this copy, and refuse
when none of them exists (2026-09-29). Repointing every launcher at the stable copy would remove the
dependency on a gitignored directory entirely, and is still the obvious next step.
