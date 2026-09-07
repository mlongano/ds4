#!/usr/bin/env python3
"""Compile the actual production worker against a CPU-only asynchronous-copy model.

The worker and completion bodies are taken verbatim from the ROCm runtime.
Artifacts stay in
/tmp for inspection; no model or ROCm installation is required.
"""
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / "rocm/ds4_rocm_runtime.cuh").read_text()
start = source.index("static int cuda_stream_read_job_upload(")
end = source.index("static void *cuda_stream_read_worker(void *arg)", start)
worker = source[start:end]
out = Path(tempfile.mkdtemp(prefix="ds4-pipeline-unit-"))
(out / "ds4_stream_pipeline_under_test.h").write_text(worker)
subprocess.run(["cc", "-std=c99", "-D_POSIX_C_SOURCE=200809L", "-Wall", "-Wextra", "-Werror",
                "-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-g", "-pthread",
                "-I", str(out), str(root / "tests/test_rocm_stream_pipeline.c"),
                "-o", str(out / "test")], check=True)
subprocess.run([str(out / "test")], timeout=60, check=True)
subprocess.run(["cc", "-std=c99", "-Wall", "-Wextra", "-Werror",
                str(root / "tests/test_rocm_stream_policy.c"),
                "-o", str(out / "policy")], check=True)
subprocess.run([str(out / "policy")], timeout=10, check=True)
start = source.index("static int cuda_model_range_upload_parallel(")
end = source.index("static const char *cuda_model_range_ptr_from_fd(", start)
(out / "ds4_model_upload_under_test.h").write_text(source[start:end])
subprocess.run(["cc", "-std=c99", "-Wall", "-Wextra", "-Werror",
                "-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-g",
                "-I", str(out), str(root / "tests/test_rocm_model_upload.c"),
                "-o", str(out / "bulk")], check=True)
subprocess.run([str(out / "bulk")], timeout=60, check=True)
start = source.index('extern "C" int ds4_rocm_release_prefill_model_ranges(void) {')
end = source.index("static void cuda_model_range_release_all(void)", start)
# Only strip C linkage syntax; compile the production function body verbatim.
(out / "ds4_decode_reset_under_test.h").write_text(source[start:end].replace('extern "C" ', '', 1))
subprocess.run(["cc", "-std=c99", "-Wall", "-Wextra", "-Werror", "-pthread",
                "-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-g",
                "-I", str(out), str(root / "tests/test_rocm_decode_model_reset.c"),
                "-o", str(out / "reset")], check=True)
subprocess.run([str(out / "reset")], timeout=60, check=True)
print("Artifacts:", out)
