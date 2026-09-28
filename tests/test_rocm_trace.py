#!/usr/bin/env python3
"""Test bounded trace collection without a ROCm installation or GPU."""
import copy
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
sys.dont_write_bytecode = True
from rocm_trace_summary import summarize, union_us

root = Path(__file__).resolve().parents[1]
out = Path(tempfile.mkdtemp(prefix="ds4-trace-unit-"))
source = (root / "rocm/ds4_rocm_trace.cuh").read_text()
# These APIs would turn a trace into a scheduling change.
assert not re.search(r"\b(?:cuda|hip)(?:DeviceSynchronize|StreamSynchronize|EventSynchronize|StreamWaitEvent)\s*\(", source)
subprocess.run(["cc", "-std=c99", "-D_POSIX_C_SOURCE=200809L", "-Wall", "-Wextra", "-Werror",
                "-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-g", "-pthread",
                str(root / "tests/test_rocm_trace.c"), "-lm", "-o", str(out / "test")], check=True)
for mode in ("disabled", "invalid-window", "existing-file", "success", "threads", "capacity",
             "unclosed", "create-failure", "record-failure", "not-ready", "query-failure",
             "elapsed-failure", "nonfinite", "operation-failure"):
    path = out / (mode + ".json")
    subprocess.run([str(out / "test"), mode, str(path)], check=True, timeout=30)
    if mode in ("disabled", "invalid-window"):
        assert not path.exists()
        continue
    if mode == "existing-file":
        assert path.read_text() == "keep\n"
        continue
    doc = json.loads(path.read_text())
    metadata, events = doc["metadata"], doc["traceEvents"]
    assert len(events) == metadata["written"] <= metadata["records"] <= 64
    assert all(event["name"] not in ("unrecorded", "outside-window", "outside-layer") for event in events)
    assert all(event["dur"] >= 0 for event in events)
    if mode in ("success", "threads"):
        assert metadata["captured_tokens"] == 2
        assert not any(metadata[key] for key in ("errors", "incomplete", "dropped"))
        assert metadata["written"] == metadata["records"]
        assert {event["args"]["position"] for event in events} == {101, 102}
    if mode == "success":
        assert {event["cat"] for event in events} == {"gpu_stream", "cpu"}
        assert any(event["name"] == "upload" and event["tid"] == 7 for event in events)
        good = copy.deepcopy(doc)
        summary = summarize(doc)
        assert summary["captured_tokens"] == 2
        assert summary["upload_GiB_per_token"] == 32 / 2**30
        assert "idle" in summary["limits"]
    elif mode == "threads":
        assert len([event for event in events if event["name"] == "read"]) == 32
    elif mode == "capacity":
        assert metadata["records"] == 64 and metadata["dropped"] > 0
    elif mode in ("unclosed", "not-ready"):
        assert metadata["incomplete"] > 0
    elif mode in ("create-failure", "record-failure", "query-failure", "elapsed-failure", "nonfinite"):
        assert metadata["errors"] > 0
    elif mode == "operation-failure":
        assert any(not event["args"]["ok"] for event in events)
    print(mode, "PASS")
assert union_us([]) == 0
assert union_us([(0, 4), (1, 3), (3, 5), (8, 9)]) == 6
for failure in ("errors", "incomplete", "dropped", "lost", "window", "nonfinite", "negative", "operation"):
    doc = copy.deepcopy(good)
    if failure in ("errors", "incomplete", "dropped"):
        doc["metadata"][failure] = 1
    elif failure == "lost":
        doc["metadata"]["written"] -= 1
    elif failure == "window":
        doc["metadata"]["requested_tokens"] += 1
    elif failure in ("nonfinite", "negative"):
        doc["traceEvents"][0]["dur"] = float("nan") if failure == "nonfinite" else -1
    else:
        doc["traceEvents"][0]["args"]["ok"] = False
    try:
        summarize(doc)
    except ValueError:
        pass
    else:
        raise AssertionError(f"Summary accepted {failure}")
print("Trace-summary acceptance and interval-union tests: PASS")
print("Artifacts:", out)
