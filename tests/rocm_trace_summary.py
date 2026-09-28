#!/usr/bin/env python3
"""Validate a ROCm trace and summarize stream intervals, not kernel busy time."""
from collections import defaultdict
import json
import math
from pathlib import Path
import sys


def union_us(intervals):
    total = 0.0
    end = None
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def summarize(doc):
    metadata, events = doc["metadata"], doc["traceEvents"]
    if any(metadata[key] for key in ("errors", "incomplete", "dropped")):
        raise ValueError("Trace has errors, unfinished events or exhausted storage")
    if not events or metadata["captured_tokens"] != metadata["requested_tokens"]:
        raise ValueError("Trace did not capture the requested decode window")
    if metadata["records"] != len(events) or metadata["written"] != len(events):
        raise ValueError("Trace lost records")
    totals = defaultdict(float)
    positions = set()
    uploads = []
    gpu_intervals = []
    transfer_bytes = 0
    for event in events:
        start, duration = event["ts"], event["dur"]
        if not math.isfinite(start) or not math.isfinite(duration) or duration < 0:
            raise ValueError("Nonfinite or negative trace duration")
        if not event["args"]["ok"]:
            raise ValueError("Trace includes a failed operation")
        category = event["cat"]
        if category not in ("gpu_stream", "cpu") or event["ph"] != "X":
            raise ValueError("Unexpected trace event type")
        positions.add(event["args"]["position"])
        totals[category + "/" + event["name"]] += duration
        if category == "gpu_stream":
            gpu_intervals.append((start, start + duration))
            if event["name"] == "upload":
                uploads.append((start, start + duration))
                transfer_bytes += event["args"]["bytes"]
    if len(positions) != metadata["captured_tokens"] or not gpu_intervals:
        raise ValueError("Missing sampled positions or GPU events")
    n = len(positions)
    upload_union = union_us(uploads)
    covered = union_us(gpu_intervals)
    span = max(end for _, end in gpu_intervals) - min(start for start, _ in gpu_intervals)
    return {
        "captured_tokens": n,
        "position_range": [min(positions), max(positions)],
        "summed_interval_ms_per_token": dict(sorted(((key, value / n / 1000) for key, value in totals.items()), key=lambda item: -item[1])),
        "upload_GiB_per_token": transfer_bytes / n / 2**30,
        "upload_interval_union_ms_per_token": upload_union / n / 1000,
        "upload_GiB_per_union_second": transfer_bytes / 2**30 / (upload_union / 1e6) if upload_union else None,
        "uncovered_gpu_interval_ms_per_token": max(0, span - covered) / n / 1000,
        "setup_ms": metadata["setup_us"] / 1000,
        "limits": "CPU spans overlap GPU work and each other. GPU stream intervals include submission/queueing gaps; uncovered intervals may contain uninstrumented work. Neither is exact kernel busy/device idle time. CPU/GPU clock alignment is approximate.",
    }


if __name__ == "__main__":
    print(json.dumps(summarize(json.loads(Path(sys.argv[1]).read_text())), indent=2))
