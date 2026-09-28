#!/usr/bin/env python3
"""Sequential local-server replay. Never starts/stops a systemd service.

Pass a frozen OpenAI request JSON. Each sample uses a fresh process and empty
working directory; temperature/tool/reasoning settings stay in the request.
Compare complete generated fields, ignoring transport chunks and random IDs.
"""
import argparse
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import tempfile
import time
from rocm_streaming_regression import vmstat


def summarize_events(events, minimum_tokens=1):
    output = {"content": "", "reasoning_content": "", "tools": {}}
    first = first_text = None
    usage = finish = None
    for event in events:
        data, seconds = event["data"], event["seconds"]
        if not math.isfinite(seconds) or seconds < 0 or "error" in data:
            raise AssertionError("Invalid SSE event")
        if data.get("usage"):
            usage = data["usage"]
        for choice in data.get("choices", []):
            delta = choice.get("delta", {})
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
            if any(delta.get(k) for k in ("content", "reasoning_content", "tool_calls")):
                if first is None:
                    first = seconds
            for key in ("content", "reasoning_content"):
                output[key] += delta.get(key) or ""
            if delta.get("content") and first_text is None:
                first_text = seconds
            for call in delta.get("tool_calls", []):
                dest = output["tools"].setdefault(str(call["index"]), {"name": "", "arguments": ""})
                for key in ("name", "arguments"):
                    dest[key] += call.get("function", {}).get(key) or ""
    if finish not in ("stop", "length", "tool_calls") or not usage:
        raise AssertionError("Missing successful finish/usage")
    if usage.get("completion_tokens", 0) < minimum_tokens or not any(output.values()):
        raise AssertionError("Empty or insufficient generation")
    return output, {"usage": usage, "finish": finish,
                    "first_output_seconds": first, "first_text_seconds": first_text}


def generation_metrics(log, tokens):
    rows = re.findall(r"gen=(\d+).*?decoding chunk=[\d.]+ t/s avg=([\d.]+) t/s ([\d.]+)s", log)
    if not rows or int(rows[-1][0]) != tokens:
        raise AssertionError("Missing completed generation timing")
    count, rate, elapsed = rows[-1]
    count, rate, elapsed = int(count), float(rate), float(elapsed)
    if not math.isfinite(rate) or not math.isfinite(elapsed) or rate <= 0 or elapsed <= 0:
        raise AssertionError("Invalid generation timing")
    result = {"generation_tps": rate, "generation_seconds": elapsed}
    # Exclude the first logged interval (normally 50 tokens), including the
    # initial cache transition. This is not the CLI benchmark's one-token cut.
    first_count, _, first_elapsed = rows[0]
    if count > int(first_count) and elapsed > float(first_elapsed):
        result["post_first_interval_tps"] = (count - int(first_count)) / (elapsed - float(first_elapsed))
    return result


def sample_environment(overrides, inherited, profile=False):
    env = {k: v for k, v in inherited.items() if not k.startswith("DS4_")}
    env.update(LD_LIBRARY_PATH="/opt/rocm/lib:" + env.get("LD_LIBRARY_PATH", ""),
               DS4_ROCM_STREAM_FREE_RESERVE_GB="2", DS4_SSD_AUTO_CACHE_PCT="65")
    if profile:
        env.update(DS4_SERVER_DECODE_PROFILE="1", DS4_ROCM_STREAM_CACHE_STATS="1",
                   DS4_ROCM_STREAM_READ_PROFILE="1")
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep or not (key.startswith("DS4_") or key == "HSA_ENABLE_SDMA"):
            raise ValueError("Environment overrides require DS4_NAME=value or HSA_ENABLE_SDMA=0/1")
        if key == "HSA_ENABLE_SDMA" and value not in ("0", "1"):
            raise ValueError("HSA_ENABLE_SDMA must be 0 or 1")
        env[key] = value
    return env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--candidate", default="./ds4-server")
    p.add_argument("--model", default="/mnt/ram/ds4flash.gguf")
    p.add_argument("--vision")
    p.add_argument("--steering-file")
    p.add_argument("--steering-scale", type=float, default=3)
    p.add_argument("--candidate-env", action="append", default=[])
    p.add_argument("--reference-env", action="append", default=[])
    p.add_argument("--profile", action="store_true")
    p.add_argument("--tokens", type=int, default=384)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--port", type=int, default=18000)
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--output")
    a = p.parse_args()
    if a.tokens < 1 or a.repeats < 1 or a.seed <= 0 or a.timeout <= 0:
        p.error("Tokens, repeats, seed and timeout must be positive")
    try:
        envs = {kind: sample_environment(getattr(a, kind + "_env"), os.environ, a.profile)
                for kind in ("reference", "candidate")}
    except ValueError as exc:
        p.error(str(exc))
    payload = json.loads(Path(a.request).read_text())
    payload.update(stream=True, stream_options={"include_usage": True},
                   seed=a.seed, max_tokens=a.tokens)
    out = Path(a.output) if a.output else Path(tempfile.mkdtemp(prefix="ds4-server-regression-"))
    if a.output:
        out.mkdir(parents=True)
    binaries = {k: str(Path(getattr(a, k)).resolve()) for k in ("reference", "candidate")}
    metadata = {"options": vars(a), "page_size": os.sysconf("SC_PAGE_SIZE"),
                "binary_sha256": {k: hashlib.sha256(Path(v).read_bytes()).hexdigest()
                                  for k, v in binaries.items()}}
    (out / "request.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    records = []
    for rep in range(a.repeats):
        outputs = {}
        order = ("reference", "candidate") if rep % 2 == 0 else ("candidate", "reference")
        for kind in order:
            work = out / f"{rep}-{kind}"
            work.mkdir()
            # Refuse an occupied port instead of sending to someone else's server.
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", a.port))
            env = envs[kind]
            args = [binaries[kind], "-m", str(Path(a.model).resolve()), "--ctx", "262144",
                    "--host", "127.0.0.1", "--port", str(a.port), "--ssd-streaming",
                    "--ssd-streaming-cold", "--ssd-streaming-cache-experts", "18GB"]
            if a.vision:
                args += ["--vision", str(Path(a.vision).resolve())]
            if a.steering_file:
                args += ["--dir-steering-file", str(Path(a.steering_file).resolve()),
                         "--dir-steering-ffn", str(a.steering_scale)]
            before, started = vmstat(), time.monotonic()
            record = {"name": work.name, "args": args,
                      "ds4_environment": {k: v for k, v in env.items() if k.startswith("DS4_")},
                      "hsa_environment": {k: env[k] for k in ("HSA_ENABLE_SDMA",) if k in env}}
            events, error, conn = [], None, None
            with (work / "server.log").open("wb") as log:
                proc = subprocess.Popen(args, env=env, cwd=work, stdout=log, stderr=log)
                try:
                    deadline = time.monotonic() + a.timeout
                    while True:
                        if proc.poll() is not None:
                            raise RuntimeError("Server exited during startup")
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Server startup timeout")
                        try:
                            health = http.client.HTTPConnection("127.0.0.1", a.port, timeout=1)
                            try:
                                health.request("GET", "/v1/models")
                                response = health.getresponse()
                                response.read()
                                if response.status == 200:
                                    break
                            finally:
                                health.close()
                        except OSError:
                            time.sleep(.1)
                    start = time.monotonic()
                    deadline = start + a.timeout
                    conn = http.client.HTTPConnection("127.0.0.1", a.port, timeout=a.timeout)
                    conn.request("POST", "/v1/chat/completions", json.dumps(payload),
                                 {"Content-Type": "application/json"})
                    response = conn.getresponse()
                    if response.status != 200:
                        raise RuntimeError(f"HTTP {response.status}: {response.read()[:1000]!r}")
                    done = False
                    while True:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError("Generation timeout")
                        if conn.sock is not None:
                            conn.sock.settimeout(remaining)
                        line = response.readline()
                        if not line:
                            break
                        if line.strip() == b"data: [DONE]":
                            done = True
                            break
                        if line.startswith(b"data: "):
                            events.append({"seconds": time.monotonic() - start,
                                           "data": json.loads(line[6:])})
                    if not done:
                        raise AssertionError("Truncated SSE response")
                    record["request_seconds"] = time.monotonic() - start
                    outputs[kind], summary = summarize_events(events, min(64, a.tokens))
                    record.update(summary)
                    (work / "output.json").write_text(json.dumps(outputs[kind], indent=2) + "\n")
                except Exception as exc:
                    error = exc
                    record["error"] = repr(exc)
                finally:
                    if conn is not None:
                        conn.close()
                    if proc.poll() is None:
                        proc.send_signal(signal.SIGINT)
                    try:
                        proc.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                        error = error or TimeoutError("Server shutdown timeout")
            after = vmstat()
            record.update(exit_code=proc.returncode, wall_seconds=time.monotonic() - started,
                          system_vmstat_delta={k: after[k] - before[k] for k in before})
            (work / "events.json").write_text(json.dumps(events, indent=2) + "\n")
            if error is None and proc.returncode == 0:
                record.update(generation_metrics((work / "server.log").read_text(),
                                                 record["usage"]["completion_tokens"]))
            records.append(record)
            (out / "results.json").write_text(json.dumps(records, indent=2) + "\n")
            if error or proc.returncode:
                raise RuntimeError(f"{work.name} failed; see {work}") from error
            print(work.name, record["generation_tps"], record.get("post_first_interval_tps"), flush=True)
        if outputs["reference"] != outputs["candidate"]:
            raise AssertionError(f"Generated fields differ for repetition {rep}")
    print("Exact generated-field comparisons passed. Results:", out)


if __name__ == "__main__":
    main()
