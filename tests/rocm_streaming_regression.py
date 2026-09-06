#!/usr/bin/env python3
"""Sequential R9700 real-model correctness and performance checks.

Run from the repository root. Results go to a new directory under /tmp unless
--output is supplied. No server is started. Reference/candidate environments
exclude inherited DS4_* tuning; --candidate-env adds an explicit experiment.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time


def vmstat():
    wanted = {"pswpin", "pswpout", "pgmajfault"}
    return {k: int(v) for k, v in
            (line.split() for line in Path("/proc/vmstat").read_text().splitlines())
            if k in wanted}


def compare(reference, candidate):
    # Whole JSON includes every selected token, alternative, logit and logprob.
    for doc in (reference, candidate):
        if not doc.get("steps"):
            raise AssertionError("Empty generation is not a correctness pass")
        for step in doc["steps"]:
            if not step["top_logprobs"]:
                raise AssertionError("Missing logprobs")
            for token in step["top_logprobs"]:
                if not all(math.isfinite(token[k]) for k in ("logit", "logprob")):
                    raise AssertionError("Nonfinite logits/logprobs")
    if reference != candidate:
        raise AssertionError("Reference and candidate logprob JSON differ")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", default="./ds4")
    p.add_argument("--candidate", default="./ds4")
    p.add_argument("--reference-bench", default="./ds4-bench")
    p.add_argument("--candidate-bench", default="./ds4-bench")
    p.add_argument("--reference-compact", action="store_true")
    p.add_argument("--candidate-env", action="append", default=[])
    p.add_argument("--model", default="/mnt/ram/ds4flash.gguf")
    p.add_argument("--output")
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--long-chars", type=int, default=40000)
    p.add_argument("--contexts", type=int, nargs="*", default=[8192, 16384])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--skip-correctness", action="store_true")
    p.add_argument("--skip-bench", action="store_true")
    p.add_argument("--interactive", action="store_true", help="Compare a two-turn CLI transcript")
    p.add_argument("--vision", help="Optional matching vision encoder for image continuation test")
    p.add_argument("--timeout", type=int, default=900)
    a = p.parse_args()
    if a.tokens < 1 or a.repeats < 1 or any(c + a.tokens >= 262144 for c in a.contexts):
        p.error("Invalid token count, repeats, or context frontier")
    out = Path(a.output) if a.output else Path(tempfile.mkdtemp(prefix="ds4-regression-"))
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        p.error("Output already contains results; use a new directory")
    env = {k: v for k, v in os.environ.items() if not k.startswith("DS4_")}
    env["LD_LIBRARY_PATH"] = "/opt/rocm/lib:" + env.get("LD_LIBRARY_PATH", "")
    envs = {"reference": env.copy(), "candidate": env.copy()}
    if a.reference_compact:
        envs["reference"]["DS4_ROCM_DISABLE_STREAMING_DECODE_SELECTED_PTRS"] = "1"
    for item in a.candidate_env:
        key, sep, value = item.partition("=")
        if not sep or not key.startswith("DS4_"):
            p.error("--candidate-env requires DS4_NAME=value")
        envs["candidate"][key] = value
    common = ["-m", a.model, "--ssd-streaming", "--ssd-streaming-cold"]
    records = []
    binaries = {str(Path(path).resolve()) for path in
                (a.reference, a.candidate, a.reference_bench, a.candidate_bench)}
    metadata = {"options": vars(a), "page_size": os.sysconf("SC_PAGE_SIZE"),
                "binary_sha256": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                  for path in sorted(binaries)}}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def run(name, kind, args, input_text=None):
        before = vmstat()
        start = time.monotonic()
        timed_out = False
        with (out / (name + ".out")).open("w") as stdout, (out / (name + ".log")).open("w") as stderr:
            try:
                result = subprocess.run(args, env=envs[kind], stdout=stdout,
                                        stderr=stderr, timeout=a.timeout,
                                        input=input_text, text=True)
                returncode = result.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                returncode = -1
        after = vmstat()
        record = {"name": name, "args": args, "returncode": returncode,
                  "timed_out": timed_out, "input_text": input_text,
                  "ds4_environment": {k: v for k, v in envs[kind].items() if k.startswith("DS4_")},
                  "wall_seconds": time.monotonic() - start,
                  "system_vmstat_delta": {k: after[k] - before[k] for k in before}}
        records.append(record)
        (out / "results.json").write_text(json.dumps(records, indent=2) + "\n")
        if returncode:
            raise RuntimeError(f"{name} failed; see {out / (name + '.log')}")
        return record

    if not a.skip_correctness:
        prompts = {
            "code": "Scrivi una funzione Python che fonde due liste ordinate senza modificarle. Spiega la complessità e mostra un test con duplicati.",
            "database": "Analyze B-trees versus LSM trees for a write-heavy database workload.",
        }
        if a.long_chars:
            prompts["long"] = Path("speed-bench/promessi_sposi.txt").read_text()[:a.long_chars] + "\nRiassumi il testo in italiano."
        for label, prompt in prompts.items():
            prompt_file = out / (label + ".txt")
            prompt_file.write_text(prompt)
            docs = []
            for kind in ("reference", "candidate"):
                name = label + "-" + kind
                output = out / (name + ".json")
                run(name, kind, [str(Path(getattr(a, kind)).resolve())] + common +
                    ["--ctx", "262144", "--nothink", "--temp", "0", "-n", str(a.tokens),
                     "--dump-logprobs", str(output), "--logprobs-top-k", "20",
                     "--prompt-file", str(prompt_file)])
                docs.append(json.loads(output.read_text()))
            if label == "long" and docs[0]["prompt_tokens"] <= 4096:
                raise AssertionError("Long test did not cross a prefill chunk; increase --long-chars")
            compare(*docs)
            print(f"{label}: exact match, {docs[0]['prompt_tokens']} prompt tokens, {len(docs[0]['steps'])} steps", flush=True)

    conversations = {}
    if a.interactive:
        conversations["two-turn"] = (
            "Explain binary search in three sentences.\n"
            "Ora mostra una funzione Python che lo implementa e spiega un caso limite.\n/quit\n", [])
    if a.vision:
        image = Path("tests/vision-fixtures/glm53/earth.jpg").resolve()
        conversations["image"] = (f"/read {image}\n/quit\n", ["--vision", a.vision])
    for label, (input_text, extra) in conversations.items():
        outputs = []
        for kind in ("reference", "candidate"):
            name = label + "-" + kind
            run(name, kind, [str(Path(getattr(a, kind)).resolve())] + common + extra +
                ["--ctx", "262144", "--nothink", "--temp", "0", "-n", str(a.tokens)], input_text)
            log = (out / (name + ".log")).read_text()
            expected_turns = 2 if label == "two-turn" else 1
            if log.count("generation:") < expected_turns:
                raise AssertionError(f"{name}: missing completed generation")
            outputs.append((out / (name + ".out")).read_bytes())
        if not outputs[0] or outputs[0] != outputs[1]:
            raise AssertionError(f"{label}: interactive output differs")
        print(f"{label}: identical nonempty stdout, completed generations checked", flush=True)

    if not a.skip_bench:
        for ctx in a.contexts:
            for rep in range(a.repeats):
                order = ("reference", "candidate") if rep % 2 == 0 else ("candidate", "reference")
                for kind in order:
                    name = f"bench-{ctx}-{rep}-{kind}"
                    record = run(name, kind, [str(Path(getattr(a, kind + "_bench")).resolve())] + common +
                        ["--ctx-alloc", "262144", "--ctx-start", str(ctx), "--ctx-max", str(ctx),
                         "--gen-tokens", str(a.tokens), "--prompt-file", "speed-bench/promessi_sposi.txt"])
                    lines = (out / (name + ".out")).read_text().splitlines()
                    rows = [line for line in lines if line.startswith("ctx_tokens,") or line.startswith(str(ctx) + ",")]
                    parsed = list(csv.DictReader(io.StringIO("\n".join(rows))))
                    if len(parsed) != 1:
                        raise AssertionError(f"Missing benchmark row in {name}")
                    record["benchmark"] = parsed[0]
                    print(name, parsed[0], flush=True)
    (out / "results.json").write_text(json.dumps(records, indent=2) + "\n")
    print("Results:", out)


if __name__ == "__main__":
    main()
