#!/usr/bin/env python3
"""Unit checks for the exact-logprob acceptance gate."""
import copy
import io
import os
import subprocess
import sys
import tempfile
import time
import unittest
sys.dont_write_bytecode = True
from rocm_streaming_regression import capture_first_output, compare


def fixture():
    return {"prompt_tokens": 10, "steps": [{"selected": {"id": 1},
        "top_logprobs": [{"token": {"id": 1}, "logit": 3.0, "logprob": -0.5},
                         {"token": {"id": 2}, "logit": 2.0, "logprob": -1.5}]}]}


class ComparisonTests(unittest.TestCase):
    def test_identical(self):
        compare(fixture(), fixture())

    def test_empty_is_failure(self):
        with self.assertRaises(AssertionError):
            compare({"steps": []}, {"steps": []})

    def test_alternative_drift_is_failure(self):
        a = fixture()
        b = copy.deepcopy(a)
        b["steps"][0]["top_logprobs"][1]["logit"] += 1e-6
        with self.assertRaises(AssertionError):
            compare(a, b)

    def test_nonfinite_is_failure(self):
        for value in (float("inf"), float("-inf"), float("nan")):
            a = fixture()
            a["steps"][0]["top_logprobs"][0]["logit"] = value
            with self.assertRaises(AssertionError):
                compare(a, a)

    def test_truncated_is_failure(self):
        a = fixture()
        b = copy.deepcopy(a)
        b["steps"].append(copy.deepcopy(b["steps"][0]))
        with self.assertRaises(AssertionError):
            compare(a, b)


class FirstOutputTests(unittest.TestCase):
    def capture(self, script, timeout=5):
        output = io.BytesIO()
        with tempfile.TemporaryFile() as stderr:
            rc, first = capture_first_output([sys.executable, "-c", script],
                os.environ.copy(), output, stderr, timeout)
        return rc, first, output.getvalue()

    def test_first_output_is_not_process_completion(self):
        start = time.monotonic()
        rc, first, output = self.capture(
            "import time,sys; time.sleep(.05); print('first',flush=True); "
            "time.sleep(.4); print('last',flush=True)")
        self.assertEqual(rc, 0)
        self.assertEqual(output, b"first\nlast\n")
        self.assertGreater(first, .04)
        self.assertLess(first, time.monotonic() - start - .2)

    def test_empty_output_has_no_ttft(self):
        rc, first, output = self.capture("pass")
        self.assertEqual((rc, first, output), (0, None, b""))

    def test_nonzero_exit_and_large_output(self):
        rc, first, output = self.capture(
            "import sys; sys.stdout.buffer.write(b'x'*1048576); sys.exit(7)")
        self.assertEqual(rc, 7)
        self.assertIsNotNone(first)
        self.assertEqual(output, b"x" * 1048576)

    def test_timeout(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            self.capture("import time; time.sleep(10)", timeout=.1)


if __name__ == "__main__":
    unittest.main()
