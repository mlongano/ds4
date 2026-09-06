#!/usr/bin/env python3
"""Unit checks for the exact-logprob acceptance gate."""
import copy
import sys
import unittest
sys.dont_write_bytecode = True
from rocm_streaming_regression import compare


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


if __name__ == "__main__":
    unittest.main()
