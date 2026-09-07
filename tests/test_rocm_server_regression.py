import copy
import sys
import unittest
sys.dont_write_bytecode = True
from rocm_server_regression import generation_metrics, summarize_events


def fixture():
    return [
        {"seconds": 1, "data": {"choices": [{"delta": {"role": "assistant"}}]}},
        {"seconds": 2, "data": {"choices": [{"delta": {"reasoning_content": "reason"}}]}},
        {"seconds": 5, "data": {"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}]}},
        {"seconds": 5.1, "data": {"usage": {"completion_tokens": 100}}},
    ]


class ServerReplayTests(unittest.TestCase):
    def test_reasoning_and_answer_timings_are_separate(self):
        output, metrics = summarize_events(fixture(), 64)
        self.assertEqual(output["reasoning_content"], "reason")
        self.assertEqual(output["content"], "answer")
        self.assertEqual(metrics["first_output_seconds"], 2)
        self.assertEqual(metrics["first_text_seconds"], 5)

    def test_fragment_boundaries_do_not_change_output(self):
        fragmented = fixture()
        fragmented[2]["data"]["choices"][0]["delta"]["content"] = "ans"
        fragmented.insert(3, {"seconds": 5.01, "data": {"choices": [{"delta": {"content": "wer"}}]}})
        self.assertEqual(summarize_events(fixture())[0], summarize_events(fragmented)[0])

    def test_reject_incomplete_and_insufficient_generation(self):
        for events in ([], fixture()[:-1], fixture()[1:2], fixture()[:1] + fixture()[-1:]):
            with self.assertRaises(AssertionError):
                summarize_events(events)
        with self.assertRaises(AssertionError):
            summarize_events(fixture(), 101)

    def test_reject_error_and_nonfinite_time(self):
        events = fixture()
        events[2]["data"]["choices"][0]["finish_reason"] = "error"
        with self.assertRaises(AssertionError):
            summarize_events(events)
        for value in (float("inf"), float("nan"), -1):
            events = fixture()
            events[0]["seconds"] = value
            with self.assertRaises(AssertionError):
                summarize_events(events)

    def test_tool_ids_are_ignored_but_arguments_are_not(self):
        a = fixture()
        a[2]["data"]["choices"][0]["delta"] = {"tool_calls": [
            {"index": 0, "id": "random", "function": {"name": "bash", "arguments": "{}"}}]}
        b = copy.deepcopy(a)
        b[2]["data"]["choices"][0]["delta"]["tool_calls"][0]["id"] = "other"
        self.assertEqual(summarize_events(a)[0], summarize_events(b)[0])
        b[2]["data"]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] = "[]"
        self.assertNotEqual(summarize_events(a)[0], summarize_events(b)[0])

    def test_generation_metrics_exclude_first_interval(self):
        log = ("gen=50 TOOLS THINKING decoding chunk=5.00 t/s avg=5.00 t/s 10.000s\n"
               "gen=150 TOOLS decoding chunk=10.00 t/s avg=7.50 t/s 20.000s\n")
        m = generation_metrics(log, 150)
        self.assertEqual(m["generation_tps"], 7.5)
        self.assertEqual(m["post_first_interval_tps"], 10)
        with self.assertRaises(AssertionError):
            generation_metrics(log, 151)
        with self.assertRaises(AssertionError):
            generation_metrics("", 150)


if __name__ == "__main__":
    unittest.main()
