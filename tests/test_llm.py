"""The supplied mock_llm.py: integrity, behaviour, and the retry helper built around it."""
import hashlib
import json
import unittest

import mock_llm
from src import triage
from src.config import ROOT

# SHA-256 of the files as printed in the brief, with line endings normalised for Windows checkouts.
SUPPLIED_FILES = {
    "mock_llm.py": "77bf9bb7cb96e67f49d808cb2e84befb0e5a346833cd508df924437b606905b6",
    "generate_data.py": "e5bd939dfa40402281d0285bdf755d3b75f42159265d78098e248e58683de6cd",
}
SYSTEM_CLASSIFY = {"role": "system", "content": "CLASSIFY"}


class SuppliedFilesAreUnmodified(unittest.TestCase):
    def test_hashes(self):
        for name, expected in SUPPLIED_FILES.items():
            content = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
            self.assertEqual(hashlib.sha256(content).hexdigest(), expected, name)

    def test_project_imports_the_root_file(self):
        self.assertEqual(ROOT / "mock_llm.py", type(ROOT)(mock_llm.__file__))


class TokenEstimator(unittest.TestCase):
    def test_formula(self):
        self.assertEqual(mock_llm.est_tokens("one two three"), 4)  # ceil(3 * 1.3)
        self.assertEqual(mock_llm.est_tokens(""), 0)
        self.assertEqual(mock_llm.est_tokens(None), 0)


class MockBehaviour(unittest.TestCase):
    def setUp(self):
        mock_llm.reset()
        self.addCleanup(mock_llm.reset)

    def test_response_shape(self):
        reply = mock_llm.chat([SYSTEM_CLASSIFY, {"role": "user", "content": "my app crashes"}])
        self.assertEqual(json.loads(reply.choices[0].message.content)["category"], "technical")
        self.assertEqual(reply.choices[0].finish_reason, "stop")
        self.assertEqual(reply.choices[0].message.tool_calls, [])
        self.assertEqual(reply.usage.total_tokens, reply.usage.prompt_tokens + reply.usage.completion_tokens)

    def test_every_ninth_call_fails_then_retry_succeeds(self):
        messages = [SYSTEM_CLASSIFY, {"role": "user", "content": "late delivery"}]
        for _ in range(8):
            mock_llm.chat(messages)
        with self.assertRaises(mock_llm.TransientLLMError):
            mock_llm.chat(messages)
        mock_llm.chat(messages)

    def test_reset_restarts_the_counters(self):
        messages = [SYSTEM_CLASSIFY, {"role": "user", "content": "late delivery"}]
        for _ in range(8):
            mock_llm.chat(messages)
        mock_llm.reset()
        for _ in range(8):
            mock_llm.chat(messages)

    def test_few_shot_marker_changes_precedence(self):
        ticket = {"role": "user", "content": "I was charged, this is fraud"}
        zero = mock_llm.chat([SYSTEM_CLASSIFY, ticket]).choices[0].message.content
        few = mock_llm.chat([{"role": "system", "content": "CLASSIFY\nEXAMPLES:"}, ticket]).choices[0].message.content
        self.assertEqual(json.loads(zero)["category"], "billing")
        self.assertEqual(json.loads(few)["category"], "fraud")

    def test_policy_answers_only_from_context(self):
        system = {"role": "system", "content": "POLICY ASSISTANT"}
        hit = {"role": "user", "content": "CONTEXT:\n[a.txt]\nRefunds take 5 days.\nQUESTION:\nHow long do refunds take?"}
        miss = {"role": "user", "content": "CONTEXT:\n[a.txt]\nRefunds take 5 days.\nQUESTION:\nWhich colour?"}
        self.assertEqual(mock_llm.chat([system, hit]).choices[0].message.content, "Refunds take 5 days.")
        self.assertTrue(mock_llm.chat([system, miss]).choices[0].message.content.startswith("I could not find"))

    def test_agent_without_order_id_asks_for_it(self):
        reply = mock_llm.chat([{"role": "system", "content": "AGENT"}, {"role": "user", "content": "refund me"}])
        self.assertTrue(reply.choices[0].message.content.startswith("NEEDS_INFO"))

    def test_unknown_marker_is_rejected(self):
        with self.assertRaises(ValueError):
            mock_llm.chat([{"role": "system", "content": "hello"}])


class Retry(unittest.TestCase):
    def test_wait_is_inside_the_jitter_range(self):
        for retry_number, (low, high) in enumerate([(0.045, 0.0575), (0.09, 0.115), (0.18, 0.23)]):
            for _ in range(200):
                self.assertTrue(low <= triage.backoff_wait(retry_number) <= high)

    def test_recovers_and_reports_each_retry(self):
        outcomes = [mock_llm.TransientLLMError("busy"), mock_llm.TransientLLMError("busy"), "ok"]

        def flaky():
            result = outcomes.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        seen = []
        result = triage.call_with_retry(flaky, on_retry=lambda n, wait, err: seen.append(n), sleep=lambda _: None)
        self.assertEqual(result, "ok")
        self.assertEqual(seen, [0, 1])

    def test_gives_up_after_three_retries(self):
        attempts = []

        def always_fails():
            attempts.append(1)
            raise mock_llm.TransientLLMError("busy")

        with self.assertRaises(mock_llm.TransientLLMError):
            triage.call_with_retry(always_fails, sleep=lambda _: None)
        self.assertEqual(len(attempts), 4)

    def test_does_not_retry_other_errors(self):
        attempts = []

        def broken():
            attempts.append(1)
            raise ValueError("bug")

        with self.assertRaises(ValueError):
            triage.call_with_retry(broken, sleep=lambda _: None)
        self.assertEqual(len(attempts), 1)


if __name__ == "__main__":
    unittest.main()
