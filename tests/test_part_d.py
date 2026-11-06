"""Part D: retrieval, grounding, tools, the agent loop, limits and crash recovery.

Agent tests write to a temporary folder, so they never touch the delivered outputs.
"""
import json
import re
import shutil
import tempfile
import unittest
from types import SimpleNamespace as NS
from pathlib import Path
from unittest import mock

import pandas as pd

import mock_llm
from src import config, policy_rag, refund_agent


class Chunking(unittest.TestCase):
    def test_windows_overlap_and_stop(self):
        chunks = policy_rag.chunk_words([str(i) for i in range(100)], 40, 10)
        self.assertEqual([c[0] for c in chunks], ["0", "30", "60"])
        self.assertEqual(chunks[-1][-1], "99")

    def test_stops_when_a_window_ends_exactly_on_the_last_word(self):
        self.assertEqual(len(policy_rag.chunk_words([str(i) for i in range(70)], 40, 10)), 2)

    def test_invalid_settings(self):
        with self.assertRaises(ValueError):
            policy_rag.chunk_words(["a"], 10, 10)

    def test_reference_counts(self):
        counts = pd.Series([c["source"] for c in policy_rag.PolicyRAG().chunks]).value_counts().to_dict()
        self.assertEqual(counts, {"refund_policy.txt": 4, "delivery_policy.txt": 3, "escalation_policy.txt": 3,
                                  "payment_policy.txt": 3, "privacy_policy.txt": 2})


class Retrieval(unittest.TestCase):
    def test_shape_and_ranking(self):
        hits = policy_rag.retrieve("refund policy late delivery")
        self.assertEqual([h["source"] for h in hits], ["refund_policy.txt"] * 3)
        self.assertEqual(set(hits[0]), {"source", "score", "text"})
        self.assertEqual([h["score"] for h in hits], sorted((h["score"] for h in hits), reverse=True))

    def test_message_format(self):
        captured = {}
        original = mock_llm.chat

        def spy(messages, tools=None):
            captured["messages"] = messages
            return original(messages, tools)

        with mock.patch("mock_llm.chat", spy):
            result = policy_rag.answer("Who must approve a refund above INR 1000?")
        system, user = captured["messages"]
        self.assertIn("POLICY ASSISTANT", system["content"])
        self.assertIn("could not find", system["content"])
        self.assertTrue(user["content"].startswith("CONTEXT:\n["))
        self.assertTrue(user["content"].endswith("\nQUESTION:\nWho must approve a refund above INR 1000?"))
        self.assertEqual(user["content"].count(".txt]\n"), 3)
        self.assertEqual(set(result), {"answer", "sources", "retrieved"})


class Evaluation(unittest.TestCase):
    def test_default_overlap(self):
        e = policy_rag.evaluate(policy_rag.PolicyRAG())
        self.assertEqual((e["doc_in_top3"], e["fact_in_answer"], e["fallback"]), (12, 12, 2))

    def test_zero_overlap_breaks_at_least_one_question(self):
        e = policy_rag.evaluate(policy_rag.PolicyRAG(overlap=0))
        self.assertGreaterEqual(len(e["failed_questions"]), 1)


class Tools(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        patcher = mock.patch.object(config, "OUTPUTS", self.folder)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(shutil.rmtree, self.folder, ignore_errors=True)

    def _ledger(self):
        path = self.folder / "refund_ledger.jsonl"
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []

    def test_every_tool_is_specified_and_mapped(self):
        self.assertEqual({s["name"] for s in refund_agent.TOOLS_SPEC}, set(refund_agent.TOOL_MAP))
        self.assertEqual(len(refund_agent.TOOLS_SPEC), 5)
        for spec in refund_agent.TOOLS_SPEC:
            self.assertTrue(spec["description"].endswith("."))
            self.assertEqual(spec["parameters"]["type"], "object")

    def test_get_order(self):
        self.assertEqual(refund_agent.get_order("ORD-999999"), {"error": "order not found"})
        found = refund_agent.get_order("ORD-100048")
        self.assertEqual(set(found), {"order_id", "status", "order_amount", "delivery_minutes"})

    def test_search_policy(self):
        self.assertEqual(len(refund_agent.search_policy("refund")["results"]), 3)

    def test_refund_rules_at_the_boundary(self):
        self.assertEqual(refund_agent.calculate_refund(100.0, 45), {"refund_inr": 0.0})
        self.assertEqual(refund_agent.calculate_refund(100.0, 46), {"refund_inr": 100.0})
        self.assertEqual(refund_agent.calculate_refund(100.0, None), {"refund_inr": 0.0})

    def _order(self, scenario_id):
        """The order named in a scenario message, e.g. S1 is late and small, S3 late and large."""
        scenarios = pd.read_csv(config.RAW / "agent_scenarios.csv", keep_default_na=False).set_index("scenario_id")
        return re.search(r"ORD-\d{6}", scenarios.loc[scenario_id, "message"]).group(0)

    def test_ledger_lines(self):
        self.assertEqual(refund_agent.issue_refund(self._order("S1"), 128.25), {"refund_inr": 128.25})
        self.assertEqual(refund_agent.escalate_to_human(self._order("S3"), 1999.0), {"refund_inr": 1999.0})
        ledger = self._ledger()
        self.assertEqual([e["tool"] for e in ledger], ["issue_refund", "escalate_to_human"])
        self.assertTrue(all("timestamp" in e for e in ledger))

    def test_money_tools_enforce_the_rules_themselves(self):
        late_small, late_big = self._order("S1"), self._order("S3")
        refused = [
            refund_agent.issue_refund(late_big, 1999.0),                 # above INR 1000 must be escalated
            refund_agent.escalate_to_human(late_small, 128.25),          # within the automatic limit
            refund_agent.issue_refund(self._order("S4"), 50.0),          # completed but delivered on time
            refund_agent.issue_refund(self._order("S5"), 50.0),          # cancelled order
            refund_agent.issue_refund("ORD-999999", 50.0),               # no such order
            refund_agent.issue_refund(late_small, 1000.0),               # not the amount owed
            refund_agent.issue_refund(late_small, -128.25),              # negative
            refund_agent.issue_refund(late_small, 0),                    # zero
        ]
        self.assertTrue(all("error" in result for result in refused), refused)
        self.assertEqual(self._ledger(), [])

    def test_a_refund_is_recorded_only_once_per_order(self):
        order = self._order("S1")
        first = refund_agent.issue_refund(order, 128.25)
        second = refund_agent.issue_refund(order, 128.25)
        self.assertEqual(first, {"refund_inr": 128.25})
        self.assertEqual(second["refund_inr"], 128.25)
        self.assertIn("already_recorded_by", second)
        self.assertEqual(len(self._ledger()), 1)

    def test_bad_requests_are_rejected_without_running(self):
        for name, args in [("nope", {}), ("get_order", {}), ("get_order", {"order_id": 5}),
                           ("get_order", {"order_id": "x", "extra": 1}), ("issue_refund", {"order_id": "x", "refund_inr": "1"})]:
            observation, executed = refund_agent.execute_tool(name, args)
            self.assertIn("error", observation)
            self.assertFalse(executed)
        self.assertEqual(self._ledger(), [])


class AgentBase(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        for name, value in (("OUTPUTS", self.folder), ("STATE", self.folder / "state"), ("METRICS", self.folder / "metrics")):
            patcher = mock.patch.object(config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(shutil.rmtree, self.folder, ignore_errors=True)
        self.scenarios = pd.read_csv(config.RAW / "agent_scenarios.csv", keep_default_na=False).set_index("scenario_id")
        self.s1 = self.scenarios.loc["S1", "message"]


class ScenarioFile(AgentBase):
    def test_generator_wrote_the_expected_rows(self):
        rows = pd.read_csv(config.RAW / "agent_scenarios.csv", keep_default_na=False)
        self.assertEqual(rows["scenario_id"].tolist(), [f"S{i}" for i in range(1, 8)])
        self.assertEqual(rows["expected_outcome"].tolist(),
                         ["REFUND", "REFUND", "ESCALATE", "DENIED", "DENIED", "NOT_FOUND", "FALLBACK"])
        self.assertEqual(rows["expected_refund_inr"].tolist()[:3], [128.25, 135.0, 1999.0])


class Scenarios(AgentBase):
    def test_all_seven(self):
        rows = {r["scenario_id"]: r for r in refund_agent.run_scenarios()}
        self.assertEqual(len(rows), 7)
        self.assertTrue(all(r["matches_expected"] for r in rows.values()))
        self.assertEqual(rows["S1"]["refund_inr"], 128.25)
        self.assertEqual(rows["S1"]["llm_calls"], 5)
        self.assertEqual(rows["S2"]["refund_inr"], 135.00)
        self.assertEqual((rows["S3"]["status"], rows["S3"]["refund_inr"]), ("ESCALATE", 1999.00))
        self.assertEqual(rows["S7"]["status"], "FALLBACK_MAX_ITERATIONS")
        self.assertEqual(rows["S7"]["llm_calls"], 6)
        self.assertLessEqual(rows["S7"]["tokens_used"], 6000)

    def test_run_log_content(self):
        refund_agent.run_scenarios()
        lines = (self.folder / "agent_runs.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 7)
        for line in lines:
            entry = json.loads(line)
            for key in ("scenario_id", "status", "final_answer", "llm_calls", "tokens_used", "steps"):
                self.assertIn(key, entry)
            for step in entry["steps"]:
                self.assertEqual({"step", "tool", "arguments", "observation"}, set(step))

    def test_result_dictionary_keys(self):
        result = refund_agent.run_agent(self.s1, "S1", run_id="keys")
        self.assertLessEqual({"status", "final_answer", "steps", "llm_calls", "tokens_used"}, set(result))

    def test_message_format_of_a_tool_step(self):
        refund_agent.run_agent(self.s1, "S1", run_id="format")
        state = json.loads((config.STATE / "format.json").read_text(encoding="utf-8"))
        system, user, assistant, tool = state["messages"][:4]
        self.assertIn("AGENT", system["content"])
        self.assertEqual(user["content"], self.s1)
        self.assertIsNone(assistant["content"])
        self.assertEqual(set(assistant["tool_calls"][0]), {"id", "name", "arguments"})
        self.assertEqual(set(tool), {"role", "tool_call_id", "name", "content"})
        self.assertEqual(json.loads(tool["content"]), state["steps"][0]["observation"])

    def test_contact_details_never_reach_the_model(self):
        result_state = refund_agent.run_agent("Refund ORD-999999, call 9876543210 or a@b.co", "X", run_id="pii")
        self.assertEqual(result_state["status"], "NOT_FOUND")
        text = (config.STATE / "pii.json").read_text(encoding="utf-8")
        self.assertNotIn("9876543210", text)
        self.assertNotIn("a@b.co", text)


class StopSafely(AgentBase):
    def test_token_limit_stops_before_running_the_tool(self):
        result = refund_agent.run_agent(self.s1, "S1", token_limit=200, run_id="tokens")
        self.assertEqual(result["status"], "FALLBACK_TOKEN_LIMIT")
        self.assertGreater(result["tokens_used"], 200)
        self.assertEqual(len(result["steps"]), result["llm_calls"] - 1)
        self.assertNotIn("issue_refund", result["tools_executed"])
        self.assertIn("human agent", result["final_answer"])

    def test_timeout_stops_before_the_first_model_call(self):
        result = refund_agent.run_agent(self.s1, "S1", time_limit_seconds=0, run_id="time")
        self.assertEqual((result["status"], result["llm_calls"], result["steps"]), ("FALLBACK_TIMEOUT", 0, []))
        self.assertIn("human agent", result["final_answer"])

    def test_max_iterations(self):
        result = refund_agent.run_agent(self.scenarios.loc["S7", "message"], "S7", run_id="loop")
        self.assertEqual((result["status"], result["llm_calls"], len(result["steps"])), ("FALLBACK_MAX_ITERATIONS", 6, 6))
        self.assertIn("human agent", result["final_answer"])

    def test_invalid_tool_request_does_not_crash_the_run(self):
        def reply(content, tool_calls):
            message = NS(content=content, tool_calls=tool_calls)
            return NS(choices=[NS(message=message, finish_reason="stop")], usage=NS(total_tokens=2))

        def bad_model(messages, tools=None):
            if any(m["role"] == "tool" for m in messages):
                return reply("DENIED: cannot proceed.", [])
            return reply(None, [NS(id="c", name="drop_table", arguments={})])

        with mock.patch("mock_llm.chat", bad_model):
            result = refund_agent.run_agent("hello", "X", run_id="bad")
        self.assertEqual(result["status"], "DENIED")
        self.assertIn("error", result["steps"][0]["observation"])
        self.assertEqual(result["tools_executed"], [])


class CrashAndResume(AgentBase):
    def _ledger_tools(self):
        path = self.folder / "refund_ledger.jsonl"
        return [json.loads(l)["tool"] for l in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []

    def test_each_tool_runs_exactly_once_across_the_crash(self):
        with self.assertRaises(refund_agent.SimulatedCrash):
            refund_agent.run_agent(self.s1, "S1", run_id="crash", crash_after_tools=2)
        checkpoint = json.loads((config.STATE / "crash.json").read_text(encoding="utf-8"))
        self.assertEqual(checkpoint["tools_executed"], ["get_order", "search_policy"])
        for key in ("run_id", "scenario_id", "messages", "tokens_used", "step_number", "tools_executed"):
            self.assertIn(key, checkpoint)
        result = refund_agent.resume_agent("crash")
        self.assertEqual(result["tools_executed"], ["get_order", "search_policy", "calculate_refund", "issue_refund"])
        self.assertEqual(result["status"], "REFUND")
        self.assertEqual(refund_agent.refund_amount(result), 128.25)
        self.assertEqual(self._ledger_tools(), ["issue_refund"])

    def test_resuming_a_finished_run_changes_nothing(self):
        first = refund_agent.run_agent(self.s1, "S1", run_id="done")
        again = refund_agent.resume_agent("done")
        self.assertEqual(first["status"], again["status"])
        self.assertEqual(self._ledger_tools(), ["issue_refund"])

    def test_resume_counts_earlier_model_calls_against_the_limit(self):
        with self.assertRaises(refund_agent.SimulatedCrash):
            refund_agent.run_agent(self.s1, "S1", max_iterations=3, run_id="tight", crash_after_tools=2)
        self.assertEqual(refund_agent.resume_agent("tight")["status"], "FALLBACK_MAX_ITERATIONS")

    def test_unknown_run_id(self):
        with self.assertRaises(FileNotFoundError):
            refund_agent.resume_agent("missing")


if __name__ == "__main__":
    unittest.main()
