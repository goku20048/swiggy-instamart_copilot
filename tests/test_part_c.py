"""Part C: guardrails, redaction, validation, retries, budget and routing."""
import json
import re
import unittest
from unittest import mock

import pandas as pd

import mock_llm
from src import triage
from src.config import OUTPUTS


class Guardrails(unittest.TestCase):
    def test_statuses(self):
        self.assertEqual(triage.input_guardrail("   \n"), "skipped_empty")
        self.assertEqual(triage.input_guardrail(""), "skipped_empty")
        self.assertEqual(triage.input_guardrail("word " * 400), "flagged_long")
        self.assertIsNone(triage.input_guardrail("word " * 300))


class Redaction(unittest.TestCase):
    def test_placeholders(self):
        text, counts = triage.redact("Mail a.b+c@x-y.co.in, call 9876543210, order ORD-123456.")
        self.assertIn("[EMAIL]", text)
        self.assertIn("[PHONE]", text)
        self.assertIn("[ORDER_ID]", text)
        self.assertEqual(counts, {"email": 1, "phone": 1, "order_id": 1})

    def test_numbers_that_are_not_phones_survive(self):
        self.assertEqual(triage.redact("1234567890 and 98765")[0], "1234567890 and 98765")

    def test_agent_mode_keeps_the_order_id(self):
        text, _ = triage.redact("Refund ORD-123456, call 9876543210", keep_order_id=True)
        self.assertIn("ORD-123456", text)
        self.assertNotIn("9876543210", text)

    def test_reference_counts(self):
        report = triage.redaction_report(triage.load_tickets())
        self.assertEqual(report, {"email": 55, "phone": 91, "order_id": 132, "total": 278, "remaining_after_redaction": 0})


class Prompts(unittest.TestCase):
    def test_required_content(self):
        for prompt in (triage.ZERO_SHOT_PROMPT, triage.FEW_SHOT_PROMPT):
            self.assertIn("CLASSIFY", prompt)
            for category in triage.ALLOWED_CATEGORIES:
                self.assertIn(category, prompt)
            self.assertIn('{"category": "...", "reason": "..."}', prompt)
        self.assertNotIn("EXAMPLES:", triage.ZERO_SHOT_PROMPT)
        self.assertIn("\nEXAMPLES:\n", triage.FEW_SHOT_PROMPT)
        self.assertIn('"category": "fraud"', triage.FEW_SHOT_PROMPT)


class OutputValidation(unittest.TestCase):
    def test_cases(self):
        cases = {
            '{"category": "fraud", "reason": "x"}': ("classified_json", "fraud"),
            '{"category": "shipping", "reason": "x"}': ("flagged_invalid_output", ""),
            "The customer appears to have a billing-related problem.": ("classified_keyword", "billing"),
            "billing or fraud, not sure": ("flagged_invalid_output", ""),
            "no idea": ("flagged_invalid_output", ""),
            "[1, 2]": ("flagged_invalid_output", ""),
            "": ("flagged_invalid_output", ""),
            '{"category": ["fraud"]}': ("flagged_invalid_output", ""),
        }
        for reply, expected in cases.items():
            self.assertEqual(triage.validate_reply(reply), expected, reply)


class Routing(unittest.TestCase):
    def test_rules(self):
        for status in ("skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"):
            self.assertEqual(triage.route_ticket(status, "", True), ("human_review", "NORMAL"))
        self.assertEqual(triage.route_ticket("classified_json", "fraud", False), ("security_team", "HIGH"))
        self.assertEqual(triage.route_ticket("classified_json", "fraud", True), ("security_team", "HIGH"))
        self.assertEqual(triage.route_ticket("classified_json", "billing", True), ("premium_desk", "HIGH"))
        self.assertEqual(triage.route_ticket("classified_keyword", "billing", False), ("support_queue", "NORMAL"))


class FullRuns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tickets = triage.load_tickets()
        cls.seen = []

        def spy(messages, tools=None):
            cls.seen.append(messages)
            return mock_llm.chat(messages, tools)

        cls.zero = triage.run_pipeline(triage.ZERO_SHOT_PROMPT, chat=spy)
        cls.few = triage.run_pipeline(triage.FEW_SHOT_PROMPT)

    def test_status_counts(self):
        for run in (self.zero, self.few):
            counts = run.results["status"].value_counts().to_dict()
            self.assertEqual(counts, {"classified_json": 135, "classified_keyword": 12, "flagged_invalid_output": 9,
                                      "skipped_empty": 4, "flagged_long": 4})

    def test_transient_errors_all_recovered(self):
        self.assertEqual((self.zero.transient_errors, self.zero.recovered_errors), (19, 19))
        self.assertEqual(len(self.zero.retry_log), 19)

    def test_retry_log_line_format_and_range(self):
        for line in self.zero.retry_log:
            match = re.fullmatch(r"ticket=(T\d{4}) retry=0 wait=([\d.]+)s error=.*", line)
            self.assertIsNotNone(match, line)
            self.assertTrue(0.045 <= float(match.group(2)) <= 0.0575)

    def test_accuracy(self):
        zero = triage.summarise_run(self.zero, self.tickets)
        few = triage.summarise_run(self.few, self.tickets)
        self.assertEqual((zero["tickets_classified"], zero["correct"], zero["accuracy_pct"]), (147, 126, 85.7))
        self.assertEqual((few["correct"], few["accuracy_pct"]), (147, 100.0))

    def test_zero_shot_errors_are_the_expected_two_kinds(self):
        table = triage.confusion_table(self.zero, self.tickets)
        self.assertEqual(table["fraud"]["billing"], 11)
        self.assertEqual(table["technical"]["billing"], 10)

    def test_blocked_tickets_never_reach_the_model(self):
        sent = [m[-1]["content"] for m in self.seen]
        self.assertEqual(len(sent), 156 + 19)  # 156 eligible tickets, 19 of them retried once
        self.assertFalse(any(not text.strip() or mock_llm.est_tokens(text) > triage.MAX_INPUT_TOKENS for text in sent))

    def test_no_pii_in_any_prompt_and_user_message_is_exactly_redacted_text(self):
        expected = {triage.redact(t)[0] for t in self.tickets["text"]}
        for messages in self.seen:
            self.assertEqual(triage.find_pii(messages[-1]["content"]), [])
            self.assertIn(messages[-1]["content"], expected)
            self.assertEqual(len(messages), 2)

    def test_budget_stops_calls(self):
        run = triage.run_pipeline(triage.ZERO_SHOT_PROMPT, token_budget=6000)
        exhausted = run.results[run.results["status"].eq("budget_exhausted")]
        self.assertGreaterEqual(len(exhausted), 1)
        self.assertTrue((exhausted["route"] == "human_review").all())
        self.assertTrue((exhausted["tokens_used"] == 0).all())
        before_last_call = run.total_tokens - run.results[run.results["tokens_used"] > 0]["tokens_used"].iloc[-1]
        self.assertLess(before_last_call, 6000)

    def test_exhausted_retries_go_to_a_human_and_do_not_crash(self):
        def always_busy(messages, tools=None):
            raise mock_llm.TransientLLMError("busy")

        with mock.patch.dict(triage.call_with_retry.__kwdefaults__, {"sleep": lambda _: None}):
            run = triage.run_pipeline(triage.ZERO_SHOT_PROMPT, chat=always_busy)
        statuses = run.results["status"].value_counts().to_dict()
        self.assertEqual(statuses["flagged_llm_error"], 156)
        self.assertEqual(run.results.loc[run.results["status"].eq("flagged_llm_error"), "route"].unique().tolist(), ["human_review"])


class OutputFiles(unittest.TestCase):
    def test_final_results(self):
        results = pd.read_csv(OUTPUTS / "triage_results.csv", keep_default_na=False)
        self.assertEqual(len(results), 164)
        self.assertEqual(results.columns.tolist(), ["ticket_id", "customer_id", "status", "category", "segment_name",
                                                    "priority", "route", "tokens_used"])
        segments = pd.read_csv(OUTPUTS / "customer_segments.csv").set_index("customer_id")["is_premium"].to_dict()
        for row in results.itertuples(index=False):
            route, priority = triage.route_ticket(row.status, row.category, bool(segments.get(row.customer_id, False)))
            self.assertEqual((row.route, row.priority), (route, priority), row.ticket_id)

    def test_order_follows_the_file(self):
        results = pd.read_csv(OUTPUTS / "triage_results.csv", keep_default_na=False)
        self.assertEqual(results["ticket_id"].tolist(), triage.load_tickets()["ticket_id"].tolist())

    def test_budget_demo_shows_exhaustion(self):
        demo = pd.read_csv(OUTPUTS / "triage_budget_demo.csv", keep_default_na=False)
        self.assertGreaterEqual(int(demo["status"].eq("budget_exhausted").sum()), 1)

    def test_outputs_contain_no_pii(self):
        for name in ("triage_results.csv", "triage_budget_demo.csv", "triage_retry_log.txt"):
            self.assertEqual(triage.find_pii((OUTPUTS / name).read_text(encoding="utf-8")), [], name)

    def test_metrics_saved(self):
        self.assertIn("zero_shot", json.loads((OUTPUTS / "metrics" / "part_c.json").read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
