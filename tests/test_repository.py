"""Repository-level checks: required paths, README contract, delivered agent outputs, hygiene."""
import json
import re
import sys
import unittest

from src.config import ROOT

REQUIRED_PATHS = [
    "generate_data.py", "mock_llm.py", "requirements.txt", ".gitignore", "README.md",
    "data/clean/customers.csv", "data/clean/products.csv", "data/clean/orders.csv", "data/instamart.db", "sql/queries.sql",
    "outputs/customer_segments.csv", "outputs/lapse_model.joblib", "outputs/retention_list.csv",
    "outputs/triage_results.csv", "outputs/triage_budget_demo.csv",
    "outputs/agent_runs.jsonl", "outputs/refund_ledger.jsonl", "state",
]
REMOVED_PATHS = ["llm.py", "src/agent.py", "GITHUB_RESOURCES.md", ".env.example", "docs"]
README_SECTIONS = ["Overview", "Reproduce", "Part A", "Part B", "Part C", "Part D", "Limitations and Next Steps"]
SOURCE_FILES = list((ROOT / "src").glob("*.py")) + [ROOT / "run_project.py"]
CODE_FILES = SOURCE_FILES + list((ROOT / "tests").glob("*.py"))
ALLOWED_PACKAGES = {"pandas", "numpy", "scikit-learn", "joblib",  # chosen
                    "scipy", "threadpoolctl", "narwhals", "cloudpickle", "python-dateutil", "tzdata", "six"}  # their dependencies


def readme() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


def section(title: str, next_title: str) -> str:
    text = readme()
    return text[text.index(f"# {title}"):text.index(f"# {next_title}")]


def words_between(start_marker: str, end_marker: str, within: str) -> int:
    body = within[within.index(start_marker) + len(start_marker):within.index(end_marker)]
    return len(body.split())


class RequiredPaths(unittest.TestCase):
    def test_all_exist(self):
        self.assertEqual([p for p in REQUIRED_PATHS if not (ROOT / p).exists()], [])

    def test_removed_files_stay_removed(self):
        self.assertEqual([p for p in REMOVED_PATHS if (ROOT / p).exists()], [])

    def test_gitignore(self):
        ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").split()
        for entry in (".env", ".venv/", "__pycache__/"):
            self.assertIn(entry, ignored)


class Dependencies(unittest.TestCase):
    def test_only_allowed_packages(self):
        lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").split()
        self.assertEqual({line.split("==")[0].lower() for line in lines} - ALLOWED_PACKAGES, set())

    def test_only_allowed_imports(self):
        allowed = {"pandas", "numpy", "sklearn", "joblib", "src", "mock_llm", "tests"}
        for path in CODE_FILES:
            for module in re.findall(r"^\s*(?:from|import)\s+([a-zA-Z_]\w*)", path.read_text(encoding="utf-8"), flags=re.M):
                self.assertTrue(module in sys.stdlib_module_names or module in allowed, f"{path.name}: {module}")


class Notebook(unittest.TestCase):
    def test_outputs_are_cleared(self):
        notebook = json.loads((ROOT / "notebooks/Customer_Intelligence_Support_Copilot.ipynb").read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            self.assertFalse(cell.get("outputs"))


class Readme(unittest.TestCase):
    def test_top_level_sections_in_order(self):
        self.assertEqual(re.findall(r"^# (.+)$", readme(), flags=re.M), README_SECTIONS)

    def test_reproduce_is_numbered_from_venv_onwards(self):
        block = section("Reproduce", "Part A")
        self.assertIn("python -m venv .venv", block)
        self.assertRegex(block, r"(?m)^1\. ")

    def test_no_dollar_signs(self):
        self.assertNotIn("$", readme())
        for path in SOURCE_FILES:
            for line in path.read_text(encoding="utf-8").splitlines():
                if "print(" in line:
                    self.assertNotIn("$", line, path.name)

    def test_word_limits(self):
        self.assertLessEqual(words_between("<!-- eda-start -->", "<!-- eda-end -->", readme()), 200)
        self.assertLessEqual(words_between("<!-- react-start -->", "<!-- react-end -->", readme()), 250)

    def test_required_statements(self):
        part_a = section("Part A", "Part B")
        for phrase in ("does not establish causation", "RANK() gives customers with equal spend"):
            self.assertIn(phrase, part_a)
        part_b = section("Part B", "Part C")
        for phrase in ("always beat K=2", "fitted on the 924 training rows only"):
            self.assertIn(phrase, part_b)
        for block in (part_b, readme()[readme().index("# Limitations and Next Steps"):]):
            for phrase in ("post-hoc whole-window label", "`outputs/retention_list.csv` is illustrative", "leakage-free evaluation"):
                self.assertIn(phrase, block)
        part_d = section("Part D", "Limitations and Next Steps")
        for phrase in ("FALLBACK_TOKEN_LIMIT", "FALLBACK_TIMEOUT", "get_order, search_policy, calculate_refund, issue_refund"):
            self.assertIn(phrase, part_d)


class ReadmeMatchesComputedResults(unittest.TestCase):
    """The reported figures must be the computed ones, not hand-typed drift."""

    @classmethod
    def setUpClass(cls):
        cls.text = readme()
        cls.b = json.loads((ROOT / "outputs/metrics/part_b.json").read_text(encoding="utf-8"))
        cls.c = json.loads((ROOT / "outputs/metrics/part_c.json").read_text(encoding="utf-8"))
        cls.d = json.loads((ROOT / "outputs/metrics/part_d_agent.json").read_text(encoding="utf-8"))

    def test_model_metrics(self):
        lapse = self.b["lapse"]
        for key in ("logistic_regression", "random_forest"):
            test = lapse[key]["test"]
            self.assertIn(f"{test['precision']:.3f} | {test['recall']:.3f} | {test['f1']:.3f}", self.text, key)
        balanced = lapse["class_balance"]["balanced"]
        self.assertIn(f"{balanced['precision']:.3f} | {balanced['recall']:.3f} | {balanced['f1']:.3f}", self.text)

    def test_triage_tokens(self):
        for run in ("zero_shot", "few_shot"):
            self.assertIn(f"{self.c[run]['total_tokens']:,}", self.text, run)
            self.assertIn(f"{self.c[run]['tokens_per_classified']:.2f}", self.text, run)
        self.assertIn(f"{self.c['budget_demo']['budget_exhausted']} tickets", self.text)

    def test_scenario_table(self):
        for row in self.d["scenarios"]:
            line = f"| {row['scenario_id']} | {row['status']} | {row['refund_inr']:.2f} | {row['llm_calls']} | {row['tokens_used']:,} |"
            self.assertIn(line, self.text)


class AgentOutputs(unittest.TestCase):
    def test_run_log_has_one_line_per_scenario(self):
        lines = (ROOT / "outputs/agent_runs.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual([json.loads(l)["scenario_id"] for l in lines], [f"S{i}" for i in range(1, 8)])

    def test_ledger_entries_are_attributed(self):
        entries = [json.loads(l) for l in (ROOT / "outputs/refund_ledger.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([e["run_id"] for e in entries], ["run_S1", "run_S2", "run_S3"])
        self.assertEqual(len({e["order_id"] for e in entries}), len(entries))

    def test_crash_demo_did_not_pay_twice(self):
        demos = [json.loads(l) for l in (ROOT / "outputs/agent_demos.jsonl").read_text(encoding="utf-8").splitlines()]
        crash = next(d for d in demos if d["run_id"] == "demo_S1_crash")
        self.assertEqual(crash["tools_executed"], ["get_order", "search_policy", "calculate_refund", "issue_refund"])
        self.assertEqual(crash["steps"][-1]["observation"].get("already_recorded_by"), "run_S1")

    def test_checkpoints_exist(self):
        self.assertTrue(list((ROOT / "state").glob("*.json")))


class Safety(unittest.TestCase):
    def test_model_calls_go_through_the_supplied_file(self):
        for name in ("triage.py", "policy_rag.py", "refund_agent.py"):
            self.assertIn("import mock_llm", (ROOT / "src" / name).read_text(encoding="utf-8"), name)

    def test_retired_module_is_not_imported(self):
        for path in CODE_FILES:
            self.assertIsNone(re.search(r"^\s*(?:import|from)\s+llm\b", path.read_text(encoding="utf-8"), flags=re.M), path.name)

    def test_no_network_client_or_secrets(self):
        pattern = re.compile(r"sk-[A-Za-z0-9]{16,}|api_key\s*=|urllib|http\.client|import requests")
        for path in SOURCE_FILES:
            self.assertIsNone(pattern.search(path.read_text(encoding="utf-8")), path.name)


if __name__ == "__main__":
    unittest.main()
