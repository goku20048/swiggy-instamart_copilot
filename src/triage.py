"""Part C: privacy-first support-ticket triage.

Pipeline per ticket, in file order: input guardrails -> PII redaction -> budget check ->
model call with retries -> output validation -> routing using Part B segments.

Run with ``python -m src.triage``. Writes ``outputs/triage_results.csv``,
``outputs/triage_budget_demo.csv``, ``outputs/triage_retry_log.txt`` and
``outputs/metrics/part_c.json``.
"""
from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Callable, TypeVar

import pandas as pd

import mock_llm
from src.config import OUTPUTS, RANDOM_SEED, RAW, save_metrics

logger = logging.getLogger("copilot.triage")
T = TypeVar("T")

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3
ALLOWED_CATEGORIES = ("billing", "delivery", "technical", "fraud")
HUMAN_REVIEW_STATUSES = {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted", "flagged_llm_error"}

# Order matters: e-mail first, then phone, then order id.
PII_PATTERNS = (
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]"),
)

ZERO_SHOT_PROMPT = (
    "CLASSIFY the support ticket into one category: billing, delivery, technical, fraud. "
    'Reply with JSON only: {"category": "...", "reason": "..."}'
)
FEW_SHOT_PROMPT = ZERO_SHOT_PROMPT + """
EXAMPLES:
Ticket: Someone used my card without permission, this is fraud. -> {"category": "fraud", "reason": "unauthorized use"}
Ticket: I was charged twice, please fix the bill. -> {"category": "billing", "reason": "duplicate charge"}
Ticket: The rider never arrived with my groceries. -> {"category": "delivery", "reason": "order not delivered"}
Ticket: The app crashes when I open the cart. -> {"category": "technical", "reason": "app crash"}"""


# --------------------------------------------------------------------------- resilience
def backoff_wait(retry_number: int) -> float:
    """Wait before retry ``n`` (0-based): BASE_DELAY x 2**n x uniform(0.9, 1.15)."""
    return BASE_DELAY * 2 ** retry_number * random.uniform(0.9, 1.15)


def call_with_retry(func: Callable[[], T], on_retry: Callable[[int, float, Exception], None] | None = None, *,
                    sleep: Callable[[float], None] = time.sleep) -> T:
    """Run ``func``, retrying only ``TransientLLMError``, with jittered exponential backoff.

    ``on_retry(retry_number, wait_seconds, error)`` fires once before each wait. When the
    retries are used up, the last error propagates so the caller can route the work to a human.
    """
    for retry_number in range(MAX_RETRIES + 1):
        try:
            return func()
        except mock_llm.TransientLLMError as error:
            if retry_number == MAX_RETRIES:
                raise
            wait = backoff_wait(retry_number)
            if on_retry is not None:
                on_retry(retry_number, wait, error)
            sleep(wait)
    raise AssertionError("unreachable")  # pragma: no cover


# --------------------------------------------------------------------------- guardrails
def input_guardrail(text: str) -> str | None:
    """Return the blocking status for a ticket, or ``None`` when it may reach the model."""
    if not text.strip():
        return "skipped_empty"
    if mock_llm.est_tokens(text) > MAX_INPUT_TOKENS:
        return "flagged_long"
    return None


def redact(text: str, keep_order_id: bool = False) -> tuple[str, dict[str, int]]:
    """Replace personal details with placeholders; return the new text and substitution counts."""
    counts: dict[str, int] = {}
    for name, pattern, placeholder in PII_PATTERNS:
        if keep_order_id and name == "order_id":
            continue
        text, counts[name] = pattern.subn(placeholder, text)
    return text, counts


def find_pii(text: str) -> list[str]:
    """Names of the PII patterns that still match ``text``."""
    return [name for name, pattern, _ in PII_PATTERNS if pattern.search(text)]


# --------------------------------------------------------------------------- output validation
def validate_reply(content: str | None) -> tuple[str, str]:
    """Return ``(status, category)``. Never guesses: anything doubtful goes to a human."""
    text = content or ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        found = {w for w in re.findall(r"[a-z]+", text.lower()) if w in ALLOWED_CATEGORIES}
        if len(found) == 1:
            return "classified_keyword", found.pop()
        return "flagged_invalid_output", ""
    category = parsed.get("category") if isinstance(parsed, dict) else None
    if isinstance(category, str) and category in ALLOWED_CATEGORIES:
        return "classified_json", category
    return "flagged_invalid_output", ""


# --------------------------------------------------------------------------- routing
def route_ticket(status: str, category: str, is_premium: bool) -> tuple[str, str]:
    """Return ``(route, priority)``. Fraud outranks segment; guardrail outcomes go to humans."""
    if status in HUMAN_REVIEW_STATUSES:
        return "human_review", "NORMAL"
    if category == "fraud":
        return "security_team", "HIGH"
    if is_premium:
        return "premium_desk", "HIGH"
    return "support_queue", "NORMAL"


# --------------------------------------------------------------------------- pipeline
@dataclass
class TriageRun:
    """Everything one full pass over the tickets produced."""

    results: pd.DataFrame
    total_tokens: int = 0
    transient_errors: int = 0
    recovered_errors: int = 0
    model_calls_attempted: int = 0
    retry_log: list[str] = field(default_factory=list)
    prompts_sent: list[str] = field(default_factory=list)


def load_tickets() -> pd.DataFrame:
    return pd.read_csv(RAW / "tickets.csv", keep_default_na=False)


def load_segments() -> dict[str, dict]:
    """Segment lookup by customer. A customer with no row is treated as not premium."""
    path = OUTPUTS / "customer_segments.csv"
    if not path.exists():
        return {}
    segments = pd.read_csv(path)
    return segments.set_index("customer_id")[["segment_name", "is_premium"]].to_dict("index")


def run_pipeline(prompt: str, token_budget: float | None = None, chat: Callable = mock_llm.chat) -> TriageRun:
    """Process all tickets once, in file order, with the given system prompt and budget."""
    tickets, segments = load_tickets(), load_segments()
    mock_llm.reset()
    random.seed(RANDOM_SEED)
    run = TriageRun(results=pd.DataFrame())
    rows = []

    for ticket in tickets.itertuples(index=False):
        status, category, tokens_used = input_guardrail(ticket.text), "", 0
        if status is None:
            safe_text, _ = redact(ticket.text)
            leaked = find_pii(safe_text)
            assert not leaked, f"{ticket.ticket_id}: PII reached the prompt builder: {leaked}"
            if token_budget is not None and run.total_tokens >= token_budget:
                status = "budget_exhausted"
            else:
                messages = [{"role": "system", "content": prompt}, {"role": "user", "content": safe_text}]
                status, category, tokens_used = _classify_ticket(ticket.ticket_id, messages, run, chat)
        segment = segments.get(ticket.customer_id, {})
        route, priority = route_ticket(status, category, bool(segment.get("is_premium", False)))
        rows.append({"ticket_id": ticket.ticket_id, "customer_id": ticket.customer_id, "status": status,
                     "category": category, "segment_name": segment.get("segment_name", ""),
                     "priority": priority, "route": route, "tokens_used": tokens_used})
    run.results = pd.DataFrame(rows)
    return run


def _classify_ticket(ticket_id: str, messages: list[dict], run: TriageRun, chat: Callable) -> tuple[str, str, int]:
    """Call the model with retries and validate the reply."""
    def attempt():
        run.model_calls_attempted += 1
        run.prompts_sent.append(messages[-1]["content"])
        return chat(messages)

    def log_retry(retry_number: int, wait: float, error: Exception) -> None:
        run.transient_errors += 1
        line = f"ticket={ticket_id} retry={retry_number} wait={wait:.4f}s error={error}"
        run.retry_log.append(line)
        logger.info("retry %s", line)

    errors_before = run.transient_errors
    try:
        response = call_with_retry(attempt, on_retry=log_retry)
    except mock_llm.TransientLLMError:
        run.transient_errors += 1
        logger.warning("ticket=%s retries exhausted; routing to human review", ticket_id)
        return "flagged_llm_error", "", 0
    run.recovered_errors += run.transient_errors - errors_before
    run.total_tokens += response.usage.total_tokens
    status, category = validate_reply(response.choices[0].message.content)
    return status, category, response.usage.total_tokens


# --------------------------------------------------------------------------- reporting
def summarise_run(run: TriageRun, tickets: pd.DataFrame) -> dict:
    """Accuracy over classified tickets, and tokens per classified ticket."""
    joined = run.results.merge(tickets[["ticket_id", "true_category"]], on="ticket_id")
    classified = joined[joined["status"].isin(["classified_json", "classified_keyword"])]
    correct = int((classified["category"] == classified["true_category"]).sum())
    return {
        "status_counts": run.results["status"].value_counts().to_dict(),
        "tickets_classified": len(classified),
        "correct": correct,
        "accuracy_pct": round(correct / len(classified) * 100, 1) if len(classified) else 0.0,
        "total_tokens": run.total_tokens,
        "tokens_per_classified": round(run.total_tokens / len(classified), 2) if len(classified) else 0.0,
        "transient_errors": run.transient_errors,
        "recovered_errors": run.recovered_errors,
        "route_counts": run.results["route"].value_counts().to_dict(),
    }


def confusion_table(run: TriageRun, tickets: pd.DataFrame) -> dict:
    joined = run.results.merge(tickets[["ticket_id", "true_category"]], on="ticket_id")
    classified = joined[joined["status"].isin(["classified_json", "classified_keyword"])]
    table = pd.crosstab(classified["true_category"], classified["category"])
    return table.reindex(index=ALLOWED_CATEGORIES, columns=ALLOWED_CATEGORIES, fill_value=0).to_dict("index")


def redaction_report(tickets: pd.DataFrame) -> dict:
    """Substitutions per type across all tickets, and proof that nothing is left afterwards."""
    totals = {name: 0 for name, _, _ in PII_PATTERNS}
    remaining = 0
    for text in tickets["text"]:
        safe, counts = redact(text)
        for name, count in counts.items():
            totals[name] += count
        remaining += sum(len(pattern.findall(safe)) for _, pattern, _ in PII_PATTERNS)
    return {**totals, "total": sum(totals.values()), "remaining_after_redaction": remaining}


def run_all() -> dict:
    """Run the comparison, the budget demonstration and the final routed output."""
    tickets = load_tickets()
    guard = tickets["text"].map(input_guardrail)
    long_tokens = sorted({mock_llm.est_tokens(t) for t in tickets.loc[guard.eq("flagged_long"), "text"]})

    zero_run = run_pipeline(ZERO_SHOT_PROMPT)
    few_run = run_pipeline(FEW_SHOT_PROMPT)

    demo_budget = 6000
    demo_run = run_pipeline(ZERO_SHOT_PROMPT, token_budget=demo_budget)
    final_run = run_pipeline(FEW_SHOT_PROMPT, token_budget=100000)

    OUTPUTS.mkdir(exist_ok=True)
    demo_run.results.to_csv(OUTPUTS / "triage_budget_demo.csv", index=False)
    final_run.results.to_csv(OUTPUTS / "triage_results.csv", index=False)
    (OUTPUTS / "triage_retry_log.txt").write_text("\n".join(final_run.retry_log) + "\n", encoding="utf-8")

    for label, run in (("zero", zero_run), ("few", few_run), ("final", final_run)):
        for text in run.prompts_sent:
            assert not find_pii(text), f"{label}: PII found in a prompt"

    waits = [float(re.search(r"wait=([\d.]+)s", line).group(1)) for line in final_run.retry_log]
    payload = {
        "guardrails": {"skipped_empty": int(guard.eq("skipped_empty").sum()), "flagged_long": int(guard.eq("flagged_long").sum()),
                       "long_ticket_token_estimates": long_tokens, "eligible": int(guard.isna().sum())},
        "redaction": redaction_report(tickets),
        "prompt_tokens": {"zero_shot": mock_llm.est_tokens(ZERO_SHOT_PROMPT), "few_shot": mock_llm.est_tokens(FEW_SHOT_PROMPT)},
        "zero_shot": summarise_run(zero_run, tickets),
        "few_shot": summarise_run(few_run, tickets),
        "zero_shot_confusion_true_by_predicted": confusion_table(zero_run, tickets),
        "few_shot_confusion_true_by_predicted": confusion_table(few_run, tickets),
        "budget_demo": {"token_budget": demo_budget, "budget_exhausted": int(demo_run.results["status"].eq("budget_exhausted").sum()),
                        "tokens_spent": demo_run.total_tokens, "status_counts": demo_run.results["status"].value_counts().to_dict()},
        "final": summarise_run(final_run, tickets),
        "retry_waits": {"count": len(waits), "min": min(waits), "max": max(waits), "all_retry_number_zero": all("retry=0" in l for l in final_run.retry_log)},
    }
    save_metrics("part_c", payload)
    logger.info("Part C finished: %s", payload["final"]["status_counts"])
    return payload


if __name__ == "__main__":
    from src.config import setup_logging
    setup_logging()
    run_all()
