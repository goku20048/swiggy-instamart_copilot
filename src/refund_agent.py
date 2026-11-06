"""Part D: the refund tools and the bounded, resumable ReAct agent that calls them.

The refund rules live here, in deterministic Python, not in the model. ``TOOLS_SPEC`` describes
each tool; ``TOOL_MAP`` maps its name to the function. ``run_agent`` loops: ask the model, run the
tool it requests, append the observation, checkpoint. Three limits (model calls, tokens, wall-clock
time) end a run with a human-handoff summary rather than an error. ``resume_agent`` continues from
the last checkpoint without repeating any tool. The ledger is a local JSONL audit trail and no real
money moves.

Run with ``python -m src.refund_agent``. Writes ``outputs/agent_runs.jsonl`` (one line per scenario), ``outputs/agent_demos.jsonl`` (stop-safely and
crash-and-resume runs), ``outputs/refund_ledger.jsonl``, checkpoints in ``state/`` and
``outputs/metrics/part_d_agent.json``.
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

import mock_llm
from src import config
from src.policy_rag import retrieve
from src.triage import call_with_retry, redact

logger = logging.getLogger("copilot.agent")

LATE_AFTER_MINUTES = 45
AUTO_APPROVE_LIMIT_INR = 1000
SYSTEM_PROMPT = ("AGENT: you decide refund requests by calling tools. Verify the order, check the policy, compute "
                 "the refund, then issue it or escalate it. Finish with STATUS: explanation.")
REFUND_TOOLS = {"issue_refund", "escalate_to_human"}
# Identifies the agent run that wrote a ledger line, so a demonstration is never mistaken for a payout.
RUN_CONTEXT: contextvars.ContextVar[dict] = contextvars.ContextVar("run_context", default={})


# --------------------------------------------------------------------------- tools
def get_order(order_id: str) -> dict:
    """Read one order from the SQLite database."""
    if not config.DB_PATH.exists():
        raise FileNotFoundError(f"{config.DB_PATH} is missing; run the Part A pipeline first")
    with sqlite3.connect(config.DB_PATH) as con:
        row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
                          (order_id,)).fetchone()
    if row is None:
        return {"error": "order not found"}
    return dict(zip(("order_id", "status", "order_amount", "delivery_minutes"), row))


def search_policy(query: str) -> dict:
    """Search the policy documents with the retrieval built for the policy assistant."""
    return {"results": retrieve(query)}


def calculate_refund(order_amount: float, delivery_minutes: float | None) -> dict:
    """Late delivery (more than 45 minutes) refunds the full amount; otherwise nothing."""
    if order_amount < 0:
        raise ValueError("order_amount must not be negative")
    late = delivery_minutes is not None and delivery_minutes > LATE_AFTER_MINUTES
    return {"refund_inr": round(float(order_amount), 2) if late else 0.0}


def _append_ledger(tool: str, order_id: str, refund_inr: float) -> dict:
    config.OUTPUTS.mkdir(exist_ok=True)
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "tool": tool, "order_id": order_id,
             "refund_inr": refund_inr, **RUN_CONTEXT.get()}
    with (config.OUTPUTS / "refund_ledger.jsonl").open("a", encoding="utf-8") as ledger:
        ledger.write(json.dumps(entry) + "\n")
    return {"refund_inr": refund_inr}


def _recorded_refund(order_id: str) -> dict | None:
    """The ledger entry already written for this order, if any."""
    path = config.OUTPUTS / "refund_ledger.jsonl"
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry.get("order_id") == order_id:
            return entry
    return None


def _refund_problem(order_id: str, refund_inr: float, escalating: bool) -> str | None:
    """Check a refund against the database and the refund rules, independently of the model.

    Returns a reason to refuse, or ``None`` when the refund is allowed for this tool.
    """
    if isinstance(refund_inr, bool) or not isinstance(refund_inr, (int, float)) or refund_inr <= 0:
        return "refund amount must be a positive number"
    order = get_order(order_id)
    if "error" in order:
        return "order not found"
    if order["status"] != "completed":
        return f"order status is {order['status']}; only completed orders can be refunded"
    owed = calculate_refund(order["order_amount"], order["delivery_minutes"])["refund_inr"]
    if owed == 0:
        return f"delivered within {LATE_AFTER_MINUTES} minutes; no refund is owed"
    if abs(refund_inr - owed) > 0.005:
        return f"refund must equal the order amount of INR {owed:.2f}"
    if not escalating and owed > AUTO_APPROVE_LIMIT_INR:
        return f"refund above INR {AUTO_APPROVE_LIMIT_INR} needs human approval"
    if escalating and owed <= AUTO_APPROVE_LIMIT_INR:
        return f"refunds up to INR {AUTO_APPROVE_LIMIT_INR} are issued automatically, not escalated"
    return None


def _record_refund(tool: str, order_id: str, refund_inr: float) -> dict:
    """Validate, then write one ledger line. A second request for the same order writes nothing."""
    problem = _refund_problem(order_id, refund_inr, escalating=tool == "escalate_to_human")
    if problem:
        return {"error": problem}
    previous = _recorded_refund(order_id)
    if previous is not None:
        return {"refund_inr": previous["refund_inr"], "already_recorded_by": previous.get("run_id", previous["tool"])}
    return _append_ledger(tool, order_id, refund_inr)


def issue_refund(order_id: str, refund_inr: float) -> dict:
    """Record an automatic refund of at most INR 1000 for a late, completed order."""
    return _record_refund("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id: str, refund_inr: float) -> dict:
    """Record that a human support lead must approve a refund above INR 1000."""
    return _record_refund("escalate_to_human", order_id, refund_inr)


TOOL_MAP: dict[str, Callable[..., dict]] = {
    "get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
    "issue_refund": issue_refund, "escalate_to_human": escalate_to_human,
}


def _spec(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required, "additionalProperties": False}}


_ORDER_ID = {"type": "string", "description": "Order identifier, for example ORD-100123."}
_REFUND = {"type": "number", "description": "Refund amount in INR."}
TOOLS_SPEC: list[dict] = [
    _spec("get_order", "Look up an order's status, amount and delivery time in the database.",
          {"order_id": _ORDER_ID}, ["order_id"]),
    _spec("search_policy", "Search the policy documents and return the most relevant passages.",
          {"query": {"type": "string", "description": "Search text."}}, ["query"]),
    _spec("calculate_refund", "Compute the refund owed for an order from its amount and delivery time.",
          {"order_amount": {"type": "number", "description": "Order amount in INR."},
           "delivery_minutes": {"type": ["number", "null"], "description": "Delivery time in minutes."}},
          ["order_amount", "delivery_minutes"]),
    _spec("issue_refund", "Issue an automatic refund of at most INR 1000 and record it in the ledger.",
          {"order_id": _ORDER_ID, "refund_inr": _REFUND}, ["order_id", "refund_inr"]),
    _spec("escalate_to_human", "Send a refund above INR 1000 to a human support lead and record it in the ledger.",
          {"order_id": _ORDER_ID, "refund_inr": _REFUND}, ["order_id", "refund_inr"]),
]
_SPEC_BY_NAME = {spec["name"]: spec["parameters"] for spec in TOOLS_SPEC}
_JSON_TYPES: dict[str, tuple[type, ...]] = {"string": (str,), "number": (int, float), "null": (type(None),)}


def validate_arguments(name: str, arguments: Any) -> str | None:
    """Check a model's tool request against ``TOOLS_SPEC``. Returns an error message or ``None``."""
    if name not in _SPEC_BY_NAME:
        return f"unknown tool: {name}"
    if not isinstance(arguments, dict):
        return f"{name}: arguments must be an object"
    spec = _SPEC_BY_NAME[name]
    missing = [key for key in spec["required"] if key not in arguments]
    if missing:
        return f"{name}: missing arguments {missing}"
    unexpected = [key for key in arguments if key not in spec["properties"]]
    if unexpected:
        return f"{name}: unexpected arguments {unexpected}"
    for key, value in arguments.items():
        declared = spec["properties"][key]["type"]
        allowed = tuple(t for d in ([declared] if isinstance(declared, str) else declared) for t in _JSON_TYPES[d])
        if isinstance(value, bool) or not isinstance(value, allowed):
            return f"{name}: argument {key} has the wrong type"
    return None


def execute_tool(name: str, arguments: Any) -> tuple[dict, bool]:
    """Validate and run one tool request. Returns ``(observation, was_executed)``.

    Bad requests and tool failures become an error observation, so the model can react
    instead of the whole run crashing.
    """
    problem = validate_arguments(name, arguments)
    if problem:
        logger.warning("tool request rejected: %s", problem)
        return {"error": problem}, False
    try:
        return TOOL_MAP[name](**arguments), True
    except (ValueError, sqlite3.Error) as error:
        logger.exception("tool %s failed", name)
        return {"error": f"{name} failed: {error}"}, True


# --------------------------------------------------------------------------- agent loop
class SimulatedCrash(RuntimeError):
    """Raised on request right after a checkpoint, to demonstrate recovery."""


def _checkpoint_path(run_id: str):
    return config.STATE / f"{run_id}.json"


def _save_checkpoint(state: dict) -> None:
    """Write atomically, so a crash during the write cannot corrupt the previous checkpoint."""
    config.STATE.mkdir(exist_ok=True)
    path = _checkpoint_path(state["run_id"])
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=1, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _handoff_summary(reason: str, state: dict) -> str:
    tools = ", ".join(state["tools_executed"]) or "none"
    return (f"Handing over to a human agent: {reason}. Request: {state['messages'][1]['content']} "
            f"Tools completed: {tools}.")


def _result(state: dict, status: str, final_answer: str) -> dict:
    """Finish a run: record the outcome in the checkpoint and return the result dictionary."""
    state["status"], state["final_answer"] = status, final_answer
    _save_checkpoint(state)
    return {"run_id": state["run_id"], "scenario_id": state["scenario_id"], "status": status,
            "final_answer": final_answer, "steps": state["steps"], "llm_calls": state["llm_calls"],
            "tokens_used": state["tokens_used"], "tools_executed": state["tools_executed"]}


def _loop(state: dict, crash_after_tools: int | None = None) -> dict:
    """The ReAct loop, shared by a fresh run and a resumed run."""
    started = time.monotonic() - state["elapsed_seconds"]
    context_token = RUN_CONTEXT.set({"run_id": state["run_id"], "scenario_id": state["scenario_id"]})
    try:
        while state["llm_calls"] < state["max_iterations"]:
            elapsed = time.monotonic() - started
            if elapsed >= state["time_limit_seconds"]:
                return _result(state, "FALLBACK_TIMEOUT", _handoff_summary(
                    f"the {state['time_limit_seconds']}-second time limit was reached", state))

            response = call_with_retry(lambda: mock_llm.chat(state["messages"], TOOLS_SPEC))
            state["llm_calls"] += 1
            state["tokens_used"] += response.usage.total_tokens
            if state["tokens_used"] > state["token_limit"]:
                return _result(state, "FALLBACK_TOKEN_LIMIT", _handoff_summary(
                    f"the {state['token_limit']}-token limit was exceeded", state))

            message = response.choices[0].message
            if not message.tool_calls:
                content = message.content or ""
                return _result(state, content.split(":", 1)[0].strip(), content)

            call = message.tool_calls[0]
            state["messages"].append({"role": "assistant", "content": None, "tool_calls": [
                {"id": call.id, "name": call.name, "arguments": call.arguments}]})
            observation, executed = execute_tool(call.name, call.arguments)
            if executed:
                state["tools_executed"].append(call.name)
            state["steps"].append({"step": len(state["steps"]) + 1, "tool": call.name,
                                   "arguments": call.arguments, "observation": observation})
            state["messages"].append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                                      "content": json.dumps(observation)})
            state["step_number"] = len(state["steps"])
            state["elapsed_seconds"] = time.monotonic() - started
            _save_checkpoint(state)
            if crash_after_tools is not None and len(state["tools_executed"]) == crash_after_tools:
                raise SimulatedCrash(f"simulated crash after tool {crash_after_tools}")

        return _result(state, "FALLBACK_MAX_ITERATIONS", _handoff_summary(
            f"no final answer after {state['max_iterations']} model calls", state))
    finally:
        RUN_CONTEXT.reset(context_token)


def run_agent(user_message: str, scenario_id: str, max_iterations: int = 6, token_limit: int = 6000,
              time_limit_seconds: float = 10, *, run_id: str | None = None,
              crash_after_tools: int | None = None) -> dict:
    """Decide one refund request. Returns status, final_answer, steps, llm_calls and tokens_used."""
    safe_message, _ = redact(user_message, keep_order_id=True)  # the order id is needed; contact details are not
    state: dict[str, Any] = {
        "run_id": run_id or f"{scenario_id}_{uuid.uuid4().hex[:8]}", "scenario_id": scenario_id,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": safe_message}],
        "tokens_used": 0, "llm_calls": 0, "step_number": 0, "tools_executed": [], "steps": [],
        "max_iterations": max_iterations, "token_limit": token_limit, "time_limit_seconds": time_limit_seconds,
        "elapsed_seconds": 0.0, "status": None, "final_answer": None,
    }
    return _loop(state, crash_after_tools)


def resume_agent(run_id: str) -> dict:
    """Continue a run from its last checkpoint. A finished run returns its result unchanged."""
    path = _checkpoint_path(run_id)
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint for run {run_id!r} in {config.STATE}")
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("status"):
        return _result(state, state["status"], state["final_answer"])
    logger.info("resuming %s after %d tools: %s", run_id, len(state["tools_executed"]), state["tools_executed"])
    return _loop(state)


# --------------------------------------------------------------------------- scenarios and demonstrations
def refund_amount(result: dict) -> float:
    """The amount issued or escalated in a run, read from the tool observations."""
    for step in reversed(result["steps"]):
        if step["tool"] in REFUND_TOOLS and "refund_inr" in step["observation"]:
            return float(step["observation"]["refund_inr"])
    return 0.0


def outcome_matches(result: dict, expected_outcome: str, expected_refund: float) -> bool:
    """Compare a run with its scenario row. ``FALLBACK`` means the max-iterations stop."""
    expected = "FALLBACK_MAX_ITERATIONS" if expected_outcome == "FALLBACK" else expected_outcome
    return result["status"] == expected and abs(refund_amount(result) - expected_refund) <= 0.01


def _log_line(result: dict) -> str:
    return json.dumps({key: result[key] for key in
                       ("run_id", "scenario_id", "status", "final_answer", "llm_calls", "tokens_used", "tools_executed", "steps")})


def run_scenarios() -> list[dict]:
    """Run every row of ``agent_scenarios.csv``, appending one line per run to the run log."""
    scenarios = pd.read_csv(config.RAW / "agent_scenarios.csv", keep_default_na=False)
    rows = []
    with (config.OUTPUTS / "agent_runs.jsonl").open("a", encoding="utf-8") as log:
        for row in scenarios.itertuples(index=False):
            mock_llm.reset()
            result = run_agent(row.message, row.scenario_id, run_id=f"run_{row.scenario_id}")
            log.write(_log_line(result) + "\n")
            rows.append({"scenario_id": row.scenario_id, "status": result["status"],
                         "refund_inr": refund_amount(result), "llm_calls": result["llm_calls"],
                         "tokens_used": result["tokens_used"], "expected_outcome": row.expected_outcome,
                         "expected_refund_inr": float(row.expected_refund_inr),
                         "matches_expected": outcome_matches(result, row.expected_outcome, float(row.expected_refund_inr))})
    return rows


def run_demonstrations() -> dict:
    """The stop-safely demonstrations and the S1 crash-and-resume demonstration."""
    scenarios = pd.read_csv(config.RAW / "agent_scenarios.csv", keep_default_na=False).set_index("scenario_id")
    s1_message = scenarios.loc["S1", "message"]

    token_demo = run_agent(s1_message, "S1", token_limit=200, run_id="demo_S1_token_limit")
    timeout_demo = run_agent(s1_message, "S1", time_limit_seconds=0, run_id="demo_S1_timeout")

    crash_id = "demo_S1_crash"
    tools_before_crash: list[str] = []
    crash_message = ""
    try:
        run_agent(s1_message, "S1", run_id=crash_id, crash_after_tools=2)
    except SimulatedCrash as crash:
        crash_message = str(crash)
        tools_before_crash = json.loads(_checkpoint_path(crash_id).read_text(encoding="utf-8"))["tools_executed"]
    resumed = resume_agent(crash_id)

    with (config.OUTPUTS / "agent_demos.jsonl").open("w", encoding="utf-8") as log:
        for result in (token_demo, timeout_demo, resumed):
            log.write(_log_line(result) + "\n")
    return {
        "token_limit_200": {"status": token_demo["status"], "llm_calls": token_demo["llm_calls"],
                            "tokens_used": token_demo["tokens_used"], "tools_executed": token_demo["tools_executed"],
                            "final_answer": token_demo["final_answer"]},
        "time_limit_0": {"status": timeout_demo["status"], "llm_calls": timeout_demo["llm_calls"],
                         "final_answer": timeout_demo["final_answer"]},
        "crash_resume": {"crash": crash_message, "tools_before_crash": tools_before_crash,
                         "tools_after_resume": resumed["tools_executed"][len(tools_before_crash):],
                         "tools_all": resumed["tools_executed"], "status": resumed["status"],
                         "refund_inr": refund_amount(resumed), "llm_calls": resumed["llm_calls"]},
    }


def run() -> dict:
    """Start from empty logs so that a rerun reproduces the same files, then run everything."""
    config.OUTPUTS.mkdir(exist_ok=True)
    config.STATE.mkdir(exist_ok=True)
    for name in ("agent_runs.jsonl", "refund_ledger.jsonl"):
        (config.OUTPUTS / name).write_text("", encoding="utf-8")
    for old in config.STATE.glob("*.json"):
        old.unlink()
    payload = {"scenarios": run_scenarios(), "demonstrations": run_demonstrations()}
    config.save_metrics("part_d_agent", payload)
    logger.info("Part D agent finished: %s", [(r["scenario_id"], r["status"]) for r in payload["scenarios"]])
    return payload


if __name__ == "__main__":
    config.setup_logging()
    run()
