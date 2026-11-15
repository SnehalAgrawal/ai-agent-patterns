"""
The three systems compared in this demo. They share the model, the system prompt,
the tools (tools.py), and the agent loop; they differ only in who decides the steps.

  W  workflow: a router (one model call, structured output) picks a known intent or
     "other". For a known intent, code runs that intent's fixed tool calls and the
     model writes the reply from their results. "other" is handed to a person.
  A  agent: the model gets every tool and chooses what to call, in what order,
     for up to MAX_STEPS model calls.
  H  hybrid: the same router; known intents take the workflow, "other" goes to the
     agent loop (same cap).

The router prompt and the workflows below were written before the first model run.
"""
import json

from llm import chat, tool_message
from tools import TOOLS, run

MAX_STEPS = 5      # model calls in the agent loop
MAX_TOKENS = 1000  # per model call

INTENTS = {
    "refund_status": "Whether one or more of the customer's invoices can be refunded, or until when.",
    "invoice_lookup": "Facts about the customer's invoices, such as an amount or a date.",
    "help_center": "General questions about the product, its prices, or its policies.",
    "other": "Anything else, including questions that need investigating the account.",
}
ROUTER_PROMPT = ("You route customer support messages. Choose exactly one intent for the customer's message.\n\n"
                 + "\n".join(f"- {name}: {text}" for name, text in INTENTS.items())
                 + '\n\nAnswer with JSON only, in the form {"intent": "<one of the intents above>"}.')
ROUTER_SCHEMA = {"type": "object", "properties": {"intent": {"type": "string", "enum": list(INTENTS)}},
                 "required": ["intent"]}

# The fixed steps for each known intent: (tool, arguments). "{question}" is replaced by the customer's message.
WORKFLOWS = {
    "refund_status": [("get_refund_status", {})],
    "invoice_lookup": [("get_invoices", {})],
    "help_center": [("search_help_center", {"query": "{question}"})],
}
HANDOFF = ("Thanks for your message. I've passed it to our billing team, and a person will reply to you "
           "by email.")


def _model_call(record: dict, role: str, response: dict) -> None:
    record["model_calls"].append({"role": role} | {k: response[k] for k in (
        "finish_reason", "input_tokens", "cached_tokens", "output_tokens", "seconds")}
        | {"tool_calls": len(response["tool_calls"])})


def _new_record() -> dict:
    return {"route": None, "route_raw": None, "path": None, "calls": [], "model_calls": [],
            "final": None, "outcome": None, "error": None}


def _route(record: dict, question: str, seed: int) -> str:
    messages = [{"role": "system", "content": ROUTER_PROMPT}, {"role": "user", "content": question}]
    response = chat(messages, [], seed, 50, schema=ROUTER_SCHEMA)
    _model_call(record, "router", response)
    record["route_raw"] = response["content"]
    try:
        intent = json.loads(response["content"])["intent"]
    except (ValueError, KeyError, TypeError):
        intent = None
    record["route"] = intent if intent in INTENTS else "other"  # an unreadable route is treated as "other"
    return record["route"]


def _workflow(record: dict, intent: str, system_prompt: str, question: str, seed: int) -> None:
    """Runs the intent's fixed tool calls in code, then one model call to write the reply from the results."""
    record["path"] = "workflow"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
    planned = [(name, {k: v.replace("{question}", question) for k, v in args.items()}) for name, args in WORKFLOWS[intent]]
    # Shown to the model as an assistant turn that made these calls, followed by their results:
    # the same format the agent sees, so the only difference is who chose the calls.
    messages.append({"role": "assistant", "content": "",
                     "tool_calls": [{"function": {"name": name, "arguments": args}} for name, args in planned]})
    for name, args in planned:
        result = run(name, json.dumps(args))
        record["calls"].append(result)
        messages.append(tool_message(name, result["result_text"]))
    response = chat(messages, [], seed, MAX_TOKENS)
    _model_call(record, "phrase", response)
    if response["error"]:
        record["error"], record["outcome"] = response["error"], "error"
        return
    record["final"], record["outcome"] = response["content"], "answer"


def _agent(record: dict, system_prompt: str, question: str, seed: int) -> None:
    """The agent loop from Articles #8 and #23, with every tool."""
    record["path"] = "agent"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
    for _ in range(MAX_STEPS):
        response = chat(messages, TOOLS, seed, MAX_TOKENS)
        _model_call(record, "agent", response)
        if response["error"]:
            record["error"], record["outcome"] = response["error"], "error"
            return
        if not response["tool_calls"]:
            record["final"], record["outcome"] = response["content"], "answer"
            return
        messages.append(response["message"])
        for call in response["tool_calls"]:
            result = run(call["name"], call["arguments"])
            record["calls"].append(result)
            messages.append(tool_message(call["name"], result["result_text"]))
    record["outcome"] = "step_limit"


def workflow_system(system_prompt: str, question: str, seed: int) -> dict:
    record = _new_record()
    intent = _route(record, question, seed)
    if intent == "other":
        record["path"], record["final"], record["outcome"] = "handoff", HANDOFF, "handoff"
    else:
        _workflow(record, intent, system_prompt, question, seed)
    return record


def agent_system(system_prompt: str, question: str, seed: int) -> dict:
    record = _new_record()
    _agent(record, system_prompt, question, seed)
    return record


def hybrid_system(system_prompt: str, question: str, seed: int) -> dict:
    record = _new_record()
    intent = _route(record, question, seed)
    if intent == "other":
        _agent(record, system_prompt, question, seed)
    else:
        _workflow(record, intent, system_prompt, question, seed)
    return record


SYSTEMS = {"W": workflow_system, "A": agent_system, "H": hybrid_system}
