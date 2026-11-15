"""
The three systems compared in this demo, built from the frozen definitions in data/agents.json.

  S  single agent: Article #24's agent A, unchanged.
  M  manager + specialists: the manager's only tools are two agents (billing, help center).
     Each specialist runs its own agent loop on the manager's request alone.
  R  agent + reviewer: S writes a draft; a reviewer agent with the five data tools checks it
     and replies APPROVE (the draft is sent) or with a replacement reply. One pass.

Every agent uses the same loop (the one from Articles #8, #23 and #24), the same model, the
same seed for every call in the conversation, and the same limits. tools.py and llm.py are
byte-identical copies of 02-agentic-workflows.
"""
import json
import re

from llm import chat, tool_message
from tools import BY_NAME, DATA, TOOLS, run

AGENTS = json.loads((DATA / "agents.json").read_text())
LIMITS = AGENTS["limits"]
MAX_STEPS = LIMITS["max_model_calls_per_agent_loop"]
MAX_TOKENS = LIMITS["max_tokens_per_call"]
SYSTEM_PROMPT = json.loads((DATA / "prompt.json").read_text())["system"]

M = AGENTS["systems"]["M"]
R = AGENTS["systems"]["R"]
SPECIALISTS = {s["tool"]: {"id": sid, **s} for sid, s in M["specialists"].items()}
NO_REPLY_TEXT = "The specialist couldn't finish this request."
NO_DRAFT_TEXT = "(The assistant produced no reply.)"
REVIEWER = R["reviewer"]


def _tool_def(t: dict) -> dict:
    """The same shape as the data tools' definitions in tools.py."""
    return {"type": "function", "function": {
        "name": t["name"], "description": t["description"],
        "parameters": {"type": "object", "properties": t["parameters"], "required": t["required"],
                       "additionalProperties": False}}}


MANAGER_TOOLS = [_tool_def(t) for t in M["manager_tools"]]


def _subset(names: list[str]) -> list[dict]:
    return [t for t in TOOLS if t["function"]["name"] in names]


def _new_record() -> dict:
    return {"model_calls": [], "calls": [], "handoffs": [], "final": None, "outcome": None, "error": None,
            "writer": None, "draft": None, "draft_outcome": None, "review_text": None, "review_outcome": None,
            "action": None}


def _model_call(record: dict, role: str, response: dict) -> None:
    record["model_calls"].append({"role": role} | {k: response[k] for k in (
        "finish_reason", "input_tokens", "cached_tokens", "output_tokens", "seconds", "error")}
        | {"tool_calls": len(response["tool_calls"])})


def _data_tool(allowed: list[str]):
    """Runs a data tool for an agent that was given only `allowed`; any other name is an unknown tool."""
    def execute(name: str, arguments: str) -> dict:
        if name not in allowed:
            return {"tool": name, "arguments": arguments, "valid": False, "executed": False,
                    "result_text": json.dumps({"error": f"Unknown tool: {name}"}, ensure_ascii=False)}
        return run(name, arguments)
    return execute


def agent_loop(record: dict, role: str, system_prompt: str, user_message: str, tools: list[dict],
               execute, seed: int) -> tuple[str | None, str]:
    """
    The agent loop from Articles #8, #23 and #24. Returns (final text, outcome), where outcome is
    "answer", "step_limit", or "error". Every model call and tool call is recorded with its role.
    """
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]
    for _ in range(MAX_STEPS):
        response = chat(messages, tools, seed, MAX_TOKENS)
        _model_call(record, role, response)
        if response["error"]:
            return None, "error"
        if not response["tool_calls"]:
            return response["content"], "answer"
        messages.append(response["message"])
        for call in response["tool_calls"]:
            result = execute(call["name"], call["arguments"])
            record["calls"].append({"role": role, **result})
            messages.append(tool_message(call["name"], result["result_text"]))
    return None, "step_limit"


# ---------- S ----------

def single_agent(question: str, seed: int) -> dict:
    record = _new_record()
    final, outcome = agent_loop(record, "agent", SYSTEM_PROMPT, question, TOOLS, run, seed)
    record["final"], record["outcome"], record["writer"] = final, outcome, "agent"
    return record


# ---------- M ----------

def _request_problem(name: str, arguments: str) -> tuple[dict | None, str | None]:
    """Validates a manager's call to a specialist the way tools.run validates a data-tool call."""
    if name not in SPECIALISTS:
        return None, f"Unknown tool: {name}"
    try:
        args = json.loads(arguments) if arguments.strip() else {}
    except json.JSONDecodeError:
        return None, "Arguments are not valid JSON."
    if not isinstance(args, dict):
        return None, "Invalid arguments: arguments must be a JSON object."
    if "request" not in args:
        return None, "Invalid arguments: missing required argument(s): request."
    extra = [k for k in args if k != "request"]
    if extra:
        return None, f"Invalid arguments: unknown argument(s): {', '.join(extra)}."
    if not isinstance(args["request"], str):
        return None, "Invalid arguments: argument 'request' must be a string."
    if not args["request"].strip():
        return None, "Invalid arguments: argument 'request' is empty."
    return args, None


def _ask_specialist(record: dict, seed: int):
    def execute(name: str, arguments: str) -> dict:
        args, problem = _request_problem(name, arguments)
        if problem:
            return {"tool": name, "arguments": arguments, "valid": False, "executed": False,
                    "result_text": json.dumps({"error": problem}, ensure_ascii=False)}
        spec = SPECIALISTS[name]
        role = f"specialist:{spec['id']}"
        reply, outcome = agent_loop(record, role, spec["prompt"], args["request"], _subset(spec["tools"]),
                                    _data_tool(spec["tools"]), seed)
        replied = bool(reply and reply.strip())
        text = reply if replied else NO_REPLY_TEXT
        record["handoffs"].append({"specialist": spec["id"], "request": args["request"], "outcome": outcome,
                                   "replied": replied, "reply": text})
        # executed = a specialist ran and its own reply came back (not the fixed text)
        return {"tool": name, "arguments": arguments, "valid": True, "executed": replied, "result_text": text}
    return execute


def manager_system(question: str, seed: int) -> dict:
    record = _new_record()
    final, outcome = agent_loop(record, "manager", SYSTEM_PROMPT, question, MANAGER_TOOLS,
                                _ask_specialist(record, seed), seed)
    record["final"], record["outcome"], record["writer"] = final, outcome, "manager"
    return record


# ---------- R ----------

def verdict(text: str | None, outcome: str) -> str:
    """The frozen verdict rule (data/agents.json, systems.R.verdict)."""
    if outcome != "answer" or text is None or not re.sub(r"\s", "", text):
        return "no_verdict"
    normalized = re.sub(r"[\s*.!\"'`]", "", text)
    return "approve" if normalized.upper() == "APPROVE" else "rewrite"


def reviewer_system(question: str, seed: int) -> dict:
    record = _new_record()
    draft, draft_outcome = agent_loop(record, "drafter", SYSTEM_PROMPT, question, TOOLS, run, seed)
    record["draft"], record["draft_outcome"] = draft, draft_outcome
    has_draft = draft_outcome == "answer" and bool(draft and draft.strip())
    message = (REVIEWER["user_message"]
               .replace("<the customer's message>", question)
               .replace("<the draft>", draft if has_draft else NO_DRAFT_TEXT))
    review, review_outcome = agent_loop(record, "reviewer", REVIEWER["prompt"], message,
                                        _subset(REVIEWER["tools"]), _data_tool(REVIEWER["tools"]), seed)
    record["review_text"], record["review_outcome"] = review, review_outcome
    record["action"] = verdict(review, review_outcome)
    if record["action"] == "rewrite":
        record["final"], record["outcome"], record["writer"] = review, "answer", "reviewer"
    else:
        record["final"], record["outcome"], record["writer"] = draft, draft_outcome, "drafter"
    return record


SYSTEMS = {"S": single_agent, "M": manager_system, "R": reviewer_system}
assert set(BY_NAME) >= {n for s in SPECIALISTS.values() for n in s["tools"]} | set(REVIEWER["tools"])
