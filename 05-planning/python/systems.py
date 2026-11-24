"""
The four designs, built from the frozen definitions in data/designs.json. They share the
model, the customer-facing prompt, the five read-only tools (tools.py, byte-identical to
04-agent-memory), and the limits; they differ in who decides the steps and when.

  W  workflow: a developer-written plan per task, run by code; the model writes the reply
  L  loop: the agent loop of Articles #8 and #23-#26; the model decides one response at a time
  P  plan, then execute: one planning call writes every step before any tool runs
  R  plan, execute, replan: P, then a replanner sees the results and is done or plans more

W, P and R share the answer step: one model call without tools, which sees the customer's
message and every attempted step with its result, in the format the loop sees them.

Every tool call goes through execute(), which applies the task's injected failure (Q7:
get_refund_status is unavailable). Tools only read, so a call that is replaced by the
injected error has no effect.
"""
import json

from llm import chat, tool_message
from tools import DATA, TOOLS, run

SPEC = json.loads((DATA / "designs.json").read_text())
LIMITS = SPEC["limits"]
MAX_CALLS = LIMITS["max_model_calls_per_conversation"]
MAX_TOKENS = LIMITS["max_tokens_per_call"]
MAX_STEPS = LIMITS["max_steps_executed_per_plan"]
MAX_REPLANS = MAX_CALLS - 2  # one call is the first plan, one is reserved for the answer
SYSTEM_PROMPT = json.loads((DATA / "prompt.json").read_text())["system"]
TASKS = {t["id"]: t for t in json.loads((DATA / "tasks.json").read_text())["tasks"]}
DESIGNS = SPEC["designs"]


def _prompt(text: str) -> str:
    return text.replace("<tool_list>", SPEC["tool_list"])


P_PLANNER = _prompt(DESIGNS["P"]["planner_system"])
R_PLANNER = _prompt(DESIGNS["R"]["planner_system"])
R_REPLANNER = _prompt(DESIGNS["R"]["replanner_system"])


def _new_record() -> dict:
    return {"model_calls": [], "calls": [], "plans": [], "final": None, "outcome": None, "limit": None,
            "error": None}


def _model_call(record: dict, role: str, response: dict) -> None:
    record["model_calls"].append({"role": role} | {k: response[k] for k in (
        "finish_reason", "input_tokens", "cached_tokens", "output_tokens", "seconds", "error")}
        | {"tool_calls": len(response["tool_calls"])})


def execute(task: dict, name: str, arguments: str, source: str) -> dict:
    """Runs one tool call through the shared tool code, then applies the task's injected failure, if any."""
    result = run(name, arguments)
    failure = task["injected_failure"]
    injected = bool(failure and name == failure["tool"] and result["valid"])
    if injected:
        result = {**result, "executed": False, "result_text": json.dumps(failure["result"], ensure_ascii=False)}
    return {**result, "injected_failure": injected, "source": source}


def _answer(record: dict, question: str, attempted: list[dict], seed: int) -> None:
    """The answer step shared by W, P and R: no tools; the attempted steps and results as the loop would show them."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    if attempted:
        calls = []
        for c in attempted:
            try:
                args = json.loads(c["arguments"])
            except ValueError:
                args = None
            # Ollama needs an object here; arguments that weren't one were rejected by the tool code anyway.
            calls.append({"function": {"name": c["tool"], "arguments": args if isinstance(args, dict) else {}}})
        messages.append({"role": "assistant", "content": "", "tool_calls": calls})
        messages += [tool_message(c["tool"], c["result_text"]) for c in attempted]
    response = chat(messages, [], seed, MAX_TOKENS)
    _model_call(record, "answer", response)
    if response["error"]:
        record["error"], record["outcome"] = response["error"], "error"
        return
    record["final"], record["outcome"] = response["content"], "answer"


def _plan_call(record: dict, role: str, system: str, user: str, schema: dict, seed: int) -> dict:
    """One planning or replanning call. Unparseable replies are recorded as such (data/designs.json, plan_parsing)."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    response = chat(messages, [], seed, MAX_TOKENS, schema=schema)
    _model_call(record, role, response)
    plan = {"role": role, "raw": response["content"], "error": response["error"], "parsed": False,
            "reasoning": None, "done": None, "steps": [], "executed": 0}
    record["plans"].append(plan)
    if response["error"]:
        return plan
    try:
        data = json.loads(response["content"])
        if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
            raise ValueError("not a plan")
    except ValueError:
        return plan
    plan.update(parsed=True, reasoning=data.get("reasoning"), done=data.get("done"), steps=data["steps"])
    return plan


def _run_plan(record: dict, task: dict, plan: dict, source: str) -> list[dict]:
    """Runs a plan's steps in order (at most MAX_STEPS). No step sees another's result."""
    attempted = []
    for step in plan["steps"][:MAX_STEPS]:
        step = step if isinstance(step, dict) else {}
        name = str(step.get("tool", ""))
        arguments = json.dumps(step.get("arguments", {}), ensure_ascii=False)
        result = execute(task, name, arguments, source)
        record["calls"].append(result)
        attempted.append(result)
    plan["executed"] = len(attempted)
    return attempted


def _replan_user(question: str, attempted: list[dict]) -> str:
    lines = [f"{i}. {c['tool']} {c['arguments']} -> {c['result_text']}" for i, c in enumerate(attempted, 1)]
    return (f"Customer's message: {question}\n\nSteps that have already run, with their results:\n"
            + ("\n".join(lines) if lines else "None."))


def workflow(task: dict, seed: int) -> dict:
    record = _new_record()
    attempted = []
    for name, args in DESIGNS["W"]["plans"][task["id"]]:
        args = {k: v.replace("{question}", task["question"]) for k, v in args.items()}
        result = execute(task, name, json.dumps(args, ensure_ascii=False), "workflow")
        record["calls"].append(result)
        attempted.append(result)
    _answer(record, task["question"], attempted, seed)
    return record


def loop(task: dict, seed: int) -> dict:
    record = _new_record()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task["question"]}]
    for _ in range(MAX_CALLS):
        response = chat(messages, TOOLS, seed, MAX_TOKENS)
        _model_call(record, "agent", response)
        if response["error"]:
            record["error"], record["outcome"] = response["error"], "error"
            return record
        if not response["tool_calls"]:
            record["final"], record["outcome"] = response["content"], "answer"
            return record
        messages.append(response["message"])
        source = f"response {len(record['model_calls'])}"
        for call in response["tool_calls"]:
            result = execute(task, call["name"], call["arguments"], source)
            record["calls"].append(result)
            messages.append(tool_message(call["name"], result["result_text"]))
    record["outcome"], record["limit"] = "step_limit", "step_limit"
    return record


def plan_execute(task: dict, seed: int) -> dict:
    record = _new_record()
    plan = _plan_call(record, "plan", P_PLANNER, task["question"], SPEC["plan_schema"], seed)
    if plan["error"]:
        record["error"], record["outcome"] = plan["error"], "error"
        return record
    attempted = _run_plan(record, task, plan, "plan 1")
    _answer(record, task["question"], attempted, seed)
    return record


def plan_replan(task: dict, seed: int) -> dict:
    record = _new_record()
    plan = _plan_call(record, "plan", R_PLANNER, task["question"], SPEC["plan_schema"], seed)
    if plan["error"]:
        record["error"], record["outcome"] = plan["error"], "error"
        return record
    attempted = _run_plan(record, task, plan, "plan 1")
    for n in range(1, MAX_REPLANS + 1):
        replan = _plan_call(record, "replan", R_REPLANNER, _replan_user(task["question"], attempted),
                            SPEC["replan_schema"], seed)
        if replan["error"]:
            record["error"], record["outcome"] = replan["error"], "error"
            return record
        # Done when the replanner says so, plans nothing, or its reply can't be read (designs.json, plan_parsing).
        if not replan["parsed"] or replan["done"] is True or not replan["steps"]:
            break
        if n == MAX_REPLANS:
            record["limit"] = "replan_limit"  # the last allowed replan still planned steps: they don't run
            break
        attempted += _run_plan(record, task, replan, f"plan {n + 1}")
    _answer(record, task["question"], attempted, seed)
    return record


SYSTEMS = {"W": workflow, "L": loop, "P": plan_execute, "R": plan_replan}
