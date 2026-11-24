"""
Design L: the agent loop. Its state is the message list, saved to loop_state.json in the
run directory after every message is appended (with the call log, so a resumed run keeps
counting model calls). The model decides every step, for up to MAX_MODEL_CALLS calls.

Resume: load the saved file; if the last assistant message has tool calls without
results, run them (the usual way to continue a transcript); then continue the loop.
"""
import json
import os
from pathlib import Path

import llm
from payments import DATA, PaymentService
from tools import TOOLS, Toolbox

DESIGNS = json.loads((DATA / "designs.json").read_text())
PROMPTS = json.loads((DATA / "prompts.json").read_text())
MAX_MODEL_CALLS = DESIGNS["limits"]["L_max_model_calls"]
MAX_TOKENS = DESIGNS["limits"]["max_tokens_per_call"]
SYSTEM = PROMPTS["loop_system"].format(system=PROMPTS["system"], refund_policy=PROMPTS["refund_policy"])


def _save(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, ensure_ascii=False))
    os.replace(tmp, path)


def _model_call(state: dict, response: dict) -> None:
    state["model_calls"].append({k: response[k] for k in (
        "finish_reason", "input_tokens", "cached_tokens", "output_tokens", "seconds", "error")}
        | {"tool_calls": len(response["tool_calls"]), "phase": state["phase"]})


def _run_pending(state: dict, toolbox: Toolbox, path: Path) -> None:
    """Run the tool calls of the last assistant message that have no result yet."""
    last = max(i for i, m in enumerate(state["messages"]) if m["role"] == "assistant")
    calls = state["messages"][last].get("tool_calls") or []
    answered = sum(1 for m in state["messages"][last + 1:] if m["role"] == "tool")
    for call in calls[answered:]:
        name = call["function"]["name"]
        result = toolbox.run(name, json.dumps(call["function"].get("arguments", {})))
        state["calls"].append(result | {"phase": state["phase"]})
        state["messages"].append(llm.tool_message(name, result["result_text"]))
        _save(path, state)


def run(run_dir: Path, message: str, seed: int, resume: bool = False, service: PaymentService | None = None) -> dict:
    path = Path(run_dir) / "loop_state.json"
    toolbox = Toolbox(service or PaymentService(Path(run_dir) / "ledger.json"))
    if resume:
        state = json.loads(path.read_text())
        state["phase"] = "resume"
        state["resumed"] = True
        if any(m["role"] == "assistant" for m in state["messages"]):
            _run_pending(state, toolbox, path)
    else:
        state = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": message}],
                 "calls": [], "model_calls": [], "final": None, "outcome": None, "phase": "first", "resumed": False}
        _save(path, state)

    while state["outcome"] is None:
        if len(state["model_calls"]) >= MAX_MODEL_CALLS:
            state["outcome"] = "step_limit"
            break
        response = llm.chat(state["messages"], TOOLS, seed, MAX_TOKENS)
        _model_call(state, response)
        if response["error"]:
            state["outcome"] = "model_error"
            break
        state["messages"].append(response["message"])
        _save(path, state)
        if not response["tool_calls"]:
            state["final"] = response["content"]
            state["outcome"] = "replied"
            break
        _run_pending(state, toolbox, path)
    _save(path, state)
    return state
