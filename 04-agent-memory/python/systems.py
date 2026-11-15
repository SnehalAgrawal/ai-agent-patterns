"""
The four memory designs, built from the frozen definitions in data/memory.json.

  N  no memory
  T  the earlier conversations' transcripts, verbatim
  S  one model-written summary per earlier conversation (Σ in the article)
  K  one model-extracted JSON notes object per earlier conversation

Each design produces a memory block (N: none), appended to the customer-facing prompt.
The new conversation then runs the agent loop from Articles #8, #23, #24 and #25, with the
five data tools. tools.py and llm.py are byte-identical copies of 02-agentic-workflows.
"""
import json

from llm import chat, tool_message
from tools import DATA, TOOLS, run

MEMORY = json.loads((DATA / "memory.json").read_text())
LIMITS = MEMORY["limits"]
MAX_STEPS = LIMITS["max_model_calls_per_conversation"]
MAX_TOKENS = LIMITS["max_tokens_per_call"]
MAX_MEMORY_TOKENS = LIMITS["max_tokens_per_memory_creation_call"]
SYSTEM_PROMPT = json.loads((DATA / "prompt.json").read_text())["system"]
CONVERSATIONS = json.loads((DATA / "conversations.json").read_text())["conversations"]
DESIGNS = MEMORY["designs"]
EMPTY_NOTES = {"preferences": [], "commitments": [], "open_items": [], "facts": []}


def _lines(conversation: dict) -> str:
    """One line per message, 'Customer: ...' or 'Assistant: ...', exactly as written."""
    return "\n".join(f"{m['role'].capitalize()}: {m['text']}" for m in conversation["messages"])


def transcripts() -> str:
    """The three conversations in the transcript format: '[date]' then the messages; blank line between."""
    return "\n\n".join(f"[{c['date']}]\n{_lines(c)}" for c in CONVERSATIONS)


def _model_call(calls: list, role: str, response: dict) -> None:
    calls.append({"role": role} | {k: response[k] for k in (
        "finish_reason", "input_tokens", "cached_tokens", "output_tokens", "seconds", "error")}
        | {"tool_calls": len(response["tool_calls"])})


def create_memory(design: str, seed: int) -> dict:
    """
    Builds the memory block a design carries into the new conversation. Returns
    {"design", "seed", "block", "items": [{"date", "output", ...}], "calls": [model-call records]}.
    """
    spec = DESIGNS[design]
    memory = {"design": design, "seed": seed, "block": None, "items": [], "calls": []}
    if design == "N":
        return memory
    if design == "T":
        memory["block"] = "Transcripts of your earlier conversations with this customer:\n\n" + transcripts()
        return memory
    creation = spec["creation"]
    for conversation in CONVERSATIONS:
        user = creation["user"].replace("<date>", conversation["date"]).replace(
            "<the conversation, in the transcript format without its date line>", _lines(conversation))
        messages = [{"role": "system", "content": creation["system"]}, {"role": "user", "content": user}]
        response = chat(messages, [], seed, MAX_MEMORY_TOKENS, schema=creation.get("schema"))
        _model_call(memory["calls"], f"create:{design}", response)
        item = {"date": conversation["date"], "raw": response["content"], "error": response["error"]}
        if design == "S":
            item["output"] = response["content"]
        else:
            try:
                notes = json.loads(response["content"])
                if not isinstance(notes, dict):
                    raise ValueError("not an object")
                item["output"], item["parse_failed"] = notes, False
            except ValueError:
                item["output"], item["parse_failed"] = EMPTY_NOTES, True
        memory["items"].append(item)
    if design == "S":
        memory["block"] = "Summaries of your earlier conversations with this customer:\n\n" + "\n\n".join(
            f"[{i['date']}] {i['output']}" for i in memory["items"])
    else:
        notes = {i["date"]: i["output"] for i in memory["items"]}
        memory["block"] = ("Notes from your earlier conversations with this customer (JSON, one object per "
                           "conversation date):\n" + json.dumps(notes, ensure_ascii=False, indent=1))
    return memory


def system_prompt(block: str | None) -> str:
    """The frozen placement: <prompt>\\n\\n<memory block>; N appends nothing."""
    return SYSTEM_PROMPT if block is None else f"{SYSTEM_PROMPT}\n\n{block}"


def conversation(block: str | None, question: str, seed: int) -> dict:
    """The new conversation: the agent loop with the five data tools and the design's memory block."""
    record = {"model_calls": [], "calls": [], "final": None, "outcome": None}
    messages = [{"role": "system", "content": system_prompt(block)}, {"role": "user", "content": question}]
    for _ in range(MAX_STEPS):
        response = chat(messages, TOOLS, seed, MAX_TOKENS)
        _model_call(record["model_calls"], "agent", response)
        if response["error"]:
            record["outcome"] = "error"
            return record
        if not response["tool_calls"]:
            record["final"], record["outcome"] = response["content"], "answer"
            return record
        messages.append(response["message"])
        for call in response["tool_calls"]:
            result = run(call["name"], call["arguments"])
            record["calls"].append(result)
            messages.append(tool_message(call["name"], result["result_text"]))
    record["outcome"] = "step_limit"
    return record
