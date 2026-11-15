"""
The agent loop, the same as in ai-engineering-examples/tool-calling (Article #8):

  1. Send the conversation and the toolset's definitions to the model.
  2. If the reply contains tool calls, run each one through the toolset
     (parse, validate, execute) and append the results to the conversation.
  3. Repeat until the model replies without tool calls, or MAX_STEPS is reached.

MAX_STEPS counts model calls, not tool calls: one reply can request several tools.
"""
from llm import chat, tool_message
from toolsets import Toolset

MAX_STEPS = 5
MAX_TOKENS = 1000


def run_conversation(toolset: Toolset, system_prompt: str, question: str, seed: int) -> dict:
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
    calls, model_calls = [], []
    final, error = None, None
    for _ in range(MAX_STEPS):
        response = chat(messages, toolset.tools, seed, MAX_TOKENS)
        model_calls.append({k: response[k] for k in ("finish_reason", "input_tokens", "cached_tokens", "output_tokens")}
                           | {"tool_calls": len(response["tool_calls"])})
        if response["error"]:
            error = response["error"]
            break
        if not response["tool_calls"]:
            final = response["content"]
            break
        messages.append(response["message"])
        for call in response["tool_calls"]:
            record = toolset.run(call["name"], call["arguments"])
            calls.append(record)
            messages.append(tool_message(call, record["result_text"]))
    return {"final": final, "calls": calls, "model_calls": model_calls,
            "step_limit": final is None and error is None, "error": error}
