"""
Chat with tool calling on a local Ollama model, over plain HTTP.

Every reply is normalised to:

  {"content": str, "tool_calls": [{"name", "arguments"}], "finish_reason",
   "input_tokens", "cached_tokens", "output_tokens", "error", "message"}

"arguments" is the JSON text of the arguments the model produced. "input_tokens"
is Ollama's prompt_eval_count: every input token of the request, including those
it reused from its cache ("cached_tokens", prompt_eval_cached_count).
"""
import json
import os

import requests

BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def model_name() -> str:
    return os.environ.get("OLLAMA_CHAT_MODEL", "llama3.2:3b")


def chat(messages: list[dict], tools: list[dict], seed: int, max_tokens: int) -> dict:
    body = {"model": model_name(), "messages": messages, "tools": tools, "stream": False,
            "options": {"seed": seed, "num_predict": max_tokens}}
    try:
        response = requests.post(f"{BASE_URL}/api/chat", json=body, timeout=300)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        # The run continues and the conversation is recorded as failed.
        return {"content": "", "tool_calls": [], "finish_reason": None, "input_tokens": 0, "cached_tokens": 0,
                "output_tokens": 0, "error": str(exc)[:500], "message": None}
    raw_calls = data["message"].get("tool_calls") or []
    # Ollama returns arguments as an object; keep JSON text, as a hosted API would return it.
    calls = [{"name": c["function"]["name"], "arguments": json.dumps(c["function"].get("arguments", {}))}
             for c in raw_calls]
    content = data["message"].get("content") or ""
    message = {"role": "assistant", "content": content}
    if raw_calls:
        message["tool_calls"] = raw_calls
    return {"content": content, "tool_calls": calls, "finish_reason": data.get("done_reason"),
            "input_tokens": data.get("prompt_eval_count", 0), "cached_tokens": data.get("prompt_eval_cached_count", 0),
            "output_tokens": data.get("eval_count", 0), "error": None, "message": message}


def tool_message(call: dict, content: str) -> dict:
    """The message that returns one tool result to the model."""
    return {"role": "tool", "tool_name": call["name"], "content": content}


def reset() -> None:
    """Unload the model, so every run starts from the same state and memory is freed afterwards."""
    requests.post(f"{BASE_URL}/api/generate", json={"model": model_name(), "keep_alive": 0}, timeout=60).raise_for_status()


def runtime() -> dict:
    """
    What is recorded with every run: Ollama's version, the model's tag and digest,
    and the chat template that turns messages and tool definitions into the prompt.
    The template decides what the model actually sees (for example, whether tool
    definitions are still in the prompt after a tool result); Ollama doesn't return
    the rendered prompt itself.
    """
    version = requests.get(f"{BASE_URL}/api/version", timeout=10).json().get("version")
    models = requests.get(f"{BASE_URL}/api/tags", timeout=10).json().get("models", [])
    digest = next((m.get("digest") for m in models if m.get("name") == model_name()), None)
    show = requests.post(f"{BASE_URL}/api/show", json={"model": model_name()}, timeout=30).json()
    return {"ollama": version, "model": model_name(), "digest": digest, "template": show.get("template", "")}
