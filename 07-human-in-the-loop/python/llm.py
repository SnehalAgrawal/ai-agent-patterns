"""
Chat on a local Ollama model over plain HTTP, with optional tools and optional
structured output (the router's JSON).

Every reply is normalised to:

  {"content", "tool_calls": [{"name", "arguments"}], "finish_reason", "input_tokens",
   "cached_tokens", "output_tokens", "seconds", "error", "message"}

"input_tokens" is Ollama's prompt_eval_count (every input token, including those reused
from its cache, "cached_tokens"). "seconds" is Ollama's total_duration for the call.
"""
import json
import os

import requests

BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def model_name() -> str:
    return os.environ.get("OLLAMA_CHAT_MODEL", "qwen2.5:3b")


def chat(messages: list[dict], tools: list[dict], seed: int, max_tokens: int, schema: dict | None = None) -> dict:
    body = {"model": model_name(), "messages": messages, "stream": False,
            "options": {"seed": seed, "num_predict": max_tokens}}
    if tools:
        body["tools"] = tools
    if schema:
        body["format"] = schema
    try:
        response = requests.post(f"{BASE_URL}/api/chat", json=body, timeout=300)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        return {"content": "", "tool_calls": [], "finish_reason": None, "input_tokens": 0, "cached_tokens": 0,
                "output_tokens": 0, "seconds": 0.0, "error": str(exc)[:500], "message": None}
    raw_calls = data["message"].get("tool_calls") or []
    calls = [{"name": c["function"]["name"], "arguments": json.dumps(c["function"].get("arguments", {}))}
             for c in raw_calls]
    content = data["message"].get("content") or ""
    message = {"role": "assistant", "content": content}
    if raw_calls:
        message["tool_calls"] = raw_calls
    return {"content": content, "tool_calls": calls, "finish_reason": data.get("done_reason"),
            "input_tokens": data.get("prompt_eval_count", 0), "cached_tokens": data.get("prompt_eval_cached_count", 0),
            "output_tokens": data.get("eval_count", 0), "seconds": round(data.get("total_duration", 0) / 1e9, 3),
            "error": None, "message": message}


def tool_message(name: str, content: str) -> dict:
    """The message that returns one tool result to the model."""
    return {"role": "tool", "tool_name": name, "content": content}


def reset() -> None:
    """Unload the model, so every run starts from the same state and memory is freed afterwards."""
    requests.post(f"{BASE_URL}/api/generate", json={"model": model_name(), "keep_alive": 0}, timeout=60).raise_for_status()


def warm_up() -> None:
    """Load the model with one throwaway request, so the first measured conversation doesn't include load time."""
    requests.post(f"{BASE_URL}/api/chat", json={"model": model_name(), "stream": False, "options": {"num_predict": 1},
                                                 "messages": [{"role": "user", "content": "Hello"}]},
                  timeout=300).raise_for_status()


def other_models_loaded() -> list[str]:
    """Models other than ours that Ollama has loaded right now: a sign another process is using the machine."""
    try:
        loaded = requests.get(f"{BASE_URL}/api/ps", timeout=10).json().get("models", [])
    except (requests.RequestException, ValueError):
        return ["unknown (Ollama /api/ps failed)"]
    return sorted(m["name"] for m in loaded if m["name"] != model_name())


def runtime() -> dict:
    """Recorded with every run: Ollama's version, the model's tag and digest, and its chat template."""
    version = requests.get(f"{BASE_URL}/api/version", timeout=10).json().get("version")
    models = requests.get(f"{BASE_URL}/api/tags", timeout=10).json().get("models", [])
    digest = next((m.get("digest") for m in models if m.get("name") == model_name()), None)
    show = requests.post(f"{BASE_URL}/api/show", json={"model": model_name()}, timeout=30).json()
    return {"ollama": version, "model": model_name(), "digest": digest, "template": show.get("template", "")}
