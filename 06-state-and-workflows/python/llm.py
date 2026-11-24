"""
Chat on a local Ollama model over plain HTTP, with optional tools and optional
structured output, plus the lock that keeps this demo from sharing Ollama with
another process during a measured run.

Every reply is normalised to:

  {"content", "tool_calls": [{"name", "arguments"}], "finish_reason", "input_tokens",
   "cached_tokens", "output_tokens", "seconds", "error", "message"}

"input_tokens" is Ollama's prompt_eval_count. "seconds" is Ollama's total_duration for the call.
chat(), tool_message(), reset(), warm_up(), and runtime() are 02-agentic-workflows' code,
unchanged; the lock and the idle checks are new.
"""
import json
import os
import shutil
import signal
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# A directory, because creating one is atomic: whoever creates it holds the lock.
# Other local tools working with the same Ollama server use the same path.
LOCK_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "snehalagrawal-ollama.lock"
IDLE_SECONDS = 60  # Ollama must have no loaded model and no open connection for this long before a run starts


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


def loaded_models() -> list[str]:
    try:
        return sorted(m["name"] for m in requests.get(f"{BASE_URL}/api/ps", timeout=10).json().get("models", []))
    except (requests.RequestException, ValueError):
        return ["unknown (Ollama /api/ps failed)"]


def other_models_loaded() -> list[str]:
    """Models other than ours that Ollama has loaded right now: a sign another process is using the machine."""
    return [m for m in loaded_models() if m != model_name()]


def open_connections() -> int:
    """Established TCP connections to Ollama's port, from any process (lsof). -1 if lsof isn't available."""
    port = BASE_URL.rsplit(":", 1)[-1].strip("/")
    try:
        out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:ESTABLISHED"], capture_output=True, text=True,
                             timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return -1
    return max(0, len(out.strip().splitlines()) - 1)  # minus the header line


def runtime() -> dict:
    """Recorded with every run: Ollama's version, the model's tag and digest, and its chat template."""
    version = requests.get(f"{BASE_URL}/api/version", timeout=10).json().get("version")
    models = requests.get(f"{BASE_URL}/api/tags", timeout=10).json().get("models", [])
    digest = next((m.get("digest") for m in models if m.get("name") == model_name()), None)
    show = requests.post(f"{BASE_URL}/api/show", json={"model": model_name()}, timeout=30).json()
    return {"ollama": version, "model": model_name(), "digest": digest, "template": show.get("template", "")}


def _wait_until_idle() -> None:
    """Wait until no model is loaded and no connection is open, continuously, for IDLE_SECONDS."""
    quiet_since = None
    was_busy = False
    while True:
        loaded, connections = loaded_models(), open_connections()
        busy = bool(loaded) or connections != 0
        now = time.monotonic()
        if busy:
            if not was_busy:
                print(f"Ollama is in use (loaded: {loaded}, connections: {connections}); waiting.", flush=True)
            was_busy = True
            quiet_since = None
        elif quiet_since is None:
            was_busy = False
            quiet_since = now
        elif now - quiet_since >= IDLE_SECONDS:
            return
        time.sleep(5)


@contextmanager
def ollama_lock(purpose: str):
    """
    Hold the shared Ollama lock for the duration of the block. If another process holds it,
    wait; never take it over. Once held, wait until Ollama has been idle for IDLE_SECONDS,
    because a process that doesn't use the lock may still be generating.
    The lock is released when the block ends, including on an exception, Ctrl-C, or SIGTERM.
    """
    waiting_said = False
    while True:
        try:
            LOCK_DIR.mkdir()
            break
        except FileExistsError:
            if not waiting_said:
                try:
                    owner = (LOCK_DIR / "owner.json").read_text()
                except OSError:
                    owner = "(no owner file)"
                print(f"Waiting for the Ollama lock {LOCK_DIR}, held by: {owner}", flush=True)
                waiting_said = True
            time.sleep(15)
    previous = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(143)))
    try:
        (LOCK_DIR / "owner.json").write_text(json.dumps({
            "pid": os.getpid(), "purpose": purpose,
            "since_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}))
        print(f"Holding the Ollama lock ({purpose}). Waiting for {IDLE_SECONDS} s of idle Ollama first.", flush=True)
        _wait_until_idle()
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)
        shutil.rmtree(LOCK_DIR, ignore_errors=True)
        print("Released the Ollama lock.", flush=True)
