"""
Runs every design on every task and saves everything.

  python run.py                                              # qwen2.5:3b, designs W L P R, seeds 1-3
  python run.py --seeds 1 --tasks Q3 Q7 --out results-check.jsonl   # a small check run

Before anything runs, data/ is checked against the hashes in data/FROZEN.json. The model is
unloaded, loaded with one throwaway request (so nothing includes load time), and unloaded at
the end. Conversations run in a fixed order: design, task, seed. "seconds" is end-to-end
wall-clock time per conversation.
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="qwen2.5:3b")
parser.add_argument("--designs", nargs="+", default=["W", "L", "P", "R"])
parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
parser.add_argument("--tasks", nargs="+")
parser.add_argument("--out")
args = parser.parse_args()
os.environ["OLLAMA_CHAT_MODEL"] = args.model

from llm import other_models_loaded, reset, runtime, warm_up  # noqa: E402
from systems import LIMITS, SYSTEMS, TASKS  # noqa: E402
from tools import DATA  # noqa: E402

frozen = json.loads((DATA / "FROZEN.json").read_text())["sha256"]
for name, expected in frozen.items():
    actual = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"data/{name} changed since it was frozen (sha256 {actual}, expected {expected}).")

tasks = [t for t in TASKS.values() if not args.tasks or t["id"] in args.tasks]
stem = args.model.replace(":", "_").replace("/", "_")
out = args.out or f"results-{stem}.jsonl"


def sha(text: str | None) -> str:
    return hashlib.sha256((text or "").encode()).hexdigest()


def power_source() -> str:
    """macOS only: the first line of `pmset -g batt` (AC or battery), recorded because timings depend on it."""
    try:
        return subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=10).stdout.splitlines()[0]
    except (OSError, IndexError, subprocess.SubprocessError):
        return "unknown"


info = runtime()
template = info.pop("template")
with open(f"template-{stem}.txt", "w") as f:
    f.write(template)
setup = info | {"template_sha256": sha(template), "limits": {k: v for k, v in LIMITS.items() if k != "_comment"},
                "sampling": "Ollama defaults, seed = run seed for every call", "frozen_sha256": frozen,
                "code_sha256": {n: hashlib.sha256((DATA.parent / "python" / n).read_bytes()).hexdigest()
                                for n in ("systems.py", "tools.py", "llm.py", "run.py")},
                "power": power_source(),
                "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
with open(out.replace(".jsonl", "-setup.json"), "w") as f:
    f.write(json.dumps(setup, indent=1) + "\n")
print(json.dumps(setup, indent=1))

reset()
warm_up()
started = time.monotonic()
total = len(args.designs) * len(tasks) * len(args.seeds)
with open(out, "w") as f:
    done = 0
    for design in args.designs:
        for task in tasks:
            for seed in args.seeds:
                before = other_models_loaded()  # checked outside the timed section
                t0 = time.monotonic()
                record = SYSTEMS[design](task, seed)
                seconds = round(time.monotonic() - t0, 3)
                after = other_models_loaded()
                row = {"model": setup["model"], "digest": setup["digest"], "ollama": setup["ollama"],
                       "design": design, "task": task["id"], "seed": seed, **record,
                       "reply_sha256": sha(record["final"]), "seconds": seconds,
                       "other_models_loaded": sorted(set(before) | set(after))}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                done += 1
                tools_used = [c["tool"] for c in record["calls"]]
                print(f"{done:>4}/{total}  {design}  {task['id']}  seed {seed}  {record['outcome']}"
                      f"{' ' + record['limit'] if record['limit'] else ''}  {len(record['model_calls'])} call(s) "
                      f"{tools_used}  {seconds:.1f} s")
reset()
print(f"\n{total} conversations in {time.monotonic() - started:.0f} s, saved to {out}. Model unloaded.")
