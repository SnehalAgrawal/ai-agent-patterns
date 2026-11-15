"""
Asks every question with every toolset and saves each conversation.

  python run.py                                  # llama3.2:3b, toolsets A B C, seeds 1-3
  python run.py --model qwen2.5:3b
  python run.py --toolsets A --seeds 1 --questions Q01 Q09 --out results-check.jsonl   # a small check run

Before anything runs, the test material in data/ is checked against the hashes
in data/FROZEN.json. The model is unloaded before and after the run, and the
conversations run in a fixed order: toolset, then question, then seed.
"""
import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="llama3.2:3b")
parser.add_argument("--toolsets", nargs="+", default=["A", "B", "C"])
parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
parser.add_argument("--questions", nargs="+")
parser.add_argument("--out")
args = parser.parse_args()
os.environ["OLLAMA_CHAT_MODEL"] = args.model

from agent import MAX_STEPS, MAX_TOKENS, run_conversation  # noqa: E402
from llm import reset, runtime  # noqa: E402
from toolsets import DATA, TOOLSETS  # noqa: E402

frozen = json.loads((DATA / "FROZEN.json").read_text())["sha256"]
for name, expected in frozen.items():
    actual = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"data/{name} changed since it was frozen (sha256 {actual}, expected {expected}).")

system_prompt = json.loads((DATA / "prompt.json").read_text())["system"]
questions = json.loads((DATA / "questions.json").read_text())["questions"]
if args.questions:
    questions = [q for q in questions if q["id"] in args.questions]
out = args.out or f"results-{args.model.replace(':', '_').replace('/', '_')}.jsonl"

info = runtime()
template = info.pop("template")
template_file = f"template-{args.model.replace(':', '_').replace('/', '_')}.txt"
with open(template_file, "w") as f:
    f.write(template)
setup = info | {"template_sha256": hashlib.sha256(template.encode()).hexdigest(), "template_file": template_file,
                "max_steps": MAX_STEPS, "max_tokens": MAX_TOKENS, "sampling": "Ollama defaults, seed = run seed",
                "frozen_sha256": frozen, "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
print(json.dumps(setup, indent=1))
with open(out.replace(".jsonl", "-setup.json"), "w") as f:
    f.write(json.dumps(setup, indent=1) + "\n")
total = len(args.toolsets) * len(questions) * len(args.seeds)

reset()
started = time.monotonic()
with open(out, "w") as f:
    done = 0
    for toolset_name in args.toolsets:
        toolset = TOOLSETS[toolset_name]
        for q in questions:
            for seed in args.seeds:
                t0 = time.monotonic()
                conversation = run_conversation(toolset, system_prompt, q["question"], seed)
                record = {"model": setup["model"], "digest": setup["digest"], "ollama": setup["ollama"],
                          "toolset": toolset_name, "question": q["id"], "seed": seed, **conversation,
                          "reply_sha256": hashlib.sha256((conversation["final"] or "").encode()).hexdigest(),
                          "seconds": round(time.monotonic() - t0, 2)}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                done += 1
                print(f"{done:>4}/{total}  {toolset_name}  {q['id']}  seed {seed}  "
                      f"{len(conversation['model_calls'])} model call(s)  {record['seconds']:.1f} s")
reset()
print(f"\n{total} conversations in {time.monotonic() - started:.0f} s, saved to {out}. Model unloaded.")
