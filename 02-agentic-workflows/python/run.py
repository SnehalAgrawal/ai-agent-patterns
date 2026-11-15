"""
Asks every question of every system and saves each conversation.

  python run.py                                      # qwen2.5:3b, systems W A H, seeds 1-3
  python run.py --systems W A H --seeds 1 --questions K1 O1 --out results-check.jsonl   # a small check run

Before anything runs, data/ is checked against the hashes in data/FROZEN.json. The model
is unloaded before the run, loaded with one throwaway request (so no conversation includes
load time), and unloaded after. Conversations run in a fixed order: system, question, seed.
"""
import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="qwen2.5:3b")
parser.add_argument("--systems", nargs="+", default=["W", "A", "H"])
parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
parser.add_argument("--questions", nargs="+")
parser.add_argument("--out")
args = parser.parse_args()
os.environ["OLLAMA_CHAT_MODEL"] = args.model

from llm import other_models_loaded, reset, runtime, warm_up  # noqa: E402
from systems import MAX_STEPS, MAX_TOKENS, ROUTER_PROMPT, SYSTEMS  # noqa: E402
from tools import DATA  # noqa: E402

frozen = json.loads((DATA / "FROZEN.json").read_text())["sha256"]
for name, expected in frozen.items():
    actual = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"data/{name} changed since it was frozen (sha256 {actual}, expected {expected}).")

system_prompt = json.loads((DATA / "prompt.json").read_text())["system"]
questions = json.loads((DATA / "questions.json").read_text())["questions"]
if args.questions:
    questions = [q for q in questions if q["id"] in args.questions]
stem = args.model.replace(":", "_").replace("/", "_")
out = args.out or f"results-{stem}.jsonl"

info = runtime()
template = info.pop("template")
with open(f"template-{stem}.txt", "w") as f:
    f.write(template)
setup = info | {"template_sha256": hashlib.sha256(template.encode()).hexdigest(),
                "router_prompt_sha256": hashlib.sha256(ROUTER_PROMPT.encode()).hexdigest(),
                "max_steps": MAX_STEPS, "max_tokens": MAX_TOKENS, "sampling": "Ollama defaults, seed = run seed",
                "frozen_sha256": frozen, "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
with open(out.replace(".jsonl", "-setup.json"), "w") as f:
    f.write(json.dumps(setup, indent=1) + "\n")
print(json.dumps(setup, indent=1))

total = len(args.systems) * len(questions) * len(args.seeds)
reset()
warm_up()
started = time.monotonic()
with open(out, "w") as f:
    done = 0
    for system in args.systems:
        for q in questions:
            for seed in args.seeds:
                before = other_models_loaded()  # checked outside the timed section
                t0 = time.monotonic()
                record = SYSTEMS[system](system_prompt, q["question"], seed)
                seconds = round(time.monotonic() - t0, 3)
                after = other_models_loaded()
                row = {"model": setup["model"], "digest": setup["digest"], "ollama": setup["ollama"],
                       "system": system, "question": q["id"], "seed": seed, **record,
                       "reply_sha256": hashlib.sha256((record["final"] or "").encode()).hexdigest(),
                       "seconds": seconds, "other_models_loaded": sorted(set(before) | set(after))}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                done += 1
                print(f"{done:>4}/{total}  {system}  {q['id']}  seed {seed}  route={record['route']}  "
                      f"{record['outcome']}  {len(record['model_calls'])} model call(s)  {seconds:.1f} s")
reset()
print(f"\n{total} conversations in {time.monotonic() - started:.0f} s, saved to {out}. Model unloaded.")
