"""
Creates each design's memory, asks every question, and saves everything.

  python run.py                                    # qwen2.5:3b, designs N T S K, seeds 1-3
  python run.py --seeds 1 --questions A1 B1 E1 --out results-check.jsonl   # a small check run

Before anything runs, data/ is checked against the hashes in data/FROZEN.json. The model is
unloaded, loaded with one throwaway request (so nothing includes load time), and unloaded at
the end. First, every design's memory is created for every seed (S and K make model calls;
N and T don't) and saved to memory-<...>.json. Then the conversations run in a fixed order:
design, question, seed. "seconds" is end-to-end wall-clock time per conversation.
"""
import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="qwen2.5:3b")
parser.add_argument("--designs", nargs="+", default=["N", "T", "S", "K"])
parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
parser.add_argument("--questions", nargs="+")
parser.add_argument("--out")
args = parser.parse_args()
os.environ["OLLAMA_CHAT_MODEL"] = args.model

from llm import other_models_loaded, reset, runtime, warm_up  # noqa: E402
from systems import MAX_STEPS, MAX_TOKENS, conversation, create_memory  # noqa: E402
from tools import DATA  # noqa: E402

frozen = json.loads((DATA / "FROZEN.json").read_text())["sha256"]
for name, expected in frozen.items():
    actual = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"data/{name} changed since it was frozen (sha256 {actual}, expected {expected}).")

questions = json.loads((DATA / "questions.json").read_text())["questions"]
if args.questions:
    questions = [q for q in questions if q["id"] in args.questions]
stem = args.model.replace(":", "_").replace("/", "_")
out = args.out or f"results-{stem}.jsonl"


def sha(text: str | None) -> str:
    return hashlib.sha256((text or "").encode()).hexdigest()


info = runtime()
template = info.pop("template")
with open(f"template-{stem}.txt", "w") as f:
    f.write(template)
setup = info | {"template_sha256": sha(template), "max_model_calls_per_conversation": MAX_STEPS,
                "max_tokens": MAX_TOKENS, "sampling": "Ollama defaults, seed = run seed for every call",
                "frozen_sha256": frozen,
                "code_sha256": {n: hashlib.sha256((DATA.parent / "python" / n).read_bytes()).hexdigest()
                                for n in ("systems.py", "tools.py", "llm.py", "run.py")},
                "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
with open(out.replace(".jsonl", "-setup.json"), "w") as f:
    f.write(json.dumps(setup, indent=1) + "\n")
print(json.dumps(setup, indent=1))

reset()
warm_up()
started = time.monotonic()

# 1. Memory, per seed and design, created once and used by every conversation of that seed.
memories = {}
for seed in args.seeds:
    for design in args.designs:
        before = other_models_loaded()
        t0 = time.monotonic()
        memory = create_memory(design, seed)
        memory["seconds"] = round(time.monotonic() - t0, 3)
        memory["other_models_loaded"] = sorted(set(before) | set(other_models_loaded()))
        memory["block_sha256"] = sha(memory["block"])
        memories[(design, seed)] = memory
        print(f"memory {design} seed {seed}: {len(memory['calls'])} creation call(s), "
              f"{len(memory['block'] or '')} characters, {memory['seconds']:.1f} s")
with open(out.replace(".jsonl", "-memory.json"), "w") as f:
    f.write(json.dumps([memories[k] for k in sorted(memories)], ensure_ascii=False, indent=1) + "\n")

# 2. The new conversations.
total = len(args.designs) * len(questions) * len(args.seeds)
with open(out, "w") as f:
    done = 0
    for design in args.designs:
        for q in questions:
            for seed in args.seeds:
                memory = memories[(design, seed)]
                before = other_models_loaded()  # checked outside the timed section
                t0 = time.monotonic()
                record = conversation(memory["block"], q["question"], seed)
                seconds = round(time.monotonic() - t0, 3)
                after = other_models_loaded()
                row = {"model": setup["model"], "digest": setup["digest"], "ollama": setup["ollama"],
                       "design": design, "question": q["id"], "seed": seed,
                       "memory_sha256": memory["block_sha256"], **record,
                       "reply_sha256": sha(record["final"]), "seconds": seconds,
                       "other_models_loaded": sorted(set(before) | set(after))}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                done += 1
                tools_used = [c["tool"] for c in record["calls"]]
                print(f"{done:>4}/{total}  {design}  {q['id']}  seed {seed}  {record['outcome']}  "
                      f"{len(record['model_calls'])} call(s) {tools_used}  {seconds:.1f} s")
reset()
print(f"\n{total} conversations in {time.monotonic() - started:.0f} s (including memory creation), "
      f"saved to {out}. Model unloaded.")
