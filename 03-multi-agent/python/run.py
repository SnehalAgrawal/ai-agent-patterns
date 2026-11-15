"""
Asks every question of every system and saves each conversation.

  python run.py                                         # qwen2.5:3b, systems S M R, seeds 1-3
  python run.py --systems S M R --seeds 1 --questions K1 O1 --out results-check.jsonl   # a small check run

Before anything runs, data/ is checked against the hashes in data/FROZEN.json. The model
is unloaded before the run, loaded with one throwaway request (so no conversation includes
load time), and unloaded after. Conversations run in a fixed order: system, question, seed.
"seconds" is end-to-end wall-clock time for the whole conversation (every agent, every tool).
"""
import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="qwen2.5:3b")
parser.add_argument("--systems", nargs="+", default=["S", "M", "R"])
parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
parser.add_argument("--questions", nargs="+")
parser.add_argument("--out")
args = parser.parse_args()
os.environ["OLLAMA_CHAT_MODEL"] = args.model

from llm import other_models_loaded, reset, runtime, warm_up  # noqa: E402
from systems import MAX_STEPS, MAX_TOKENS, SYSTEMS  # noqa: E402
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

info = runtime()
template = info.pop("template")
with open(f"template-{stem}.txt", "w") as f:
    f.write(template)
setup = info | {"template_sha256": hashlib.sha256(template.encode()).hexdigest(),
                "max_steps_per_agent_loop": MAX_STEPS, "max_tokens": MAX_TOKENS,
                "sampling": "Ollama defaults, seed = run seed for every call in the conversation",
                "frozen_sha256": frozen,
                "code_sha256": {n: hashlib.sha256((DATA.parent / "python" / n).read_bytes()).hexdigest()
                                for n in ("systems.py", "tools.py", "llm.py", "run.py")},
                "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
with open(out.replace(".jsonl", "-setup.json"), "w") as f:
    f.write(json.dumps(setup, indent=1) + "\n")
print(json.dumps(setup, indent=1))


def sha(text: str | None) -> str:
    return hashlib.sha256((text or "").encode()).hexdigest()


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
                record = SYSTEMS[system](q["question"], seed)
                seconds = round(time.monotonic() - t0, 3)
                after = other_models_loaded()
                for h in record["handoffs"]:
                    h["request_sha256"], h["reply_sha256"] = sha(h["request"]), sha(h["reply"])
                row = {"model": setup["model"], "digest": setup["digest"], "ollama": setup["ollama"],
                       "system": system, "question": q["id"], "seed": seed, **record,
                       "reply_sha256": sha(record["final"]), "draft_sha256": sha(record["draft"]),
                       "seconds": seconds, "other_models_loaded": sorted(set(before) | set(after))}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                done += 1
                roles = {}
                for c in record["model_calls"]:
                    roles[c["role"]] = roles.get(c["role"], 0) + 1
                extra = f"  action={record['action']}" if system == "R" else ""
                print(f"{done:>4}/{total}  {system}  {q['id']}  seed {seed}  {record['outcome']}{extra}  "
                      f"calls {roles}  {seconds:.1f} s")
reset()
print(f"\n{total} conversations in {time.monotonic() - started:.0f} s, saved to {out}. Model unloaded.")
