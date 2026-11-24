"""
The measured run: every scenario, every seed, and every paused conversation continued five ways.

  python run.py                                  # qwen2.5:3b, scenarios S1-S8, seeds 1-3
  python run.py --seeds 1 --scenarios S1 S6 --out results-check.jsonl   # a small check run

Before anything runs, data/ is checked against data/FROZEN.json. The model is unloaded, loaded
with one throwaway request, and unloaded at the end.

Phase 1: the agent gets the customer's message and runs until it replies or pauses on a refund.
Phase 2: a paused conversation's database (its checkpoint) is copied once per branch, and each
copy is continued with one outcome: 'pending' (the agent replies while the request waits),
then the scripted decisions from data/decisions.json: approve, modify, reject, expire. The
branches share the same checkpoint and seed, so they differ only in the outcome.
Times come from data/decisions.json (simulated), so expiry behaves the same on every run.
"""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="qwen2.5:3b")
parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
parser.add_argument("--scenarios", nargs="+")
parser.add_argument("--out")
args = parser.parse_args()
os.environ["OLLAMA_CHAT_MODEL"] = args.model

from agent import MAX_STEPS, MAX_TOKENS, RESULTS, VERSION, Agent, _fill  # noqa: E402
from llm import other_models_loaded, reset, runtime, warm_up  # noqa: E402
from policy import half_rounded_down  # noqa: E402
from store import Store  # noqa: E402
from tools import DATA  # noqa: E402

frozen = json.loads((DATA / "FROZEN.json").read_text())["sha256"]
for name, expected in frozen.items():
    actual = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"data/{name} changed since it was frozen (sha256 {actual}, expected {expected}).")

SCENARIOS = json.loads((DATA / "scenarios.json").read_text())["scenarios"]
if args.scenarios:
    SCENARIOS = [s for s in SCENARIOS if s["id"] in args.scenarios]
SCRIPT = json.loads((DATA / "decisions.json").read_text())
CLOCK, REVIEWERS, DECISIONS = SCRIPT["clock"], SCRIPT["reviewers"], SCRIPT["measured_decisions"]
BRANCHES = ["pending", "approve", "modify", "reject", "expire"]
stem = args.model.replace(":", "_").replace("/", "_")
out = args.out or f"results-{stem}.jsonl"


def sha(text: str | None) -> str:
    return hashlib.sha256((text or "").encode()).hexdigest()


info = runtime()
template = info.pop("template")
with open(f"template-{stem}.txt", "w") as f:
    f.write(template)
setup = info | {"template_sha256": sha(template), "max_model_calls_per_segment": MAX_STEPS, "max_tokens": MAX_TOKENS,
                "sampling": "Ollama defaults, seed = run seed for every call", "checkpoint_version": VERSION,
                "frozen_sha256": frozen,
                "code_sha256": {n: hashlib.sha256((DATA.parent / "python" / n).read_bytes()).hexdigest()
                                for n in ("agent.py", "policy.py", "store.py", "tools.py", "llm.py", "run.py")},
                "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
with open(out.replace(".jsonl", "-setup.json"), "w") as f:
    f.write(json.dumps(setup, indent=1) + "\n")
print(json.dumps(setup, indent=1))


def branch(base: str, request_id: str, kind: str, seed: int, workdir: str) -> dict:
    path = os.path.join(workdir, f"{kind}.sqlite")
    shutil.copyfile(base, path)
    store = Store(path)
    try:
        agent = Agent(store, seed=seed)
        req = store.get(request_id)
        reviewer = REVIEWERS[req["tier"]]
        decision = None
        if kind == "pending":
            result = _fill(RESULTS["pending"], request_id=request_id, invoice_id=req["action"]["invoice_id"],
                           requested_amount=req["action"]["amount"])
            record = agent.continue_with(req, result, CLOCK["request_created_utc"]) | {"tool_result": result}
        else:
            if kind == "approve":
                decision = agent.decide(request_id, "approve", reviewer["reviewer"], reviewer["role"],
                                        req["action_sha256"], CLOCK["decided_utc"])
            elif kind == "modify":
                decision = agent.decide(request_id, "modify", reviewer["reviewer"], reviewer["role"],
                                        req["action_sha256"], CLOCK["decided_utc"],
                                        amount=half_rounded_down(req["action"]["amount"]),
                                        reason=DECISIONS["modify"]["note"])
            elif kind == "reject":
                decision = agent.decide(request_id, "reject", reviewer["reviewer"], reviewer["role"],
                                        req["action_sha256"], CLOCK["decided_utc"],
                                        reason=DECISIONS["reject"]["reason"][req["tier"]])
            when = CLOCK["expiry_checked_utc"] if kind == "expire" else CLOCK["decided_utc"]
            if decision is not None and not decision["ok"]:
                record = {"outcome": "decision_refused", "error": decision["error"], "final": None,
                          "model_calls": [], "calls": []}
            else:
                record = agent.resume(request_id, when)
        record["refunds_after"] = store.issued()
        record["events"] = [(e["actor"], e["type"]) for e in store.events()]
        return record | {"decision": decision}
    finally:
        store.close()


reset()
warm_up()
started = time.monotonic()
total = len(SCENARIOS) * len(args.seeds)
done = 0
with open(out, "w") as f:
    for scenario in SCENARIOS:
        for seed in args.seeds:
            with tempfile.TemporaryDirectory() as workdir:
                base = os.path.join(workdir, "base.sqlite")
                store = Store(base)
                conversation_id = f"{scenario['id']}-seed{seed}"
                before = other_models_loaded()
                t0 = time.monotonic()
                record = Agent(store, seed=seed).start(conversation_id, scenario["message"],
                                                       CLOCK["request_created_utc"])
                seconds = round(time.monotonic() - t0, 3)
                view = store.get(record["request_id"])["view"] if record["request_id"] else None
                store.close()
                row = {"phase": "proposal", "model": setup["model"], "digest": setup["digest"],
                       "ollama": setup["ollama"], "scenario": scenario["id"], "seed": seed, **record,
                       "approval_request": view, "reply_sha256": sha(record["final"]), "seconds": seconds,
                       "other_models_loaded": sorted(set(before) | set(other_models_loaded()))}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                done += 1
                print(f"{done:>3}/{total}  {scenario['id']}  seed {seed}  {record['outcome']}  "
                      f"{[c['tool'] for c in record['calls']]}  {seconds:.1f} s")
                if record["outcome"] != "paused":
                    continue
                for kind in BRANCHES:
                    before = other_models_loaded()
                    t0 = time.monotonic()
                    result = branch(base, record["request_id"], kind, seed, workdir)
                    seconds = round(time.monotonic() - t0, 3)
                    row = {"phase": "branch", "model": setup["model"], "scenario": scenario["id"], "seed": seed,
                           "branch": kind, "request_id": record["request_id"],
                           "proposed_action": view["action"], "tier": view["tier"], **result,
                           "reply_sha256": sha(result.get("final")), "seconds": seconds,
                           "other_models_loaded": sorted(set(before) | set(other_models_loaded()))}
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"       {kind:<8} {result['outcome']}  {result.get('tool_result', {}).get('status')}  "
                          f"{[c['tool'] for c in result['calls']]}  {seconds:.1f} s")
reset()
print(f"\n{total} conversations (plus their branches) in {time.monotonic() - started:.0f} s, saved to {out}. "
      "Model unloaded.")
