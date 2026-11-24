"""
Runs the experiments and saves everything.

  python run.py all      # the main run, then the crash run (one Ollama session)
  python run.py main     # 6 scenarios x designs L and S x seeds 1-3
  python run.py crash    # C1 x designs L and S x seeds 1-3, each in two processes

Before anything runs, data/ is checked against data/FROZEN.json. The run then takes the
shared Ollama lock (llm.ollama_lock), waits until Ollama has been idle for 60 s, unloads
the model, loads it with one throwaway request, and unloads it at the end. Every
conversation gets a fresh ledger in a temporary directory, removed afterwards; what it
contained is saved in the results. Conversations run in a fixed order: scenario, design,
seed. "seconds" is end-to-end wall-clock time.

The crash run starts each conversation in a child process. The payment service ends that
process (exit code 75) right after the first refund is written to the ledger. A second
child process then recovers from what is on disk: the ledger and the design's own saved
state (L: the message list; S: the case record).
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
SEEDS = (1, 2, 3)
DESIGNS = ("L", "S")
CODE_FILES = ("llm.py", "payments.py", "tools.py", "agent_loop.py", "workflow.py", "run.py")


def sha(text: str | None) -> str:
    return hashlib.sha256((text or "").encode()).hexdigest()


def check_frozen() -> dict:
    frozen = json.loads((DATA / "FROZEN.json").read_text())["files"]
    for name, expected in frozen.items():
        actual = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"data/{name} changed since it was frozen (sha256 {actual}, expected {expected}).")
    return frozen


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def outcome_of(design: str, run_dir: Path) -> dict:
    """What the design's own saved state says, plus the ledger, read from disk."""
    from payments import new_refunds
    ledger = json.loads((run_dir / "ledger.json").read_text())
    common = {"new_refunds": new_refunds(ledger), "handoffs": ledger["handoffs"],
              "refund_requests": ledger["requests"]}
    if design == "L":
        state = json.loads((run_dir / "loop_state.json").read_text())
        return common | {"final": state["final"], "outcome": state["outcome"], "calls": state["calls"],
                         "model_calls": state["model_calls"], "messages": state["messages"]}
    case = json.loads((run_dir / "case.json").read_text())
    return common | {"final": case["reply"], "outcome": "done" if case["node"] == "done" else case["node"],
                     "calls": [], "model_calls": case["model_calls"], "case": case}


def run_design(design: str, run_dir: Path, scenario: dict, seed: int, resume: bool) -> None:
    import agent_loop
    import workflow
    if design == "L":
        agent_loop.run(run_dir, scenario["message"], seed, resume=resume)
    else:
        workflow.run(run_dir, f"{scenario['id']}-S-s{seed}", scenario["message"], seed, resume=resume)


def busy_signs() -> dict:
    """Between our own requests: other loaded models, and open connections to Ollama (ours are all closed)."""
    from llm import open_connections, other_models_loaded
    return {"other_models": other_models_loaded(), "connections": open_connections()}


def main_run(scenarios: list[dict], out: Path) -> None:
    from payments import PaymentService
    with out.open("w") as f:
        for scenario in scenarios:
            for design in DESIGNS:
                for seed in SEEDS:
                    run_dir = Path(tempfile.mkdtemp(prefix=f"sw-{scenario['id']}-{design}-{seed}-"))
                    PaymentService.create(run_dir / "ledger.json", scenario["prior_refunds"], scenario["fault"])
                    before = busy_signs()
                    started = time.monotonic()
                    run_design(design, run_dir, scenario, seed, resume=False)
                    seconds = round(time.monotonic() - started, 3)
                    result = outcome_of(design, run_dir)
                    after = busy_signs()
                    shutil.rmtree(run_dir)
                    record = {"scenario": scenario["id"], "design": design, "seed": seed, "seconds": seconds,
                              "reply_sha256": sha(result["final"]), "busy_before": before, "busy_after": after,
                              **result}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"{scenario['id']} {design} seed {seed}: {record['outcome']}, "
                          f"refunds {[r['refund_of'] for r in record['new_refunds']]}, "
                          f"handoffs {len(record['handoffs'])}, {seconds} s", flush=True)


def crash_run(scenario: dict, out: Path) -> None:
    from payments import CRASH_EXIT_CODE, PaymentService
    with out.open("w") as f:
        for design in DESIGNS:
            for seed in SEEDS:
                run_dir = Path(tempfile.mkdtemp(prefix=f"sw-{scenario['id']}-{design}-{seed}-"))
                PaymentService.create(run_dir / "ledger.json", scenario["prior_refunds"], scenario["fault"])
                before = busy_signs()
                child = [sys.executable, str(HERE / "run.py"), "_child", design, str(run_dir), str(seed)]
                first = subprocess.run(child + ["first"], cwd=HERE, env=os.environ.copy())
                phases = {"first_exit_code": first.returncode}
                after_first = outcome_of(design, run_dir)
                phases["after_first"] = {"new_refunds": after_first["new_refunds"],
                                         "model_calls": len(after_first["model_calls"])}
                if first.returncode == CRASH_EXIT_CODE:
                    second = subprocess.run(child + ["resume"], cwd=HERE, env=os.environ.copy())
                    phases["resume_exit_code"] = second.returncode
                    crashed = True
                else:
                    crashed = False
                result = outcome_of(design, run_dir)
                after = busy_signs()
                shutil.rmtree(run_dir)
                record = {"scenario": scenario["id"], "design": design, "seed": seed, "crashed": crashed, **phases,
                          "reply_sha256": sha(result["final"]), "busy_before": before, "busy_after": after, **result}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                print(f"{scenario['id']} {design} seed {seed}: crashed={crashed}, {record['outcome']}, refunds "
                      f"{[r['refund_of'] for r in record['new_refunds']]}", flush=True)


def child(design: str, run_dir: str, seed: str, phase: str) -> None:
    scenario = json.loads((DATA / "scenarios.json").read_text())["crash_scenario"]
    run_design(design, Path(run_dir), scenario, int(seed), resume=(phase == "resume"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["all", "main", "crash", "_child"])
    parser.add_argument("rest", nargs="*")
    parser.add_argument("--model", default="qwen2.5:3b")
    args = parser.parse_args()
    os.environ["OLLAMA_CHAT_MODEL"] = args.model
    frozen = check_frozen()
    if args.command == "_child":
        child(*args.rest)
        return

    from llm import ollama_lock, reset, runtime, warm_up
    scenarios = json.loads((DATA / "scenarios.json").read_text())
    stem = args.model.replace(":", "_").replace("/", "_")
    with ollama_lock(f"ai-agent-patterns/06-state-and-workflows run.py {args.command}"):
        info = runtime()
        template = info.pop("template")
        (HERE / f"template-{stem}.txt").write_text(template)
        power = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout.strip().splitlines()
        setup = info | {"template_sha256": sha(template), "command": args.command,
                        "sampling": "Ollama defaults, seed = run seed for every model call",
                        "frozen_sha256": frozen,
                        "code_sha256": {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in CODE_FILES},
                        "python": platform.python_version(), "machine": platform.machine(),
                        "power": power[0] if power else None, "started_utc": utc()}
        reset()
        warm_up()
        if args.command in ("all", "main"):
            main_run(scenarios["scenarios"], HERE / f"results-{stem}.jsonl")
        if args.command in ("all", "crash"):
            crash_run(scenarios["crash_scenario"], HERE / f"results-crash-{stem}.jsonl")
        reset()
        setup["finished_utc"] = utc()
        (HERE / f"results-{stem}-setup.json").write_text(json.dumps(setup, indent=2) + "\n")


if __name__ == "__main__":
    main()
