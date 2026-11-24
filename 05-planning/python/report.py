"""
Grades saved conversations with the frozen rules in data/evaluation.json and prints the comparison.
Calls no model.

  python report.py results-qwen2.5_3b.jsonl
  python report.py results-qwen2.5_3b.jsonl --todo     # replies that still need a hand label

Hand labels live in labels.json: "<task>|<SHA-256 of the reply>" -> {label, fabricated_action, note}.
"""
import argparse
import json
import re
from pathlib import Path
from statistics import mean, median

from tools import BY_NAME, DATA

HERE = Path(__file__).resolve().parent
TASKS = {t["id"]: t for t in json.loads((DATA / "tasks.json").read_text())["tasks"]}
DESIGNS = ["W", "L", "P", "R"]
OUTCOMES = ["success", "correct_unsupported", "wrong_with_evidence", "wrong_without_evidence"]
ID = re.compile(r"\b(?:INV|PAY)-\d+\b")


def labels() -> dict:
    path = HERE / "labels.json"
    return json.loads(path.read_text())["answers"] if path.exists() else {}


LABELS = labels()


def answer(r: dict) -> dict | None:
    return LABELS.get(f"{r['task']}|{r['reply_sha256']}")


def satisfied(task: dict, calls: list[dict]) -> bool:
    options = task["record_evidence"]
    if not options:
        return True
    seen = "\n".join(c["result_text"] for c in calls if c["executed"] and c["tool"] in BY_NAME)
    return any(all(s in seen for s in option) for option in options)


def evidence(r: dict) -> bool:
    return satisfied(TASKS[r["task"]], r["calls"])


def first_plan_evidence(r: dict) -> bool:
    return satisfied(TASKS[r["task"]], [c for c in r["calls"] if c["source"] == "plan 1"])


def outcome(r: dict) -> str | None:
    a = answer(r)
    if a is None:
        return None
    ev = evidence(r)
    if a["label"] == "correct":
        return "success" if ev else "correct_unsupported"
    return "wrong_with_evidence" if ev else "wrong_without_evidence"


def _args(text: str):
    try:
        return json.loads(text)
    except ValueError:
        return text


def call_flags(r: dict) -> list[dict]:
    """Per call: repeated, outside the useful tools, failed, and ids used before they could have been seen."""
    task = TASKS[r["task"]]
    flags, earlier = [], []
    for i, c in enumerate(r["calls"]):
        before = [p for p in r["calls"][:i] if p["source"] != c["source"]]  # a batch can't see its own results
        seen = task["question"] + "\n" + "\n".join(p["result_text"] for p in before)
        ids = set(ID.findall(c["arguments"]))
        flags.append({
            "repeated": any(p["tool"] == c["tool"] and _args(p["arguments"]) == _args(c["arguments"]) for p in earlier),
            "outside_useful": c["tool"] not in task["useful_tools"],
            "failed": not c["executed"],
            "unknown_tool": c["tool"] not in BY_NAME,
            "invalid_arguments": c["tool"] in BY_NAME and not c["valid"],
            "injected": c.get("injected_failure", False),
            "unseen_ids": sorted(i for i in ids if i not in seen)})
        earlier.append(c)
    return flags


def pct(values: list, q: float) -> float:
    return sorted(values)[min(len(values) - 1, int(q * len(values)))] if values else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("--todo", action="store_true")
    a = parser.parse_args()
    rows = [json.loads(line) for line in (HERE / a.results).read_text().splitlines() if line.strip()]

    if a.todo:
        missing = {}
        for r in rows:
            if answer(r) is None:
                missing.setdefault(f"{r['task']}|{r['reply_sha256']}", []).append(f"{r['design']}{r['seed']}")
        for key, where in missing.items():
            r = next(x for x in rows if f"{x['task']}|{x['reply_sha256']}" == key)
            print(f"=== {key}  ({', '.join(where)})\n{r['final']}\n")
        print(f"{len(missing)} distinct replies need a label.")
        return

    designs = [d for d in DESIGNS if any(r["design"] == d for r in rows)]
    unlabeled = sum(answer(r) is None for r in rows)
    print(f"{len(rows)} conversations; {unlabeled} without a hand label (shown as ?).")
    others = sorted({m for r in rows for m in r["other_models_loaded"]})
    print(f"other models loaded during any conversation: {others or 'none'}\n")

    # 1. Answers and outcomes per design.
    print("design   n  correct  incorrect  no answer  fabricated  success  unsupp.  wrong+ev  wrong-ev  evidence")
    for d in designs:
        rs = [r for r in rows if r["design"] == d]
        lab = [answer(r)["label"] if answer(r) else "?" for r in rs]
        outs = [outcome(r) for r in rs]
        fab = sum(bool(answer(r) and answer(r)["fabricated_action"]) for r in rs)
        print(f"{d:<6}{len(rs):>4}{lab.count('correct'):>9}{lab.count('incorrect'):>11}{lab.count('no_answer'):>11}"
              f"{fab:>12}" + "".join(f"{outs.count(o):>{w}}" for o, w in zip(OUTCOMES, (9, 9, 10, 10), strict=True))
              + f"{sum(evidence(r) for r in rs):>10}")

    # 2. Successes and evidence per task.
    print("\nsuccesses (evidence retrieved), per task, of " + str(len({r['seed'] for r in rows})) + " seeds")
    print("task  " + "".join(f"{d:>10}" for d in designs) + "   tests")
    for t in TASKS:
        rs = [r for r in rows if r["task"] == t]
        if not rs:
            continue
        cells = []
        for d in designs:
            ds = [r for r in rs if r["design"] == d]
            cells.append(f"{sum(outcome(r) == 'success' for r in ds)} ({sum(evidence(r) for r in ds)})")
        print(f"{t:<6}" + "".join(f"{c:>10}" for c in cells) + "   " + TASKS[t]["tests"].split(":")[0])

    # 2b. Post-hoc (labels.json, main_answer_right): not part of the frozen evaluation; changes no outcome.
    print("\npost-hoc, not frozen: replies whose main answer was right, and of those, how many also stated "
          "something false")
    print("task  " + "".join(f"{d:>10}" for d in designs))
    for t in [*TASKS, "all"]:
        rs = [r for r in rows if (t == "all" or r["task"] == t) and answer(r)]
        if not rs:
            continue
        cells = []
        for d in designs:
            ds = [r for r in rs if r["design"] == d and answer(r).get("main_answer_right")]
            cells.append(f"{len(ds)} ({sum(answer(r)['label'] != 'correct' for r in ds)})")
        print(f"{t:<6}" + "".join(f"{c:>10}" for c in cells))

    # 3. Plans (P and R) and the loop's tool requests.
    print("\nplans                         P        R")
    for name, fn in [
        ("first plans", lambda rs: len(rs)),
        ("first plan steps, mean", lambda rs: round(mean(len(r["plans"][0]["steps"]) for r in rs), 2) if rs else 0),
        ("empty first plans", lambda rs: sum(r["plans"][0]["parsed"] and not r["plans"][0]["steps"] for r in rs)),
        ("unparseable plans (any)", lambda rs: sum(not p["parsed"] for r in rs for p in r["plans"])),
        ("first plan got the evidence", lambda rs: sum(first_plan_evidence(r) for r in rs)),
        ("final evidence", lambda rs: sum(evidence(r) for r in rs)),
        ("replanning calls, total", lambda rs: sum(p["role"] == "replan" for r in rs for p in r["plans"])),
        ("replans that added steps", lambda rs: sum(p["role"] == "replan" and p["executed"] > 0
                                                    for r in rs for p in r["plans"])),
        ("replan_limit", lambda rs: sum(r["limit"] == "replan_limit" for r in rs)),
    ]:
        vals = [fn([r for r in rows if r["design"] == d and r["plans"]]) for d in ("P", "R")]
        print(f"{name:<28}" + "".join(f"{v:>9}" for v in vals))

    # 4. Calls.
    print("\ncalls                 " + "".join(f"{d:>8}" for d in designs))
    stats = {d: [call_flags(r) for r in rows if r["design"] == d] for d in designs}
    for name, key in [("unknown tool", "unknown_tool"), ("invalid arguments", "invalid_arguments"),
                      ("injected failure", "injected"), ("repeated", "repeated"),
                      ("outside useful tools", "outside_useful")]:
        print(f"{name:<22}" + "".join(f"{sum(f[key] for fl in stats[d] for f in fl):>8}" for d in designs))
    print(f"{'unseen id in args':<22}" + "".join(
        f"{sum(bool(f['unseen_ids']) for fl in stats[d] for f in fl):>8}" for d in designs))
    print(f"{'tool calls, total':<22}" + "".join(f"{sum(len(fl) for fl in stats[d]):>8}" for d in designs))
    print(f"{'step_limit (L)':<22}" + "".join(
        f"{sum(r['limit'] == 'step_limit' for r in rows if r['design'] == d):>8}" for d in designs))

    # 5. Q7: the injected failure.
    q7 = [r for r in rows if r["task"] == "Q7"]
    if q7:
        print("\nQ7 (get_refund_status unavailable)   " + "".join(f"{d:>6}" for d in designs))
        for name, fn in [
            ("conversations", lambda r: True),
            ("hit the failure", lambda r: any(c.get("injected_failure") for c in r["calls"])),
            ("called it again after failing", lambda r: sum(c.get("injected_failure") for c in r["calls"]) > 1),
            ("got invoice date + policy", evidence),
            ("success", lambda r: outcome(r) == "success"),
        ]:
            print(f"{name:<37}" + "".join(f"{sum(bool(fn(r)) for r in q7 if r['design'] == d):>6}" for d in designs))

    # 6. Cost.
    print("\ncost per conversation   model calls  tool calls  input tok med/max  largest call  seconds med/max"
          "  total input tok")
    for d in designs:
        rs = [r for r in rows if r["design"] == d]
        tok = [sum(m["input_tokens"] for m in r["model_calls"]) for r in rs]
        big = max(m["input_tokens"] for r in rs for m in r["model_calls"])
        secs = [r["seconds"] for r in rs]
        print(f"{d:<24}{mean(len(r['model_calls']) for r in rs):>11.2f}{mean(len(r['calls']) for r in rs):>12.2f}"
              f"{median(tok):>11,.0f} / {max(tok):<6,}{big:>12,}{median(secs):>9.2f} / {max(secs):<6.1f}"
              f"{sum(tok):>14,}")

    # 7. Context growth: input tokens by model-call position.
    print("\ninput tokens by model-call position (median; conversations reaching it)")
    for d in ("L", "R"):
        rs = [r for r in rows if r["design"] == d]
        if not rs:
            continue
        cells = []
        for pos in range(8):
            vals = [r["model_calls"][pos]["input_tokens"] for r in rs if len(r["model_calls"]) > pos]
            if vals:
                cells.append(f"{pos + 1}: {median(vals):,.0f} ({len(vals)})")
        print(f"{d}  " + "   ".join(cells))


if __name__ == "__main__":
    main()
