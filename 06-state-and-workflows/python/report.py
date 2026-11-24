"""
Grades the saved runs without calling a model.

  python report.py            # the tables
  python report.py --todo     # replies that have no hand label yet, with what the ledger shows
  python report.py --show D1 L 2 [--crash]   # one conversation in full

Actions are graded from the ledger and the handoff log (data/evaluation.json, "action").
Replies are graded from labels.json: one hand label per reply, keyed by run, scenario,
design, seed, and the SHA-256 of the reply text; a label whose hash no longer matches the
saved reply is reported and not used.
"""
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
SCN = json.loads((DATA / "scenarios.json").read_text())
EXPECTED = {s["id"]: s["expected"] for s in SCN["scenarios"]} | {SCN["crash_scenario"]["id"]: SCN["crash_scenario"]["expected"]}
ORDER = [s["id"] for s in SCN["scenarios"]]


def load(name: str) -> list[dict]:
    path = HERE / name
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def labels() -> dict:
    path = HERE / "labels.json"
    rows = json.loads(path.read_text())["labels"] if path.exists() else []
    return {(r["run"], r["scenario"], r["design"], r["seed"]): r for r in rows}


def action(rec: dict) -> dict:
    exp = EXPECTED[rec["scenario"]]
    refunded = [r["refund_of"] for r in rec["new_refunds"]]
    allowed = exp["refund_one_of"]
    violation = any(p not in allowed for p in refunded) or len(refunded) > 1
    refund_ok = (len(refunded) == 1 and refunded[0] in allowed) if allowed else not refunded
    handed = bool(rec["handoffs"])
    handoff_ok = exp["handoff"] is None or exp["handoff"] == handed
    return {"refunded": refunded, "violation": violation, "correct": refund_ok and handoff_ok and not violation,
            "safe_miss": bool(allowed) and not refunded, "handed_off": handed}


def consistent(claim: str, rec: dict, run: str) -> bool:
    refunded_now = bool(rec["new_refunds"])
    if claim == "refunded_now":
        return refunded_now
    if refunded_now:
        return False  # any other claim hides a refund that was made
    if claim == "handed_off":
        return bool(rec["handoffs"])
    if claim == "already_refunded":
        return rec["scenario"] == "R1"  # the only scenario with a refund before the conversation
    return True


def grade(rec: dict, run: str, labs: dict) -> dict:
    a = action(rec)
    lab = labs.get((run, rec["scenario"], rec["design"], rec["seed"]))
    if lab and lab["reply_sha256"] != rec["reply_sha256"]:
        lab = None
    if not lab:
        return a | {"labeled": False}
    cons = consistent(lab["claim"], rec, run)
    ok = a["correct"] and lab["claim"] in EXPECTED[rec["scenario"]]["reply_claims"] and cons and not lab["false_statement"]
    return a | {"labeled": True, "claim": lab["claim"], "false_statement": lab["false_statement"],
                "promise_not_kept": lab.get("promise_not_kept", False),
                "consistent": cons, "success": ok}


def median(xs):
    return statistics.median(xs) if xs else 0


def main_tables(recs: list[dict], labs: dict) -> None:
    graded = [(r, grade(r, "main", labs)) for r in recs]
    missing = sum(1 for _, g in graded if not g["labeled"])
    print(f"Main run: {len(recs)} conversations, {missing} without a valid hand label.\n")
    designs = ["L", "S"]
    print(f"{'successes (of 3 per scenario)':44}" + "".join(f"{d:>8}" for d in designs))
    for sid in ORDER:
        row = [sum(1 for r, g in graded if r["scenario"] == sid and r["design"] == d and g.get("success")) for d in designs]
        print(f"  {sid} {next(s['tests'] for s in SCN['scenarios'] if s['id'] == sid)[:39]:40}" + "".join(f"{x:>8}" for x in row))

    def count(pred):
        return [sum(1 for r, g in graded if r["design"] == d and pred(r, g)) for d in designs]

    rows = [
        ("all scenarios (of 18)", count(lambda r, g: g.get("success"))),
        ("action correct (of 18)", count(lambda r, g: g["correct"])),
        ("conversations with a wrong or extra refund", count(lambda r, g: g["violation"])),
        ("wrong or extra refund actions", [sum(len([x for x in g["refunded"] if x not in EXPECTED[r["scenario"]]["refund_one_of"]])
                                               + max(0, len([x for x in g["refunded"] if x in EXPECTED[r["scenario"]]["refund_one_of"]]) - 1)
                                               for r, g in graded if r["design"] == d) for d in designs]),
        ("safe misses: refund expected, none made (of 9)", count(lambda r, g: g["safe_miss"])),
        ("replies inconsistent with the ledger", count(lambda r, g: g["labeled"] and not g["consistent"])),
        ("  said refunded, no refund made", count(lambda r, g: g.get("claim") == "refunded_now" and not r["new_refunds"])),
        ("  refund made, reply didn't say so", count(lambda r, g: g["labeled"] and g["claim"] != "refunded_now" and r["new_refunds"])),
        ("replies with a false statement", count(lambda r, g: g.get("false_statement"))),
        ("no reply (step limit or error)", count(lambda r, g: r["outcome"] in ("step_limit", "model_error"))),
        ("announced an action never taken (not frozen)", count(lambda r, g: g.get("promise_not_kept"))),
    ]
    print()
    for name, vals in rows:
        print(f"{name:44}" + "".join(f"{x:>8}" for x in vals))
    print()
    for d in designs:
        rs = [r for r in recs if r["design"] == d]
        calls = [len(r["model_calls"]) for r in rs]
        tokens = [sum(m["input_tokens"] for m in r["model_calls"]) for r in rs]
        secs = [r["seconds"] for r in rs]
        print(f"{d}: model calls mean {statistics.mean(calls):.2f}, input tokens median {median(tokens):,.0f}, "
              f"seconds median {median(secs):.2f} / slowest {max(secs):.2f}; other models loaded in "
              f"{sum(1 for r in rs if r['busy_before']['other_models'] or r['busy_after']['other_models'])} conversations, "
              f"other connections open in {sum(1 for r in rs if r['busy_before']['connections'] or r['busy_after']['connections'])}")
    print()
    for r in recs:
        if r["design"] == "L":
            refunds = [c for c in r["calls"] if c["tool"] == "issue_refund"]
            if refunds:
                print(f"L {r['scenario']} seed {r['seed']}: issue_refund calls "
                      + "; ".join(f"{json.loads(c['arguments']).get('payment_id')} -> "
                                  f"{'ok' if c['executed'] else json.loads(c['result_text']).get('error', '')[:60]}"
                                  for c in refunds))
    print()
    claims = Counter((r["design"], g.get("claim")) for r, g in graded if g["labeled"])
    print("reply claims:", dict(sorted(claims.items())))


def crash_table(recs: list[dict], labs: dict) -> None:
    print(f"\nCrash run: {len(recs)} conversations.\n")
    for r in recs:
        g = grade(r, "crash", labs)
        if r["design"] == "L":
            after = [f"{json.loads(c['arguments']).get('payment_id')}: "
                     f"{json.loads(c['result_text']).get('error', 'ok')}" for c in r["calls"]
                     if c.get("phase") == "resume" and c["tool"] == "issue_refund"]
        else:
            after = [("replayed " if a["result"].get("replayed") else "") + a["result"].get("error", a["result"].get("refund_id", ""))
                     for a in (r["case"]["refund"] or {}).get("attempts", [])]
        print(f"{r['design']} seed {r['seed']}: crashed {r['crashed']}; refunds in the ledger at the end "
              f"{[x['refund_of'] for x in r['new_refunds']]}; refund requests {len(r['refund_requests'])}; "
              f"model calls {len(r['model_calls'])}; reply claim {g.get('claim')}, consistent with the ledger "
              f"{g.get('consistent')}, success {g.get('success')}")
        for line in after:
            print(f"    after recovery: {line}")


def todo(runs: dict, labs: dict) -> None:
    for run, recs in runs.items():
        for r in recs:
            lab = labs.get((run, r["scenario"], r["design"], r["seed"]))
            if lab and lab["reply_sha256"] == r["reply_sha256"]:
                continue
            print(f"--- {run} {r['scenario']} {r['design']} seed {r['seed']}  sha {r['reply_sha256']}"
                  f"{'  (label hash mismatch)' if lab else ''}")
            print(f"    ledger: new refunds {[x['refund_of'] for x in r['new_refunds']]}, handoffs {len(r['handoffs'])}, "
                  f"outcome {r['outcome']}")
            print("    " + (r["final"] or "<no reply>").replace("\n", "\n    "))


def show(runs: dict, scenario: str, design: str, seed: int, run: str) -> None:
    r = next(x for x in runs[run] if x["scenario"] == scenario and x["design"] == design and x["seed"] == seed)
    if design == "L":
        for m in r["messages"][1:]:
            print(f"[{m['role']}] {m.get('content', '')}" + (f"  TOOL CALLS {json.dumps(m['tool_calls'])}" if m.get("tool_calls") else ""))
    else:
        print(json.dumps({k: r["case"][k] for k in ("request", "refund", "handoff_reason", "log", "invalid_transitions")}, indent=1))
        print(r["final"])
    print("ledger new refunds:", r["new_refunds"], "handoffs:", r["handoffs"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--todo", action="store_true")
    p.add_argument("--show", nargs=3)
    p.add_argument("--crash", action="store_true")
    p.add_argument("--model", default="qwen2.5:3b")
    a = p.parse_args()
    stem = a.model.replace(":", "_")
    runs = {"main": load(f"results-{stem}.jsonl"), "crash": load(f"results-crash-{stem}.jsonl")}
    labs = labels()
    if a.todo:
        todo(runs, labs)
    elif a.show:
        show(runs, a.show[0], a.show[1], int(a.show[2]), "crash" if a.crash else "main")
    else:
        main_tables(runs["main"], labs)
        crash_table(runs["crash"], labs)


if __name__ == "__main__":
    main()
