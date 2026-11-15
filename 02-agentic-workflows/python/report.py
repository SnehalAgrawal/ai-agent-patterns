"""
Grades saved conversations and prints the comparison. Calls no model.

  python report.py results-qwen2.5_3b.jsonl
  python report.py results-qwen2.5_3b.jsonl --unlabeled    # replies that still need a hand label

The rules are the ones frozen in data/questions.json:
  route        W and H: the router's intent against the question's frozen intent
  handoff      W only: the router chose "other"; reported on its own, never labeled
  facts        every fact of the question appears in the text of a tool result that executed
  label        correct / incorrect / no_answer, by hand, in labels.json, with the reply's SHA-256
  success      correct AND facts
"""
import argparse
import json
from pathlib import Path
from statistics import median

from tools import DATA

HERE = Path(__file__).resolve().parent
QUESTIONS = {q["id"]: q for q in json.loads((DATA / "questions.json").read_text())["questions"]}


def key(r: dict) -> str:
    return f"{r['model']}|{r['system']}|{r['question']}|{r['seed']}"


def missing_facts(r: dict) -> list[str]:
    seen = "\n".join(c["result_text"] for c in r["calls"] if c["executed"])
    return [f["fact"] for f in QUESTIONS[r["question"]]["facts"] if not any(s in seen for s in f["any_of"])]


def label_of(r: dict, labels: dict) -> str | None:
    if r["outcome"] == "handoff":
        return "handoff"
    entry = labels.get(key(r))
    if entry is None:
        return None
    return entry["label"] if entry["sha256"] == r["reply_sha256"] else "hash mismatch"


def summary(rows: list[dict], labels: dict) -> dict:
    lab = [label_of(r, labels) for r in rows]
    facts = [not missing_facts(r) for r in rows]
    routed = [r for r in rows if r["route"] is not None]
    return {
        "n": len(rows),
        "route errors": sum(1 for r in routed if r["route"] != QUESTIONS[r["question"]]["intent"]),
        "handoff": lab.count("handoff"),
        "correct": lab.count("correct"), "incorrect": lab.count("incorrect"), "no answer": lab.count("no_answer"),
        "facts": sum(facts),
        "success": sum(1 for la, fa in zip(lab, facts, strict=True) if la == "correct" and fa),
        "unsupp.": sum(1 for la, fa in zip(lab, facts, strict=True) if la == "correct" and not fa),
        "step limit": sum(1 for r in rows if r["outcome"] == "step_limit"),
        "model calls": sum(len(r["model_calls"]) for r in rows) / len(rows),
        "input tok": median(sum(m["input_tokens"] for m in r["model_calls"]) for r in rows),
        "median s": median(r["seconds"] for r in rows),
        "max s": max(r["seconds"] for r in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--unlabeled", action="store_true")
    args = parser.parse_args()
    records = [json.loads(line) for f in args.files for line in open(f)]
    labels_file = HERE / "labels.json"
    labels = json.loads(labels_file.read_text())["labels"] if labels_file.exists() else {}

    if args.unlabeled:
        for r in records:
            if label_of(r, labels) in (None, "hash mismatch"):
                q = QUESTIONS[r["question"]]
                print(f"\n=== {key(r)}  route={r['route']}  sha256 {r['reply_sha256']}\nQ: {q['question']}\n"
                      f"expected: {q['expected']}")
                for c in r["calls"]:
                    print(f"  CALL {c['tool']} {c['arguments']}  executed={c['executed']}\n    -> {c['result_text'][:300]}")
                print(f"  missing facts: {missing_facts(r)}\nREPLY: {r['final']!r}")
        return

    columns = ["n", "route errors", "handoff", "correct", "incorrect", "no answer", "facts", "success", "unsupp.",
               "step limit", "model calls", "input tok", "median s", "max s"]
    print(f"{'questions':<11}{'system':<8}" + "".join(f"{c:>13}" for c in columns))
    for kind, label in [("known", "known (8)"), ("open", "open (4)"), (None, "all (12)")]:
        for system in ["W", "A", "H"]:
            rows = [r for r in records if r["system"] == system and (kind is None or QUESTIONS[r["question"]]["kind"] == kind)]
            if not rows:
                continue
            s = summary(rows, labels)
            cells = "".join(f"{s[c]:>13.2f}" if isinstance(s[c], float) and c != "input tok" else f"{s[c]:>13}"
                            for c in columns)
            print(f"{label:<11}{system:<8}{cells}")
    shared = [key(r) for r in records if r.get("other_models_loaded")]
    print(f"\nconversations with another model loaded in Ollama (timings not comparable): {len(shared)}")
    unlabeled = sum(1 for r in records if label_of(r, labels) is None)
    print(f"\nunlabeled: {unlabeled}   label hash mismatches: {sum(1 for r in records if label_of(r, labels) == 'hash mismatch')}")
    print("route errors: W and H only.  handoff: W's router chose 'other' (not labeled, not a success).")
    print("success: labeled correct AND facts retrieved.  unsupp.: correct without the facts.")
    print("model calls: mean per conversation (router, agent, and phrasing calls).  input tok: median per conversation.")
    print("median s / max s: wall-clock seconds for the automated path, per conversation.")

    print("\nOutcome per question (S success, c correct without facts, x incorrect, - no answer, h handoff), seeds 1-3:\n")
    for system in ["W", "A", "H"]:
        cells = []
        for q in QUESTIONS:
            rows = sorted((r for r in records if r["system"] == system and r["question"] == q), key=lambda r: r["seed"])
            mark = {"correct": "c", "incorrect": "x", "no_answer": "-", "handoff": "h"}
            cells.append("".join("S" if label_of(r, labels) == "correct" and not missing_facts(r)
                                 else mark.get(label_of(r, labels), "?") for r in rows))
        print(f"{system:<4}" + "".join(f"{q:>6}" for q in QUESTIONS) + "\n    " + "".join(f"{c:>6}" for c in cells))


if __name__ == "__main__":
    main()
