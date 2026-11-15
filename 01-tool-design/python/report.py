"""
Grades saved conversations and prints the comparison. Calls no model.

  python report.py results-llama3.2_3b.jsonl results-qwen2.5_3b.jsonl
  python report.py results-llama3.2_3b.jsonl --unlabeled     # print replies that still need a hand label

The rules are the ones frozen in data/questions.json:
  facts retrieved   every fact of the question appears in the text of a tool result that executed
  answer label      correct / incorrect / no_answer, by hand, in labels.json, stored with the
                    SHA-256 of the exact reply (a label whose hash doesn't match is reported, not used)
  successful        correct AND facts retrieved
"""
import argparse
import json
import re
from pathlib import Path
from statistics import median

from toolsets import DATA

HERE = Path(__file__).resolve().parent
QUESTIONS = {q["id"]: q for q in json.loads((DATA / "questions.json").read_text())["questions"]}
CALL_AS_TEXT = re.compile(r'"name"\s*:\s*"(search_help_center|get_invoices|get_refund_status)"'
                          r'|\b(search_help_center|get_invoices|get_refund_status)\s*\(')


def key(r: dict) -> str:
    return f"{r['model']}|{r['toolset']}|{r['question']}|{r['seed']}"


def missing_facts(r: dict) -> list[str]:
    """Facts of the question that no executed tool result contained."""
    seen = "\n".join(c["result_text"] for c in r["calls"] if c["executed"])
    return [f["fact"] for f in QUESTIONS[r["question"]]["facts"] if not any(s in seen for s in f["any_of"])]


def label_of(r: dict, labels: dict) -> str | None:
    entry = labels.get(key(r))
    if entry is None:
        return None
    if entry["sha256"] != r["reply_sha256"]:
        return "hash mismatch"
    return entry["label"]


def percent(part: int, whole: int) -> str:
    return f"{part}/{whole}" if whole else "-"


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
                print(f"\n=== {key(r)}  sha256 {r['reply_sha256']}\nQ: {q['question']}\nexpected: {q['expected']}")
                for c in r["calls"]:
                    print(f"  CALL {c['tool']} {c['arguments']}  executed={c['executed']}\n    -> {c['result_text'][:300]}")
                print(f"  missing facts: {missing_facts(r)}\nREPLY: {r['final']!r}")
        return

    print("Per model and toolset (n = conversations):\n")
    header = (f"{'model':<14}{'toolset':<9}{'n':>4}{'correct':>9}{'incorrect':>11}{'no answer':>11}"
              f"{'facts':>8}{'success':>9}{'unsupp.':>9}{'invalid':>9}{'as text':>9}"
              f"{'model calls':>13}{'tool calls':>12}{'input tok':>11}")
    print(header)
    groups = sorted({(r["model"], r["toolset"]) for r in records})
    for model, toolset in groups:
        rows = [r for r in records if r["model"] == model and r["toolset"] == toolset]
        lab = [label_of(r, labels) for r in rows]
        facts = [not missing_facts(r) for r in rows]
        n = len(rows)
        print(f"{model:<14}{toolset:<9}{n:>4}"
              f"{percent(lab.count('correct'), n):>9}{percent(lab.count('incorrect'), n):>11}"
              f"{percent(lab.count('no_answer'), n):>11}{percent(sum(facts), n):>8}"
              f"{percent(sum(1 for la, fa in zip(lab, facts, strict=True) if la == 'correct' and fa), n):>9}"
              f"{percent(sum(1 for la, fa in zip(lab, facts, strict=True) if la == 'correct' and not fa), n):>9}"
              f"{sum(1 for r in rows for c in r['calls'] if not c['valid']):>9}"
              f"{sum(1 for r in rows if r['final'] and CALL_AS_TEXT.search(r['final'])):>9}"
              f"{sum(len(r['model_calls']) for r in rows) / n:>13.2f}"
              f"{sum(len(r['calls']) for r in rows) / n:>12.2f}"
              f"{median(sum(m['input_tokens'] for m in r['model_calls']) for r in rows):>11.0f}")
    unlabeled = sum(1 for r in records if label_of(r, labels) is None)
    mismatched = sum(1 for r in records if label_of(r, labels) == "hash mismatch")
    print(f"\nunlabeled: {unlabeled}   label hash mismatches: {mismatched}")
    print("facts: the question's facts all appeared in executed tool results.  success: correct AND facts.")
    print("unsupp.: labeled correct without the facts.  invalid: tool calls that failed the name, JSON, or schema check.")
    print("as text: final replies containing a tool call written as text.  input tok: median per conversation, "
          "summed over its model calls (Ollama prompt_eval_count, cached tokens included).")

    print("\nSuccessful conversations per question (correct AND facts), out of the seeds run:\n")
    qids = sorted({r["question"] for r in records})
    print(f"{'model':<14}{'toolset':<9}" + "".join(f"{q:>6}" for q in qids))
    for model, toolset in groups:
        cells = []
        for q in qids:
            rows = [r for r in records if r["model"] == model and r["toolset"] == toolset and r["question"] == q]
            ok = sum(1 for r in rows if label_of(r, labels) == "correct" and not missing_facts(r))
            cells.append(f"{ok}/{len(rows)}")
        print(f"{model:<14}{toolset:<9}" + "".join(f"{c:>6}" for c in cells))


if __name__ == "__main__":
    main()
