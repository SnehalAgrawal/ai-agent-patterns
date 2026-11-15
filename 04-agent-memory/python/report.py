"""
Grades saved conversations with the frozen rules in data/evaluation.json and prints the comparison.
Calls no model.

  python report.py results-qwen2.5_3b.jsonl
  python report.py results-qwen2.5_3b.jsonl --todo     # replies and memory blocks that still need a hand label

Reads results-<...>.jsonl, results-<...>-memory.json (written by run.py), and the hand labels:
  labels.json         answers: "<question>|<SHA-256 of the reply>" -> {label, invented_memory, note,
                      b1: {current_status, no_stale_claim, change_noted} (B1 only),
                      b2_intention_as_action (B2 only)}
  memory_labels.json  blocks: "<design>|<seed>|<SHA-256 of the memory block>" -> {m1..m5: present|misstated|absent}
"""
import argparse
import json
from pathlib import Path
from statistics import mean, median

from tools import BY_NAME, DATA

HERE = Path(__file__).resolve().parent
QFILE = json.loads((DATA / "questions.json").read_text())
QUESTIONS = {q["id"]: q for q in QFILE["questions"]}
FACTS = list(QFILE["memory_facts"])
DESIGNS = ["N", "T", "S", "K"]
GROUPS = [("memory", ["A1", "A2"]), ("memory vs records", ["B1", "B2"]), ("memory + records", ["C1"]),
          ("records only", ["C2", "D1", "D2"]), ("never said", ["E1", "E2"]), ("all", list(QUESTIONS))]
OUTCOMES = ["substantive_success", "appropriate_abstention", "correct_unsupported", "failure"]


def load(name: str, section: str) -> dict:
    path = HERE / name
    return json.loads(path.read_text())[section] if path.exists() else {}


class Grader:
    def __init__(self) -> None:
        self.answers = load("labels.json", "answers")
        self.blocks = load("memory_labels.json", "blocks")

    def answer(self, r: dict) -> dict | None:
        return self.answers.get(f"{r['question']}|{r['reply_sha256']}")

    def label(self, r: dict) -> str | None:
        a = self.answer(r)
        return a["label"] if a else None

    def memory_fact(self, r: dict, fact: str) -> str | None:
        """present / misstated / absent, or None if a needed hand label is missing."""
        if r["design"] == "N":
            return "absent"
        if r["design"] == "T":
            return "present"
        entry = self.blocks.get(f"{r['design']}|{r['seed']}|{r['memory_sha256']}")
        return entry[fact] if entry else None

    def records(self, r: dict) -> bool:
        options = QUESTIONS[r["question"]]["record_evidence"]
        if not options:
            return True
        seen = "\n".join(c["result_text"] for c in r["calls"] if c["executed"] and c["tool"] in BY_NAME)
        return any(all(s in seen for s in option) for option in options)

    def memory_ok(self, r: dict) -> bool | None:
        q = QUESTIONS[r["question"]]
        states = [self.memory_fact(r, f) for f in q["required_memory"]]
        if None in states:
            return None
        history = (r["design"] != "N") if q["requires_complete_history"] else True
        return all(s == "present" for s in states) and history

    def outcome(self, r: dict) -> str | None:
        label, mem = self.label(r), self.memory_ok(r)
        if label is None or mem is None:
            return None
        available = mem and self.records(r)
        if label == "correct":
            return "substantive_success" if available else "correct_unsupported"
        if label == "no_answer" and not mem:
            return "appropriate_abstention"
        return "failure"


def rows_for(records: list[dict], design: str, qids: list[str]) -> list[dict]:
    return [r for r in records if r["design"] == design and r["question"] in qids]


def table(records: list[dict], g: Grader) -> None:
    cols = ["n", "correct", "incorrect", "no answer", "invented", "success", "abstain", "unsupp.", "failure"]
    print(f"{'questions':<19}{'design':<8}" + "".join(f"{c:>10}" for c in cols))
    for name, qids in GROUPS:
        for d in DESIGNS:
            rows = rows_for(records, d, qids)
            if not rows:
                continue
            lab = [g.label(r) for r in rows]
            out = [g.outcome(r) for r in rows]
            inv = [g.answer(r)["invented_memory"] for r in rows if g.answer(r)]
            s = {"n": len(rows), "correct": lab.count("correct"), "incorrect": lab.count("incorrect"),
                 "no answer": lab.count("no_answer"), "invented": sum(inv), "success": out.count(OUTCOMES[0]),
                 "abstain": out.count(OUTCOMES[1]), "unsupp.": out.count(OUTCOMES[2]), "failure": out.count(OUTCOMES[3])}
            if None in lab or None in out:
                s["failure"] = f"{s['failure']}+?"
            print(f"{name:<19}{d:<8}" + "".join(f"{s[c]!s:>10}" for c in cols))


def memory_table(memories: list[dict], g: Grader) -> None:
    print("\nMemory facts carried by S and K, per seed (P present, m misstated, - absent, ? unlabeled):")
    print(f"{'':<10}" + "".join(f"{f:>4}" for f in FACTS) + "   notes parse failures")
    for m in memories:
        if m["design"] not in ("S", "K"):
            continue
        entry = g.blocks.get(f"{m['design']}|{m['seed']}|{m['block_sha256']}")
        cells = "".join(f"{({'present': 'P', 'misstated': 'm', 'absent': '-'}.get(entry[f]) if entry else '?'):>4}"
                        for f in FACTS)
        fails = sum(1 for i in m["items"] if i.get("parse_failed")) if m["design"] == "K" else "-"
        print(f"{m['design']} seed {m['seed']:<3}{cells}   {fails}")


def stale(records: list[dict], g: Grader) -> None:
    print("\nB1 (memory says pending; records say refunded), conversations per design:")
    print(f"{'':<8}{'retrieved':>10}{'current':>10}{'no stale':>10}{'change noted':>14}")
    for d in DESIGNS:
        rows = rows_for(records, d, ["B1"])
        if not rows:
            continue
        sub = [(g.answer(r) or {}).get("b1") or {} for r in rows]
        noted = [s.get("change_noted") for s in sub]
        applicable = [n for n in noted if n is not None]
        retrieved = sum(any("PAY-5008" in c["result_text"] for c in r["calls"] if c["executed"]) for r in rows)
        print(f"{d:<8}{retrieved:>10}"
              f"{sum(1 for s in sub if s.get('current_status')):>10}{sum(1 for s in sub if s.get('no_stale_claim')):>10}"
              f"{(f'{sum(applicable)} of {len(applicable)}' if applicable else 'n/a'):>14}")
    print("B2: replies stating that a seat was removed (intention reported as action):  " + "  ".join(
        f"{d} {sum(1 for r in rows_for(records, d, ['B2']) if (g.answer(r) or {}).get('b2_intention_as_action'))}"
        for d in DESIGNS))


def cost(records: list[dict], memories: list[dict]) -> None:
    first = {(r["design"], r["question"], r["seed"]): r["model_calls"][0]["input_tokens"] for r in records if r["model_calls"]}
    print("\nCost (every model call):")
    print(f"{'design':<8}{'creation calls':>15}{'creation in/out tok':>21}{'memory tok':>12}{'calls':>8}"
          f"{'input tok':>11}{'of which memory':>17}{'output tok':>12}{'median s':>10}{'max s':>8}{'total in tok':>14}")
    for d in DESIGNS:
        rows = [r for r in records if r["design"] == d]
        if not rows:
            continue
        mems = [m for m in memories if m["design"] == d]
        c_calls = sum(len(m["calls"]) for m in mems)
        c_in = sum(c["input_tokens"] for m in mems for c in m["calls"])
        c_out = sum(c["output_tokens"] for m in mems for c in m["calls"])
        block = [first[(d, r["question"], r["seed"])] - first[("N", r["question"], r["seed"])] for r in rows
                 if (d, r["question"], r["seed"]) in first and ("N", r["question"], r["seed"]) in first]
        carried = [(first[(d, r["question"], r["seed"])] - first[("N", r["question"], r["seed"])]) * len(r["model_calls"])
                   for r in rows if (d, r["question"], r["seed"]) in first and ("N", r["question"], r["seed"]) in first]
        tin = [sum(c["input_tokens"] for c in r["model_calls"]) for r in rows]
        tout = [sum(c["output_tokens"] for c in r["model_calls"]) for r in rows]
        print(f"{d:<8}{c_calls:>15}{f'{c_in} / {c_out}':>21}{(median(block) if block else 0):>12}"
              f"{mean(len(r['model_calls']) for r in rows):>8.2f}{median(tin):>11}{(median(carried) if carried else 0):>17}"
              f"{median(tout):>12}{median(r['seconds'] for r in rows):>10.2f}{max(r['seconds'] for r in rows):>8.2f}"
              f"{sum(tin) + c_in:>14}")
    print("creation: all memory-creation calls over all seeds.  memory tok: median memory-block tokens (first call minus N's).\n"
          "calls: mean per conversation.  input/output tok, of which memory: median per conversation.\n"
          "total in tok: every conversation's input tokens plus memory creation, whole experiment.")


def grid(records: list[dict], g: Grader) -> None:
    print("\nOutcome per question (S success, a appropriate abstention, c correct unsupported, x failure, ? unlabeled), seeds 1-3:\n")
    mark = {"substantive_success": "S", "appropriate_abstention": "a", "correct_unsupported": "c", "failure": "x"}
    for d in DESIGNS:
        cells = ["".join(mark.get(g.outcome(r), "?") for r in sorted(rows_for(records, d, [q]), key=lambda r: r["seed"]))
                 for q in QUESTIONS]
        print(f"{d:<4}" + "".join(f"{q:>6}" for q in QUESTIONS) + "\n    " + "".join(f"{c:>6}" for c in cells))


def todo(records: list[dict], memories: list[dict], g: Grader) -> None:
    for m in memories:
        if m["design"] in ("S", "K") and f"{m['design']}|{m['seed']}|{m['block_sha256']}" not in g.blocks:
            print(f"MEMORY {m['design']}|{m['seed']}|{m['block_sha256']}\n{m['block']}\n")
    seen = set()
    for r in records:
        k = f"{r['question']}|{r['reply_sha256']}"
        if k not in g.answers and k not in seen:
            seen.add(k)
            print(f"ANSWER {k}  ({r['design']} s{r['seed']}, tools {[c['tool'] for c in r['calls']]})\n  {r['final']!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--todo", action="store_true")
    args = parser.parse_args()
    records = [json.loads(line) for line in open(args.file)]
    memories = json.loads(Path(args.file.replace(".jsonl", "-memory.json")).read_text())
    g = Grader()
    if args.todo:
        todo(records, memories, g)
        return
    table(records, g)
    print("success: correct and evidence available.  abstain: no answer when needed memory wasn't available.\n"
          "unsupp.: correct without the evidence.  invented: replies flagged for invented memory.")
    memory_table(memories, g)
    stale(records, g)
    cost(records, memories)
    grid(records, g)
    print(f"\nconversations with another model loaded in Ollama: {sum(1 for r in records if r['other_models_loaded'])}"
          f";  memory creations with another model loaded: {sum(1 for m in memories if m['other_models_loaded'])}")


if __name__ == "__main__":
    main()
