"""
Grades saved conversations with the frozen rules in data/evaluation.json and prints the comparison.
Calls no model.

  python report.py results-qwen2.5_3b.jsonl
  python report.py results-qwen2.5_3b.jsonl --todo     # texts that still need a hand label

Hand labels:
  labels.json           answers: "<question>|<sha256 of the text>" -> label (the reply the customer
                        gets, and R's drafts). One label per distinct text and question.
  handoff_labels.json   replies:  "<question>|<sha256 of a specialist reply>" -> {fact: present|misstated|absent}
                        requests: "<question>|<sha256 of a request>" -> {fact: true|false}  (does the request,
                        read on its own, ask for the information the fact carries; used only for M's
                        delegation stage)
"""
import argparse
import json
import re
from pathlib import Path
from statistics import mean, median

from tools import BY_NAME, DATA, run

HERE = Path(__file__).resolve().parent
QUESTIONS = {q["id"]: q for q in json.loads((DATA / "questions.json").read_text())["questions"]}
GROUPS = [("known", "known (8)"), ("open", "open (4)"), (None, "all (12)")]
SYSTEMS = ["S", "M", "R"]

# Which specialist can provide each fact (evaluation.json, m_failure_stage.can_provide), computed from the data.
_BILLING = "".join(run(n, "")["result_text"] for n in ("get_invoices", "get_refund_status", "get_payments",
                                                         "get_plan_history"))
_HELP = json.dumps(json.loads((DATA / "help_center.json").read_text()), ensure_ascii=False)
CAN_PROVIDE = {(q, f["fact"]): {s for s, text in (("billing", _BILLING), ("help_center", _HELP))
                                if any(a in text for a in f["any_of"])}
               for q, qq in QUESTIONS.items() for f in qq["facts"]}


def facts_of(q: str) -> list[dict]:
    return QUESTIONS[q]["facts"]


def missing(q: str, calls: list[dict]) -> list[str]:
    """Facts not in any executed data-tool result among `calls` (string match, as F1)."""
    seen = "\n".join(c["result_text"] for c in calls if c["executed"] and c["tool"] in BY_NAME)
    return [f["fact"] for f in facts_of(q) if not any(a in seen for a in f["any_of"])]


def load(name: str, section: str) -> dict:
    path = HERE / name
    return json.loads(path.read_text())[section] if path.exists() else {}


class Grader:
    def __init__(self) -> None:
        self.answers = load("labels.json", "answers")
        self.replies = load("handoff_labels.json", "replies")
        self.requests = load("handoff_labels.json", "requests")

    def label(self, q: str, sha: str) -> str | None:
        entry = self.answers.get(f"{q}|{sha}")
        return entry["label"] if entry else None

    def final(self, r: dict) -> str | None:
        return self.label(r["question"], r["reply_sha256"])

    def draft(self, r: dict) -> str | None:
        return self.label(r["question"], r["draft_sha256"]) if r["system"] == "R" else None

    # ---- facts ----
    def f1(self, r: dict) -> bool:
        return not missing(r["question"], r["calls"])

    def passed_on(self, r: dict) -> dict[str, str | None]:
        """Per fact: 'present' if any specialist reply states it correctly, else 'misstated'/'absent'/None (unlabeled)."""
        out = {}
        for f in facts_of(r["question"]):
            seen = []
            for h in r["handoffs"]:
                if not h["replied"]:
                    seen.append("absent")  # the fixed "couldn't finish" text passes nothing on
                    continue
                entry = self.replies.get(f"{r['question']}|{h['reply_sha256']}")
                seen.append(entry["facts"].get(f["fact"]) if entry else None)
            if "present" in seen:
                out[f["fact"]] = "present"
            elif None in seen:
                out[f["fact"]] = None
            else:
                out[f["fact"]] = "misstated" if "misstated" in seen else "absent"
        return out

    def f2(self, r: dict) -> bool | None:
        p = self.passed_on(r)
        if any(v is None for v in p.values()):
            return None
        return all(v == "present" for v in p.values())

    def f3(self, r: dict) -> bool | None:
        if r["system"] == "S":
            return self.f1(r)
        if r["system"] == "M":
            return self.f2(r)
        own = [c for c in r["calls"] if c["role"] == ("reviewer" if r["writer"] == "reviewer" else "drafter")]
        return not missing(r["question"], own)

    def conflicts(self, r: dict) -> list[str]:
        out = []
        for f in facts_of(r["question"]):
            labels = {self.replies.get(f"{r['question']}|{h['reply_sha256']}", {}).get("facts", {}).get(f["fact"])
                      for h in r["handoffs"] if h["replied"]}
            if {"present", "misstated"} <= labels:
                out.append(f["fact"])
        return out

    # ---- R ----
    def reviewer_outcome(self, r: dict) -> str | None:
        d, f = self.draft(r), self.final(r)
        if d is None or f is None:
            return None
        if r["action"] == "no_verdict":
            return "no_verdict"
        if d == "correct":
            if r["action"] == "approve":
                return "approved_correct"
            return "rewrote_still_correct" if f == "correct" else "damaged"
        if r["action"] == "approve":
            return "approved_wrong"
        return "fixed" if f == "correct" else "rewrote_still_wrong"

    # ---- M ----
    def stage(self, r: dict) -> str | None:
        if r["system"] != "M" or self.final(r) in (None, "correct"):
            return None
        q = r["question"]
        asked = {h["specialist"] for h in r["handoffs"]}
        if not r["handoffs"]:
            return "delegation"
        for f in facts_of(q):
            providers = CAN_PROVIDE[(q, f["fact"])]
            if not providers & asked:
                return "delegation"
            reqs = [self.requests.get(f"{q}|{h['request_sha256']}", {}).get("facts", {}).get(f["fact"])
                    for h in r["handoffs"] if h["specialist"] in providers]
            if None in reqs:
                return None  # needs a request label
            if not any(reqs):
                return "delegation"
        if not self.f1(r):
            return "retrieval"
        f2 = self.f2(r)
        if f2 is None:
            return None
        return "handoff" if not f2 else "synthesis"


def table(records: list[dict], g: Grader) -> None:
    cols = ["n", "correct", "incorrect", "no answer", "F1", "F3", "success", "succ. F1", "unsupp.", "step limit"]
    print(f"{'questions':<11}{'system':<8}" + "".join(f"{c:>11}" for c in cols))
    for kind, name in GROUPS:
        for system in SYSTEMS:
            rows = [r for r in records if r["system"] == system and (kind is None or QUESTIONS[r["question"]]["kind"] == kind)]
            if not rows:
                continue
            lab = [g.final(r) for r in rows]
            f1 = [g.f1(r) for r in rows]
            f3 = [g.f3(r) for r in rows]
            s = {"n": len(rows), "correct": lab.count("correct"), "incorrect": lab.count("incorrect"),
                 "no answer": lab.count("no_answer"), "F1": sum(f1),
                 "F3": "?" if None in f3 else sum(f3),
                 "success": "?" if None in f3 else sum(1 for la, x in zip(lab, f3, strict=True) if la == "correct" and x),
                 "succ. F1": sum(1 for la, x in zip(lab, f1, strict=True) if la == "correct" and x),
                 "unsupp.": "?" if None in f3 else sum(1 for la, x in zip(lab, f3, strict=True) if la == "correct" and not x),
                 "step limit": sum(1 for r in rows if r["outcome"] == "step_limit")}
            print(f"{name:<11}{system:<8}" + "".join(f"{s[c]!s:>11}" for c in cols))


def cost(records: list[dict]) -> None:
    print(f"\n{'questions':<11}{'system':<8}{'model calls':>12}{'input tok':>11}{'output tok':>11}{'median s':>10}{'max s':>8}"
          "   mean model calls by role")
    for kind, name in GROUPS:
        for system in SYSTEMS:
            rows = [r for r in records if r["system"] == system and (kind is None or QUESTIONS[r["question"]]["kind"] == kind)]
            if not rows:
                continue
            roles = {}
            for r in rows:
                for c in r["model_calls"]:
                    role = c["role"].replace("specialist:", "")
                    roles[role] = roles.get(role, 0) + 1
            by_role = ", ".join(f"{k} {v / len(rows):.2f}" for k, v in roles.items())
            print(f"{name:<11}{system:<8}{mean(len(r['model_calls']) for r in rows):>12.2f}"
                  f"{median(sum(c['input_tokens'] for c in r['model_calls']) for r in rows):>11}"
                  f"{median(sum(c['output_tokens'] for c in r['model_calls']) for r in rows):>11}"
                  f"{median(r['seconds'] for r in rows):>10.2f}{max(r['seconds'] for r in rows):>8.2f}   {by_role}")


def reviewer(records: list[dict], g: Grader) -> None:
    rows = [r for r in records if r["system"] == "R"]
    if not rows:
        return
    names = ["approved_correct", "rewrote_still_correct", "damaged", "fixed", "approved_wrong",
             "rewrote_still_wrong", "no_verdict"]
    out = [g.reviewer_outcome(r) for r in rows]
    print(f"\nR reviewer outcomes ({len(rows)}):  " + "  ".join(f"{n} {out.count(n)}" for n in names)
          + f"  unlabeled {out.count(None)}")
    acts = [r["action"] for r in rows]
    checked = [any(c["role"] == "reviewer" and c["executed"] for c in r["calls"]) for r in rows]
    evid = [not missing(r["question"], [c for c in r["calls"] if c["role"] == "reviewer"]) for r in rows]
    print(f"  actions: approve {acts.count('approve')}, rewrite {acts.count('rewrite')}, no_verdict {acts.count('no_verdict')};"
          f"  checked {sum(checked)};  evidence in front {sum(evid)};"
          f"  approved unchecked {sum(1 for a, c in zip(acts, checked, strict=True) if a == 'approve' and not c)}")
    near = [r for r in rows if r["action"] == "rewrite" and re.sub(r"[^A-Za-z]", "", r["review_text"] or "").upper() == "APPROVE"]
    print(f"  rewrites whose letters alone spell APPROVE (sent to the customer as written, per the frozen rule): {len(near)}"
          + (f"  {[(r['question'], r['seed'], r['review_text']) for r in near]}" if near else ""))
    drafts = [g.draft(r) for r in rows]
    print(f"  draft labels: correct {drafts.count('correct')}, incorrect {drafts.count('incorrect')}, "
          f"no answer {drafts.count('no_answer')}, unlabeled {drafts.count(None)}")


def manager(records: list[dict], g: Grader) -> None:
    rows = [r for r in records if r["system"] == "M"]
    if not rows:
        return
    stages = [g.stage(r) for r in rows if g.final(r) not in (None, "correct")]
    names = ["delegation", "retrieval", "handoff", "synthesis"]
    print(f"\nM first failing stage ({len(stages)} not correct):  " + "  ".join(f"{n} {stages.count(n)}" for n in names)
          + f"  unlabeled {stages.count(None)}")
    hs = [h for r in rows for h in r["handoffs"]]
    print(f"  specialist calls {len(hs)} (billing {sum(h['specialist'] == 'billing' for h in hs)}, help center "
          f"{sum(h['specialist'] == 'help_center' for h in hs)});  couldn't finish {sum(not h['replied'] for h in hs)};  "
          f"conversations with no specialist call {sum(not r['handoffs'] for r in rows)}")
    invalid = [c for r in rows for c in r["calls"] if c["role"].startswith("specialist") and not c["valid"]]
    print(f"  specialists' invalid data-tool calls: {len(invalid)}")
    conflicts = [(r["question"], r["seed"], g.conflicts(r)) for r in rows if g.conflicts(r)]
    print(f"  facts both passed on and misstated in one conversation: {conflicts or 'none'}")


def outcomes(records: list[dict], g: Grader) -> None:
    print("\nOutcome per question (S success, c correct without F3, x incorrect, - no answer, ? unlabeled), seeds 1-3:\n")
    mark = {"correct": "c", "incorrect": "x", "no_answer": "-"}
    for system in SYSTEMS:
        cells = []
        for q in QUESTIONS:
            rows = sorted((r for r in records if r["system"] == system and r["question"] == q), key=lambda r: r["seed"])
            cells.append("".join("S" if g.final(r) == "correct" and g.f3(r) else mark.get(g.final(r), "?") for r in rows))
        print(f"{system:<4}" + "".join(f"{q:>6}" for q in QUESTIONS) + "\n    " + "".join(f"{c:>6}" for c in cells))


def todo(records: list[dict], g: Grader) -> None:
    seen = set()
    for r in records:
        q = r["question"]
        texts = [("final", r["reply_sha256"], r["final"])]
        if r["system"] == "R":
            texts.append(("draft", r["draft_sha256"], r["draft"]))
        for kind, sha, text in texts:
            k = f"{q}|{sha}"
            if g.label(q, sha) is None and k not in seen:
                seen.add(k)
                print(f"ANSWER {k}  ({r['system']} s{r['seed']} {kind})\n  {text!r}")
        for h in r["handoffs"]:
            k = f"{q}|{h['reply_sha256']}"
            if h["replied"] and k not in g.replies and k not in seen:
                seen.add(k)
                print(f"REPLY {k}  ({h['specialist']})\n  {h['reply']!r}")
            k = f"{q}|{h['request_sha256']}"
            if k not in g.requests and k not in seen:
                seen.add(k)
                print(f"REQUEST {k}  ({h['specialist']})\n  {h['request']!r}")


def side_check(records: list[dict]) -> None:
    path = HERE.parent.parent / "02-agentic-workflows" / "python" / "results-qwen2.5_3b.jsonl"
    if not path.exists():
        return
    a = {(r["question"], r["seed"]): r["final"] for r in map(json.loads, open(path)) if r["system"] == "A"}
    rows = [r for r in records if r["system"] == "S"]
    same = sum(1 for r in rows if a.get((r["question"], r["seed"])) == r["final"])
    print(f"\nSide check: S replies identical to Article #24's agent A: {same} of {len(rows)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--todo", action="store_true")
    args = parser.parse_args()
    records = [json.loads(line) for f in args.files for line in open(f)]
    g = Grader()
    if args.todo:
        todo(records, g)
        return
    table(records, g)
    print("F1: some agent's executed data-tool results contain every fact.  F3: the writer had every fact as evidence.\n"
          "success: correct AND F3.  succ. F1: correct AND F1 (Article #24's definition).  unsupp.: correct without F3.")
    cost(records)
    print("model calls: mean per conversation, every agent.  input/output tok: median per conversation, all calls.\n"
          "median s / max s: end-to-end wall-clock seconds per conversation.")
    reviewer(records, g)
    manager(records, g)
    outcomes(records, g)
    side_check(records)
    print(f"\nconversations with another model loaded in Ollama: {sum(1 for r in records if r['other_models_loaded'])}")


if __name__ == "__main__":
    main()
