"""Prints conversations compactly for hand labeling: python show.py <results> <question ids...>"""
import json
import sys

from report import QUESTIONS, missing_facts

rows = [json.loads(line) for line in open(sys.argv[1])]
for q in sys.argv[2:]:
    print(f"\n######## {q} (intent {QUESTIONS[q]['intent']}): {QUESTIONS[q]['question']}\n   expected: {QUESTIONS[q]['expected']}")
    for r in rows:
        if r["question"] != q:
            continue
        calls = "; ".join(f"{c['tool']}({c['arguments']}){'' if c['executed'] else ' [NOT EXECUTED]'}" for c in r["calls"]) or "no tool"
        print(f"-- {r['system']} s{r['seed']} | route={r['route']} | {r['outcome']} | {calls} | missing: {missing_facts(r) or '-'}")
        print("   REPLY:", (r["final"] or "<EMPTY>").replace("\n", " ⏎ ")[:900])
