"""
Every table, computed from results-<set>.jsonl (the guard verdicts) and the ground truth in
sources.py (data files, recorded fields, and the original hand labels). No model is called.

    python report.py --set test > report-test.txt
    python report.py --set crosscheck > report-crosscheck.txt

Three kinds of evidence, kept apart in every table:
  objective      computed from data files and recorded fields (id status, repeats, refund actions)
  hand           labels made by hand for Articles #23-#29, read as they are (answer, claims, handoffs)
  replay         this demo's guard verdicts about recorded decisions
"""
import argparse
import collections
import json
from pathlib import Path

import guards
import sources

HERE = Path(__file__).resolve().parent
LABEL_COLUMNS = ["correct", "incorrect", "no_answer", "success", "failure", None]
SHORT = {"01-tool-design": "01", "02-agentic-workflows": "02", "03-multi-agent": "03", "04-agent-memory": "04",
         "05-planning": "05", "06-state-and-workflows": "06", "07-human-in-the-loop": "07"}


def load(which: str) -> tuple[list[dict], dict]:
    rows = [json.loads(line) for line in (HERE / f"results-{which}.jsonl").read_text().splitlines()]
    convs = {c["key"]: c for c in sources.load(which)}
    return rows, convs


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else f"{n}/0"


def head(title: str) -> None:
    print(f"\n{title}\n{'=' * len(title)}")


def by_index(conv: dict) -> dict:
    return {c["i"]: c for c in conv["calls"]}


def columns(totals) -> list[str]:
    cols = ["correct", "success", "incorrect", "failure", "no_answer"]
    return cols + (["unlabelled"] if totals["unlabelled"] else [])


def label_name(lab) -> str:
    return "unlabelled" if lab is None else lab


# --- 0. what was replayed ------------------------------------------------------------------
def overview(rows, convs) -> None:
    head("0. What was replayed")
    t = collections.Counter((r["demo"], r["design"]) for r in rows)
    calls = collections.Counter()
    for r in rows:
        calls[(r["demo"], r["design"])] += len(convs[r["key"]]["calls"])
    print(f"{'demo':24}{'design':>8}{'conversations':>15}{'tool calls':>12}")
    for (d, des), n in sorted(t.items()):
        print(f"{d:24}{des:>8}{n:>15}{calls[(d, des)]:>12}")
    print(f"{'all':24}{'':>8}{sum(t.values()):>15}{sum(calls.values()):>12}")
    labs = collections.Counter(label_name(r["label"]) for r in rows)
    print("\nConversation labels (hand; 06: #28's frozen success): " + ", ".join(f"{k} {v}" for k, v in sorted(labs.items())))


# --- 1. baseline: what the recorded models did after an error ------------------------------
def after_error(rows, convs) -> None:
    head("1. Baseline (objective, recorded): the schema check every demo already had, and what the model did next")
    invalid = notexec = 0
    nxt = collections.Counter()
    nxt_by_demo = collections.defaultdict(collections.Counter)
    for r in rows:
        conv = convs[r["key"]]
        order = [c for kind, c in guards.timeline(conv) if kind == "call"]
        for j, c in enumerate(order):
            if c["valid"] and c["executed"]:
                continue
            if not c["valid"]:
                invalid += 1
            else:
                notexec += 1
            later = [x for x in order[j + 1:] if x["agent"] == c["agent"]]
            if not later:
                kind = "none (replied or run ended)"
            elif guards._canonical(later[0]) is not None and guards._canonical(later[0]) == guards._canonical(c):
                kind = "identical call again"
            elif later[0]["tool"] == c["tool"]:
                kind = "same tool, different arguments"
            else:
                kind = "a different tool"
            nxt[kind] += 1
            nxt_by_demo[SHORT[r["demo"]]][kind] += 1
    total = invalid + notexec
    print(f"Calls with an error result: {total} (schema-invalid {invalid}, valid but failed {notexec}).")
    print("The error went back to the model. The same agent's next call was:")
    kinds = ["identical call again", "same tool, different arguments", "a different tool", "none (replied or run ended)"]
    for k in kinds:
        print(f"  {k:34}{pct(nxt[k], total):>16}")
    print("\nBy demo:      " + "".join(f"{k[:18]:>20}" for k in kinds))
    for d in sorted(nxt_by_demo):
        n = sum(nxt_by_demo[d].values())
        print(f"  {d} ({n:>3})   " + "".join(f"{nxt_by_demo[d][k]:>20}" for k in kinds))


# --- 2. independent per-guard evaluation ---------------------------------------------------
def id_guards(rows, convs) -> None:
    head("2a. provenance and id_format, independently (unit: id arguments; truth: objective id status)")
    stat = collections.defaultdict(collections.Counter)
    examples = collections.defaultdict(list)
    for r in rows:
        conv = convs[r["key"]]
        calls = by_index(conv)
        for v in r["independent"]["calls"]:
            c = calls[v["i"]]
            for arg, kind, value in guards.id_values(c):
                if value in conv["owned"][kind]:
                    truth = "owned"
                elif value in conv["other"][kind]:
                    truth = "other_customer"
                else:
                    truth = "nonexistent"
                validity = "valid" if c["valid"] else "schema-invalid"
                for g in ("provenance", "id_format"):
                    stat[(g, truth)]["total"] += 1
                    stat[(g, truth, validity)]["total"] += 1
                    if v[g]:
                        stat[(g, truth)]["fired"] += 1
                        stat[(g, truth, validity)]["fired"] += 1
                if truth == "nonexistent" and len(examples[SHORT[r["demo"]]]) < 6 and value not in examples[SHORT[r["demo"]]]:
                    examples[SHORT[r["demo"]]].append(value)
                if truth == "owned" and v["provenance"]:
                    examples["owned but no provenance"].append(f"{r['key'].split('|', 2)[2]} {arg}={value}")
    print(f"{'truth':18}{'provenance fired':>22}{'id_format fired':>22}")
    for truth in ("nonexistent", "other_customer", "owned"):
        print(f"{truth:18}" + "".join(f"{pct(stat[(g, truth)]['fired'], stat[(g, truth)]['total']):>22}"
                                      for g in ("provenance", "id_format")))
    print("\nOf which on calls the schema check had already rejected / passed:")
    for truth in ("nonexistent", "other_customer", "owned"):
        for validity in ("schema-invalid", "valid"):
            s = [stat[(g, truth, validity)] for g in ("provenance", "id_format")]
            if s[0]["total"]:
                print(f"  {truth:16}{validity:16}" + "".join(f"{pct(x['fired'], x['total']):>22}" for x in s))
    print("\nExamples of nonexistent ids, by demo: ")
    for d, vals in sorted(examples.items()):
        if d != "owned but no provenance":
            print(f"  {d}: " + ", ".join(vals))
    if examples["owned but no provenance"]:
        print("Owned ids the provenance guard would block (false alarms):")
        for e in examples["owned but no provenance"]:
            print("  " + e)


def repeat_guard(rows, convs) -> None:
    head("2b. repeat, independently (unit: calls; truth: objective, was the result byte-identical?)")
    stat = collections.Counter()
    per = collections.Counter()
    for r in rows:
        conv = convs[r["key"]]
        calls = by_index(conv)
        for v in r["independent"]["calls"]:
            stat["calls"] += 1
            if not v["repeat"]:
                continue
            c = calls[v["i"]]
            e = guards.repeated_from(conv, c)
            truth = "redundant" if c["result_text"] == e["result_text"] else "informative"
            stat[truth] += 1
            per[(SHORT[r["demo"]], r["design"], truth)] += 1
    fired = stat["redundant"] + stat["informative"]
    print(f"Fired on {fired} of {stat['calls']} calls.")
    print(f"  redundant (same result as the earlier call): {pct(stat['redundant'], fired)}")
    print(f"  informative (the result differed):           {pct(stat['informative'], fired)}")
    print("By demo and design: " + ", ".join(f"{d} {des} {t} {n}" for (d, des, t), n in sorted(per.items())))


def budget_guard(rows, convs) -> None:
    head("2c. budget, independently (unit: conversations; truth: the conversation's label)")
    totals = collections.Counter(label_name(r["label"]) for r in rows)
    cols = columns(totals)
    print("Conversations that would hit the budget at K tool calls (the frozen K is 8). Columns: label (denominator).")
    print(f"{'K':>4}" + "".join(f"{c + ' (' + str(totals[c]) + ')':>18}" for c in cols))
    for k in json.loads((HERE.parent / "data" / "evaluation.json").read_text())["guards"]["budget"]["sweep"]:
        hit = collections.Counter()
        for r in rows:
            n = len(convs[r["key"]]["calls"])
            if n > k:
                hit[label_name(r["label"])] += 1
        print(f"{k:>4}" + "".join(f"{hit[c]:>18}" for c in cols))
    over = collections.Counter((SHORT[r["demo"]], r["design"]) for r in rows if len(convs[r["key"]]["calls"]) > guards.K)
    print(f"At K={guards.K}, by demo and design: " + ", ".join(f"{d} {des} {n}" for (d, des), n in sorted(over.items())))


def refund_guard(rows, convs) -> None:
    head("2d. refund_policy, independently (06, design L; unit: issue_refund calls; truth: objective, #28's frozen rule)")
    stat = collections.Counter()
    lines = []
    for r in rows:
        if r["demo"] != "06-state-and-workflows":
            continue
        conv = convs[r["key"]]
        calls = by_index(conv)
        for v in r["independent"]["calls"]:
            c = calls[v["i"]]
            if c["tool"] != "issue_refund":
                continue
            truth = conv["refund_truth"][c["i"]]
            fired = bool(v["refund_policy"])
            stat[(truth, fired)] += 1
            lines.append(f"  {conv['question']} seed {conv['seed']} call {c['i']}: {c['arguments']:28} {truth:10} "
                         f"{v['refund_policy'] or 'passes'}")
    for truth in ("wrong", "right", "no_effect"):
        n = stat[(truth, True)] + stat[(truth, False)]
        print(f"  {truth:10} refunds: guard fired on {pct(stat[(truth, True)], n)}")
    print("Every call:")
    print("\n".join(lines))


def handoff_guard(rows, convs) -> None:
    head("2e. handoff_evidence, independently (03 M; unit: specialist replies; truth: hand, F2 facts per reply)")
    hl = json.loads((sources.ROOT / "03-multi-agent" / "python" / "handoff_labels.json").read_text())["replies"]
    stat = collections.Counter()
    final = collections.Counter()
    for r in rows:
        if r["demo"] != "03-multi-agent" or r["design"] != "M":
            continue
        conv = convs[r["key"]]
        calls = by_index(conv)
        any_fire = False
        for h in r["independent"]["handoffs"]:
            c = calls[h["i"]]
            lab = hl.get(f"{conv['question']}|{sources.sha256(c['result_text'])}")
            if lab is None:
                truth = "unlabelled"
            else:
                truth = "carried evidence" if any(v == "present" for v in lab["facts"].values()) else "no fact present"
            reason = h["reason"].split(" ")[1] if h["reason"] else "passes"
            stat[(truth, reason)] += 1
            any_fire = any_fire or bool(h["reason"])
        final[(any_fire, r["label"])] += 1
    reasons = ["lookup_failed", "no_lookup", "passes"]
    print(f"{'hand label of the reply':24}" + "".join(f"{x:>16}" for x in reasons))
    for truth in ("no fact present", "carried evidence", "unlabelled"):
        print(f"{truth:24}" + "".join(f"{stat[(truth, x)]:>16}" for x in reasons))
    print("\nM conversations (unit: conversations; final answer label, hand):")
    for fired in (True, False):
        n = sum(v for (f, _), v in final.items() if f == fired)
        print(f"  {'with' if fired else 'without'} a fired handoff ({n}): " +
              ", ".join(f"{lab} {v}" for (f, lab), v in sorted(final.items(), key=str) if f == fired))


def reply_checks(rows, convs) -> None:
    head("2f. reply checks, independently (unit: final replies; truth: hand labels)")
    totals = collections.Counter(label_name(r["label"]) for r in rows)
    cols = columns(totals)
    print(f"{'check':16}" + "".join(f"{c + ' (' + str(totals[c]) + ')':>18}" for c in cols))
    for name in guards.REPLY_ORDER:
        hit = collections.Counter(label_name(r["label"]) for r in rows if r["independent"]["reply"].get(name))
        print(f"{name:16}" + "".join(f"{hit[c]:>18}" for c in cols))
    print("\nunbacked_claim against the hand action-claim labels (05 fabricated_action, 06 claim vs ledger, 07 false_claim):")
    cm = collections.Counter()
    fps = []
    for r in rows:
        if r["action_claim"] is None:
            continue
        fired = bool(r["independent"]["reply"].get("unbacked_claim"))
        cm[(r["action_claim"], fired)] += 1
        if fired != r["action_claim"]:
            fps.append(f"  {'missed' if r['action_claim'] else 'false alarm'}: {r['key'].split('|', 2)[2]}: "
                       f"{convs[r['key']]['final'][:150]!r}")
    pos = cm[(True, True)] + cm[(True, False)]
    neg = cm[(False, True)] + cm[(False, False)]
    print(f"  labelled false claim: fired on {pct(cm[(True, True)], pos)}")
    print(f"  labelled not a false claim: fired on {pct(cm[(False, True)], neg)}")
    print("\n".join(fps))
    unl = [r for r in rows if r["action_claim"] is None and r["independent"]["reply"].get("unbacked_claim")]
    if unl:
        print(f"Fired on {len(unl)} replies without an action-claim label (01-04, unlabelled for claims):")
        for r in unl:
            print(f"  {r['key'].split('|', 2)[2]} [{r['label']}]: {r['independent']['reply']['unbacked_claim'][:90]}")


# --- 3. sequential first-fire replay -------------------------------------------------------
def sequential(rows, convs) -> None:
    head("3. Sequential first-fire replay (unit: conversations; replay outcome x hand label)")
    print("Order: budget, [recorded schema check], provenance, repeat, refund_policy; handoff_evidence on each specialist\n"
          "reply; then empty_reply, tool_call_text, unbacked_claim, no_evidence on the final reply. The first guard that\n"
          "fires ends the run: the call doesn't run, or the reply isn't sent. What the model would have done next is unknown.")
    totals = collections.Counter(label_name(r["label"]) for r in rows)
    cols = columns(totals)
    t = collections.Counter()
    for r in rows:
        s = r["sequential"]
        t[(s["guard"] or "delivered", label_name(r["label"]))] += 1
    order = ["budget", "provenance", "repeat", "refund_policy", "handoff_evidence"] + guards.REPLY_ORDER + ["delivered"]
    print(f"\n{'first fire':18}" + "".join(f"{c + ' (' + str(totals[c]) + ')':>18}" for c in cols) + f"{'all':>8}")
    for g in order:
        n = sum(t[(g, c)] for c in cols)
        print(f"{g:18}" + "".join(f"{t[(g, c)]:>18}" for c in cols) + f"{n:>8}")
    good = {"correct", "success"}
    bad = {"incorrect", "failure"}
    stopped = [r for r in rows if r["sequential"]["outcome"] != "delivered"]
    delivered = [r for r in rows if r["sequential"]["outcome"] == "delivered"]
    n_good = sum(1 for r in rows if r["label"] in good)
    n_bad = sum(1 for r in rows if r["label"] in bad)
    n_na = sum(1 for r in rows if r["label"] == "no_answer")
    print("\nSummary:")
    print(f"  correct (or 06 success), still delivered:      {pct(sum(1 for r in delivered if r['label'] in good), n_good)}")
    print(f"  correct (or 06 success), stopped by a guard:   {pct(sum(1 for r in stopped if r['label'] in good), n_good)}")
    print(f"  incorrect (or 06 failure), stopped:            {pct(sum(1 for r in stopped if r['label'] in bad), n_bad)}")
    print(f"  incorrect (or 06 failure), passed every guard: {pct(sum(1 for r in delivered if r['label'] in bad), n_bad)}")
    print(f"  no_answer, stopped:                            {pct(sum(1 for r in stopped if r['label'] == 'no_answer'), n_na)}")
    wb = [r for r in stopped if r["sequential"]["writes_before_stop"]]
    if wb:
        print(f"  stopped after a write had already executed: {len(wb)} conversations: " +
              ", ".join(r["key"].split("|", 2)[2] for r in wb))
    head("3a. By demo (unit: conversations)")
    print(f"{'demo':6}{'design':>7}{'conv':>6}{'good kept':>11}{'good lost':>11}{'bad stopped':>13}{'bad passed':>12}{'n/a stopped':>13}")
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(SHORT[r["demo"]], r["design"])].append(r)
    for (d, des), rs in sorted(groups.items()):
        cell = collections.Counter()
        for r in rs:
            stop = r["sequential"]["outcome"] != "delivered"
            if r["label"] in good:
                cell["good lost" if stop else "good kept"] += 1
            elif r["label"] in bad:
                cell["bad stopped" if stop else "bad passed"] += 1
            elif r["label"] == "no_answer" and stop:
                cell["n/a stopped"] += 1
        print(f"{d:6}{des:>7}{len(rs):>6}{cell['good kept']:>11}{cell['good lost']:>11}{cell['bad stopped']:>13}"
              f"{cell['bad passed']:>12}{cell['n/a stopped']:>13}")
    head("3b. Correct conversations a guard would have stopped (each one)")
    for r in stopped:
        if r["label"] in good:
            print(f"  {r['key'].split('|', 2)[2]}: {r['sequential']['reason'][:140]}")


# --- 4. the cross-check with #29 -----------------------------------------------------------
def crosscheck(rows, convs) -> None:
    head("4. Cross-check with #29's hard checks (07, proposal phase; unit: issue_refund calls)")
    print("#29's hard checks ran in the loop (recorded verdict). This demo's guards are replayed on the same calls.")
    agree = collections.Counter()
    for r in rows:
        conv = convs[r["key"]]
        calls = by_index(conv)
        for v in r["independent"]["calls"]:
            c = calls[v["i"]]
            if c["tool"] != "issue_refund":
                continue
            hard = sources.hard_check_verdict(c)
            args = c["args"] or {}
            ids = guards.id_values(c)
            status = ("owned" if ids and ids[0][2] in conv["owned"]["invoice"] else
                      "other_customer" if ids and ids[0][2] in conv["other"]["invoice"] else "nonexistent")
            print(f"  {conv['question']} seed {conv['seed']}: invoice_id={args.get('invoice_id')!r:28} id {status:15} "
                  f"#29 {hard:16} provenance {'fires' if v['provenance'] else 'passes':7} "
                  f"id_format {'fires' if v['id_format'] else 'passes'}")
            agree[(hard.startswith("blocked"), bool(v["provenance"]))] += 1
    print("\n  #29 blocked and provenance fires:   ", agree[(True, True)])
    print("  #29 blocked, provenance passes:     ", agree[(True, False)])
    print("  #29 passed, provenance fires:       ", agree[(False, True)])
    print("  both passed:                        ", agree[(False, False)])
    head("4a. 07 reply checks against #29's false_claim labels (unit: proposal replies)")
    paused = [r for r in rows if not convs[r["key"]]["final"]]
    print(f"  {len(paused)} conversations paused for approval and have no reply in this phase; excluded here.")
    cm = collections.Counter()
    for r in rows:
        if r in paused or r["action_claim"] is None:
            continue
        cm[(r["action_claim"], bool(r["independent"]["reply"].get("unbacked_claim")))] += 1
        if r["action_claim"] or r["independent"]["reply"].get("unbacked_claim"):
            print(f"  {r['key'].split('|', 2)[2]}: false_claim={r['action_claim']}, unbacked_claim="
                  f"{r['independent']['reply'].get('unbacked_claim')!r}")
    pos = cm[(True, True)] + cm[(True, False)]
    neg = cm[(False, True)] + cm[(False, False)]
    print(f"  labelled false claim: fired on {pct(cm[(True, True)], pos)}; labelled not: fired on {pct(cm[(False, True)], neg)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["dev", "test", "crosscheck"], required=True)
    which = ap.parse_args().set
    rows, convs = load(which)
    frozen = HERE.parent / "data" / "FROZEN.json"
    print(f"08-failure-modes report, set: {which}")
    if frozen.exists():
        print(f"Frozen at {json.loads(frozen.read_text())['frozen_at_utc']} UTC (data/FROZEN.json). No model was called.")
    print("Evidence kinds: objective = computed from data files and recorded fields; hand = labels from the original\n"
          "articles, unchanged; replay = this demo's guard verdicts about recorded decisions.\n"
          "The replay cannot establish how the model would have recovered after a guard blocked it.")
    overview(rows, convs)
    if which == "crosscheck":
        crosscheck(rows, convs)
        return
    after_error(rows, convs)
    id_guards(rows, convs)
    repeat_guard(rows, convs)
    budget_guard(rows, convs)
    refund_guard(rows, convs)
    handoff_guard(rows, convs)
    reply_checks(rows, convs)
    sequential(rows, convs)


if __name__ == "__main__":
    main()
