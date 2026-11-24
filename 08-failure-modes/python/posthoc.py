"""
Post-hoc analyses: written after the frozen test run (data/FROZEN.json), not part of the
frozen protocol, and reported separately from it. No model is called.

    python posthoc.py > report-posthoc.txt

1. The repeat guard with a different policy. The frozen replay ends the run when a guard fires.
   For `repeat`, a natural alternative is to hand the model the earlier result again. Every one
   of the 97 repeats in the test set got a byte-identical result, so under that policy the model
   would have received exactly the bytes it received in the recording, and the recorded run is
   what that policy would have produced (assuming, as the repeat runs of 06 and 07 showed in
   their setups, that the same input gives the same output). This is the one place where the
   replay can say what happens after a guard fires; it is checked, not assumed, below.
2. provenance and id_format together: which invented ids each one catches that the other misses.
"""
import collections
import json
from pathlib import Path

import guards
import sources

HERE = Path(__file__).resolve().parent


def label_kind(lab):
    return "good" if lab in ("correct", "success") else "bad" if lab in ("incorrect", "failure") else lab


def main() -> None:
    rows = [json.loads(line) for line in (HERE / "results-test.jsonl").read_text().splitlines()]
    convs = {c["key"]: c for c in sources.load("test")}
    print("08-failure-modes: post-hoc analyses (not part of the frozen protocol). No model was called.")

    # 1. repeat -> serve the cached result
    print("\n1. repeat as 'serve the earlier result' instead of 'end the run' (unit: conversations)")
    redundant = all(
        convs[r["key"]]["calls"][v["i"]]["result_text"] ==
        guards.repeated_from(convs[r["key"]], convs[r["key"]]["calls"][v["i"]])["result_text"]
        for r in rows for v in r["independent"]["calls"] if v["repeat"])
    print(f"   every repeat got a byte-identical result: {redundant}")
    if not redundant:
        print("   -> the substitution is not valid; stopping.")
        return
    frozen_repeat = guards.CALL_GUARDS["repeat"]
    guards.CALL_GUARDS["repeat"] = lambda conv, call: None     # serve cached: the recorded run continues
    try:
        variant = {r["key"]: guards.sequential(convs[r["key"]]) for r in rows}
    finally:
        guards.CALL_GUARDS["repeat"] = frozen_repeat
    t_frozen, t_variant = collections.Counter(), collections.Counter()
    for r in rows:
        k = label_kind(r["label"])
        t_frozen[(k, r["sequential"]["outcome"] != "delivered")] += 1
        t_variant[(k, variant[r["key"]]["outcome"] != "delivered")] += 1
    print(f"   {'':44}{'frozen (end the run)':>22}{'serve the earlier result':>26}")
    for k, name in (("good", "correct (or 06 success)"), ("bad", "incorrect (or 06 failure)"), ("no_answer", "no_answer")):
        n = t_frozen[(k, True)] + t_frozen[(k, False)]
        print(f"   {name + ', stopped (of ' + str(n) + ')':44}{t_frozen[(k, True)]:>22}{t_variant[(k, True)]:>26}")
    moved = collections.Counter(variant[r["key"]]["guard"] or "delivered"
                                for r in rows if r["sequential"]["guard"] == "repeat")
    print("   the 27 conversations the frozen replay stopped at 'repeat', under the variant: " +
          ", ".join(f"{g} {n}" for g, n in sorted(moved.items())))
    calls_saved = sum(1 for r in rows for v in r["independent"]["calls"] if v["repeat"])
    print(f"   tool executions the cache would have saved: {calls_saved}; model calls saved: 0 "
          "(the model still asks for the call and still reads the result)")

    # 2. provenance and id_format together
    print("\n2. provenance and id_format on invented ids (unit: id arguments whose value doesn't exist)")
    both = collections.Counter()
    only = collections.defaultdict(list)
    for r in rows:
        conv = convs[r["key"]]
        for v in r["independent"]["calls"]:
            c = conv["calls"][v["i"]]
            for arg, kind, val in guards.id_values(c):
                if val in conv["owned"][kind] or val in conv["other"][kind]:
                    continue
                p, f = bool(v["provenance"]), bool(v["id_format"])
                both[(p, f)] += 1
                if p != f:
                    only["provenance only" if p else "id_format only"].append(f"{arg}={val!r}")
    n = sum(both.values())
    print(f"   caught by both {both[(True, True)]}, provenance only {both[(True, False)]}, "
          f"id_format only {both[(False, True)]}, neither {both[(False, False)]} (of {n})")
    for k, vals in sorted(only.items()):
        print(f"   {k}: " + ", ".join(sorted(set(vals))) + f" ({len(vals)} arguments)")


if __name__ == "__main__":
    main()
