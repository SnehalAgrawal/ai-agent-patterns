"""
Replays recorded agent decisions through the guards and saves every verdict.

    python replay.py --set dev          # development set (no labels; no freeze check)
    python replay.py --set test         # the frozen test set: 01-06
    python replay.py --set crosscheck   # 07, compared with #29's hard checks
    python replay.py --freeze           # write data/FROZEN.json (done once, before the first test run)

Writes results-<set>.jsonl: one line per conversation with its labels, every guard's
independent verdicts, and the sequential (first-fire) outcome. report.py turns that file
into report-<set>.txt. No model is called.
"""
import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import guards
import sources

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
FROZEN = DATA / "FROZEN.json"
CODE = [DATA / "evaluation.json", HERE / "guards.py", HERE / "sources.py", HERE / "replay.py", HERE / "report.py"]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(sources.ROOT))


def manifest() -> dict:
    return {"code": {_rel(p): file_hash(p) for p in CODE},
            "inputs": {_rel(p): file_hash(p) for p in sources.input_files()}}


def freeze() -> None:
    if FROZEN.exists():
        sys.exit(f"{FROZEN} already exists; the freeze is done once.")
    m = manifest()
    body = {"_comment": "Written by `python replay.py --freeze` before the first run on the test or cross-check "
                        "set. replay.py refuses to run on those sets if any file listed here has changed.",
            "frozen_at_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
            "guard_order": {"per_call": guards.CALL_ORDER, "on_specialist_reply": ["handoff_evidence"],
                            "on_final_reply": guards.REPLY_ORDER},
            "budget_K": guards.K, **m}
    FROZEN.write_text(json.dumps(body, indent=1) + "\n")
    print(f"froze {len(m['code'])} code files and {len(m['inputs'])} input files at {body['frozen_at_utc']} UTC")


def check_frozen() -> None:
    if not FROZEN.exists():
        sys.exit("data/FROZEN.json is missing: run `python replay.py --freeze` first.")
    frozen = json.loads(FROZEN.read_text())
    now = manifest()
    changed = [f for part in ("code", "inputs") for f, h in frozen[part].items() if now[part].get(f) != h]
    added = [f for part in ("code", "inputs") for f in now[part] if f not in frozen[part]]
    if changed or added:
        sys.exit("Changed since the freeze: " + ", ".join(changed + added))


def label(conv: dict) -> str | None:
    if conv["demo"] == "06-state-and-workflows":
        return None if conv["success_06"] is None else ("success" if conv["success_06"] else "failure")
    return conv["answer"]


def replay(which: str) -> Path:
    convs = sources.load(which)
    out = HERE / f"results-{which}.jsonl"
    with out.open("w") as f:
        for conv in convs:
            row = {"key": conv["key"], "demo": conv["demo"], "model": conv["model"], "design": conv["design"],
                   "question": conv["question"], "seed": conv["seed"], "label": label(conv),
                   "action_claim": conv["action_claim"], "reply_sha256": sources.sha256(conv["final"]),
                   "independent": guards.independent(conv), "sequential": guards.sequential(conv)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["dev", "test", "crosscheck"])
    ap.add_argument("--freeze", action="store_true")
    a = ap.parse_args()
    if a.freeze:
        freeze()
        return
    if not a.set:
        ap.error("--set or --freeze is required")
    if a.set != "dev":
        check_frozen()
    out = replay(a.set)
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
