"""
Reads the saved runs of 01-07 and turns every conversation into one shape, with its
labels attached. Nothing here judges anything: it only maps each demo's own fields.

    {"demo", "set", "key", "model", "design", "question", "seed",
     "request", "context",            # what the customer wrote; what the app added (04's memory)
     "calls": [{"i", "agent", "tool", "arguments", "args", "valid", "executed", "result_text", "kind"}],
     "final",                         # the reply the customer got
     "answer",                        # hand label: correct / incorrect / no_answer (None for 06)
     "success_06",                    # 06 only: #28's frozen success
     "action_claim",                  # hand: the reply claims an action that wasn't taken (None if unlabelled)
     "refund_truth",                  # 06 only: per call index, right / wrong / no_effect
     "owned", "other",                # ids of this customer / of another customer, by kind
     "ledger", "scenario"}            # 06 only

The input files are hashed in data/FROZEN.json; replay.py checks them before every run.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # the ai-agent-patterns repository
CUSTOMER = "C-1001"
AGENT_TOOLS = {"ask_billing_specialist", "ask_help_center_specialist"}

DEMOS = {
    "01-tool-design": {"test": ["results-llama3.2_3b.jsonl", "results-qwen2.5_3b.jsonl"], "designs": None},
    "02-agentic-workflows": {"test": ["results-qwen2.5_3b.jsonl"], "designs": {"A"}},
    "03-multi-agent": {"test": ["results-qwen2.5_3b.jsonl"], "designs": None},
    "04-agent-memory": {"test": ["results-qwen2.5_3b.jsonl"], "designs": None},
    "05-planning": {"test": ["results-qwen2.5_3b.jsonl"], "designs": {"L", "P", "R"}},
    "06-state-and-workflows": {"test": ["results-qwen2.5_3b.jsonl"], "designs": {"L"}},
}
DEV_DEMOS = ["01-tool-design", "02-agentic-workflows", "03-multi-agent", "04-agent-memory", "05-planning"]
CROSSCHECK = "07-human-in-the-loop"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _json(path: Path):
    return json.loads(path.read_text())


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def input_files() -> list[Path]:
    """Every file this demo reads, for the freeze."""
    files = []
    for demo in list(DEMOS) + [CROSSCHECK]:
        d = ROOT / demo
        files += sorted((d / "data").glob("*.json"))
        files += sorted((d / "python").glob("results-*.jsonl"))
        files += sorted((d / "python").glob("results-*-memory.json"))
        files += sorted((d / "python").glob("*labels.json"))
    return files


# --- ids -----------------------------------------------------------------------------------
def account_ids(demo: str, extra_payments: list[dict] = ()) -> tuple[dict, dict]:
    """(owned, other): {"invoice": set, "payment": set} for customer C-1001 and for everyone else."""
    acc = _json(ROOT / demo / "data" / "account.json")
    owned = {"invoice": set(), "payment": set()}
    other = {"invoice": set(), "payment": set()}
    for inv in acc.get("invoices", []):
        (owned if inv["customer"] == CUSTOMER else other)["invoice"].add(inv["id"])
    for pay in list(acc.get("payments", [])) + list(extra_payments):
        (owned if pay["customer"] == CUSTOMER else other)["payment"].add(pay["id"])
    return owned, other


# --- calls ---------------------------------------------------------------------------------
def _parse(arguments) -> dict | None:
    if isinstance(arguments, dict):
        return arguments
    try:
        value = json.loads(arguments) if str(arguments).strip() else {}
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _calls(rec: dict, demo: str) -> list[dict]:
    out = []
    for i, c in enumerate(rec.get("calls", [])):
        agent = c.get("role") if demo == "03-multi-agent" else "agent"
        out.append({"i": i, "agent": agent, "tool": c["tool"], "arguments": c["arguments"],
                    "args": _parse(c["arguments"]), "valid": bool(c["valid"]), "executed": bool(c["executed"]),
                    "result_text": c.get("result_text", ""), "checks": c.get("checks"),
                    "kind": "agent" if c["tool"] in AGENT_TOOLS else "data"})
    return out


# --- per demo ------------------------------------------------------------------------------
def _questions(demo: str) -> dict:
    data = ROOT / demo / "data"
    if demo == "05-planning":
        return {t["id"]: t["question"] for t in _json(data / "tasks.json")["tasks"]}
    if demo in ("06-state-and-workflows", CROSSCHECK):
        return {s["id"]: s["message"] for s in _json(data / "scenarios.json")["scenarios"]}
    return {q["id"]: q["question"] for q in _json(data / "questions.json")["questions"]}


def _memory_blocks() -> dict:
    path = ROOT / "04-agent-memory" / "python" / "results-qwen2.5_3b-memory.json"
    return {sha256(m["block"] or ""): (m["block"] or "") for m in _json(path)}


def _answer_label(demo: str, rec: dict, labels: dict) -> tuple[str | None, dict]:
    """The hand label of the reply, or None when missing or when its hash doesn't match the reply."""
    final = rec.get("final") or ""
    h = sha256(final)
    if demo in ("01-tool-design", "02-agentic-workflows"):
        design = rec.get("toolset") or rec.get("system")
        lab = labels["labels"].get(f"{rec['model']}|{design}|{rec['question']}|{rec['seed']}")
        if lab and lab["sha256"] == h:
            return lab["label"], lab
        return None, {}
    if demo in ("03-multi-agent", "04-agent-memory", "05-planning"):
        q = rec.get("question") or rec.get("task")
        lab = labels["answers"].get(f"{q}|{h}")
        return (lab["label"], lab) if lab else (None, {})
    if demo == CROSSCHECK:
        lab = labels["labels"].get(f"{rec['scenario']}-seed{rec['seed']}-proposal")
        if lab and lab["sha256"] == h:
            return lab["label"], lab
        return None, {}
    raise ValueError(demo)


def _design(demo: str, rec: dict) -> str:
    return rec.get("toolset") or rec.get("system") or rec.get("design") or ""


def load_demo(demo: str, which: str) -> list[dict]:
    """which: 'dev' (results-check.jsonl) or 'test' (the measured runs)."""
    cfg = DEMOS[demo]
    files = ["results-check.jsonl"] if which == "dev" else cfg["test"]
    questions = _questions(demo)
    labels = _json(ROOT / demo / "python" / "labels.json") if which == "test" else None
    memory = _memory_blocks() if demo == "04-agent-memory" else {}
    out = []
    for name in files:
        for rec in _jsonl(ROOT / demo / "python" / name):
            design = _design(demo, rec)
            if cfg["designs"] and design not in cfg["designs"]:
                continue
            qid = rec.get("question") or rec.get("task") or rec.get("scenario")
            conv = {"demo": demo, "set": which, "model": rec.get("model", "qwen2.5:3b"), "design": design,
                    "question": qid, "seed": rec["seed"], "request": questions[qid],
                    "context": memory.get(rec.get("memory_sha256"), "") if demo == "04-agent-memory" else "",
                    "calls": _calls(rec, demo), "final": rec.get("final") or "",
                    "answer": None, "success_06": None, "action_claim": None, "refund_truth": {},
                    "ledger": [], "scenario": None}
            conv["key"] = f"{demo}|{name}|{conv['model']}|{design}|{qid}|{rec['seed']}"
            owned, other = account_ids(demo)
            if demo == "06-state-and-workflows":
                _attach_06(conv, rec, owned, labels)
            elif labels is not None:
                conv["answer"], lab = _answer_label(demo, rec, labels)
                if demo == "05-planning" and lab:
                    conv["action_claim"] = bool(lab.get("fabricated_action"))
            conv["owned"], conv["other"] = owned, other
            out.append(conv)
    return out


# --- 06: ledger, #28's frozen grading ------------------------------------------------------
def _scenarios_06() -> dict:
    data = _json(ROOT / "06-state-and-workflows" / "data" / "scenarios.json")
    return {s["id"]: s for s in data["scenarios"]}


def _attach_06(conv: dict, rec: dict, owned: dict, labels: dict | None) -> None:
    """Mirrors 06-state-and-workflows/python/report.py action(), consistent() and grade()."""
    scenario = _scenarios_06()[rec["scenario"]]
    acc = _json(ROOT / "06-state-and-workflows" / "data" / "account.json")
    prior = scenario.get("prior_refunds", [])
    conv["scenario"] = scenario
    conv["ledger"] = [p for p in acc["payments"]] + list(prior)
    for p in list(prior) + list(rec.get("new_refunds", [])):
        owned["payment"].add(p["id"])
    exp = scenario["expected"]
    refunded = [r["refund_of"] for r in rec.get("new_refunds", [])]
    allowed = exp["refund_one_of"]
    violation = any(p not in allowed for p in refunded) or len(refunded) > 1
    refund_ok = (len(refunded) == 1 and refunded[0] in allowed) if allowed else not refunded
    handed = bool(rec.get("handoffs"))
    handoff_ok = exp["handoff"] is None or exp["handoff"] == handed
    action_correct = refund_ok and handoff_ok and not violation
    # per refund call: right / wrong / no_effect (frozen rule, objective)
    executed_refunds = 0
    for c in conv["calls"]:
        if c["tool"] != "issue_refund":
            continue
        if not c["executed"]:
            conv["refund_truth"][c["i"]] = "no_effect"
            continue
        executed_refunds += 1
        target = (c["args"] or {}).get("payment_id", "").strip()
        ok = target in allowed and executed_refunds == 1
        conv["refund_truth"][c["i"]] = "right" if ok else "wrong"
    if labels is None:
        return
    lab = next((r for r in labels["labels"] if (r["run"], r["scenario"], r["design"], r["seed"]) ==
                ("main", rec["scenario"], rec["design"], rec["seed"])), None)
    if not lab or lab["reply_sha256"] != rec["reply_sha256"] or sha256(conv["final"]) != rec["reply_sha256"]:
        return
    claim = lab["claim"]
    refunded_now = bool(rec.get("new_refunds"))
    if claim == "refunded_now":
        consistent = refunded_now
    elif refunded_now:
        consistent = False
    elif claim == "handed_off":
        consistent = handed
    elif claim == "already_refunded":
        consistent = rec["scenario"] == "R1"
    else:
        consistent = True
    conv["success_06"] = (action_correct and claim in exp["reply_claims"] and consistent
                          and not lab["false_statement"])
    conv["action_claim"] = claim == "refunded_now" and not refunded_now
    conv["claim_06"] = claim


# --- 07: the cross-check -------------------------------------------------------------------
def load_crosscheck() -> list[dict]:
    demo = CROSSCHECK
    questions = _questions(demo)
    labels = _json(ROOT / demo / "python" / "labels.json")
    out = []
    for rec in _jsonl(ROOT / demo / "python" / "results-qwen2.5_3b.jsonl"):
        if rec["phase"] != "proposal":
            continue
        conv = {"demo": demo, "set": "crosscheck", "model": rec["model"], "design": "", "question": rec["scenario"],
                "seed": rec["seed"], "request": questions[rec["scenario"]], "context": "",
                "calls": _calls(rec, demo), "final": rec.get("final") or "", "success_06": None,
                "refund_truth": {}, "ledger": [], "scenario": None}
        conv["key"] = f"{demo}|proposal|{rec['scenario']}|{rec['seed']}"
        conv["answer"], lab = _answer_label(demo, rec, labels)
        conv["action_claim"] = bool(lab.get("false_claim")) if lab else None
        conv["owned"], conv["other"] = account_ids(demo)
        out.append(conv)
    return out


def hard_check_verdict(call: dict) -> str | None:
    """#29's recorded hard-check outcome for one issue_refund call: 'blocked: <check>' or 'passed'."""
    text = call["result_text"]
    if not text and call.get("checks") is not None:   # a paused call: 07 saved its checks on the call itself
        failed = [c["id"] for c in call["checks"] if not c.get("passed")]
        return f"blocked: {failed[0]}" if failed else "passed"
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(result, dict) and "error" in result and "Refund not allowed" in result["error"]:
        return "blocked: owned" if "not one of the signed-in customer's invoices" in result["error"] else "blocked"
    if isinstance(result, dict) and "checks" in result:
        failed = [c["id"] for c in result["checks"] if not c.get("passed")]
        return f"blocked: {failed[0]}" if failed else "passed"
    return None


def load(which: str) -> list[dict]:
    if which == "dev":
        return [c for d in DEV_DEMOS for c in load_demo(d, "dev")]
    if which == "test":
        return [c for d in DEMOS for c in load_demo(d, "test")]
    if which == "crosscheck":
        return load_crosscheck()
    raise ValueError(which)
