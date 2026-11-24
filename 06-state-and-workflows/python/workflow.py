"""
Design S: a state machine around the model. The state is a case record owned by code,
saved to case.json in the run directory after every transition and before the refund
request is sent. The model is called at two nodes: intake (classify the message, JSON
schema) and reply (write the reply from facts code wrote). Verification, the refund,
and the handoff are code.

  intake ──► verify ──► refund ──► reply ──► done
     │          │  └──────────────► reply
     └──────────┴──────► handoff ──────────► done
                refund ─► handoff

Every transition goes through transition(), which checks the table and the guard.
A transition that isn't allowed raises InvalidTransition; run() records it and sends
the case to handoff.

Resume: load case.json and run its current node again from the start. A node must
therefore be safe to run twice; the refund node is, because it sends the same
idempotency key every time.
"""
import json
import os
import re
import time
from pathlib import Path

import llm
from payments import DATA, TODAY, PaymentService

DESIGNS = json.loads((DATA / "designs.json").read_text())
PROMPTS = json.loads((DATA / "prompts.json").read_text())
LIMITS = DESIGNS["limits"]
TRANSITIONS = {src: set(dsts) for src, dsts in DESIGNS["designs"]["S"]["transitions"].items()}
MAX_ATTEMPTS = 3
RETRY_WAITS = (0.2, 0.4)  # seconds before the 2nd and 3rd attempt
HANDOFF_REPLY = "Thanks for your message. I've passed it to our billing team. " + PROMPTS["handoff_message"]


class InvalidTransition(Exception):
    pass


# --- state ---------------------------------------------------------------------------------

def new_case(case_id: str, message: str) -> dict:
    return {"case_id": case_id, "message": message, "node": "intake",
            "request": None, "verification": None, "refund": None, "handoff_reason": None,
            "reply": None, "log": [], "invalid_transitions": [], "model_calls": [], "resumed": False}


def checkpoint(path: Path, case: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(case, indent=1, ensure_ascii=False))
    os.replace(tmp, path)


def _guard(case: dict, src: str, dst: str) -> str | None:
    """Why the transition src -> dst isn't allowed for this case, or None if it is."""
    if dst not in TRANSITIONS.get(src, set()):
        return f"no transition {src} -> {dst}"
    if (src, dst) == ("intake", "verify") and (case["request"] or {}).get("kind") != "duplicate_charge":
        return "intake -> verify needs a duplicate_charge request"
    if (src, dst) == ("verify", "refund"):
        v = case["verification"] or {}
        if v.get("outcome") != "refund" or not v.get("refund_payment_id"):
            return "verify -> refund needs exactly one duplicate invoice and a charge to refund"
        if v["refund_payment_id"] not in v.get("succeeded_charges", []) or v.get("refunded_by"):
            return "verify -> refund: the charge isn't an unrefunded successful charge of the duplicate invoice"
        if case["refund"] is not None:
            return "verify -> refund: this case already has a refund"
    if (src, dst) == ("refund", "reply") and (case["refund"] or {}).get("status") != "succeeded":
        return "refund -> reply needs a succeeded refund"
    return None


def transition(case: dict, dst: str) -> None:
    src = case["node"]
    problem = _guard(case, src, dst)
    if problem:
        raise InvalidTransition(problem)
    case["log"].append(f"{src} -> {dst}")
    case["node"] = dst


# --- nodes: each reads the case, updates it, and returns the next node -----------------------

def _model_call(case: dict, node: str, response: dict) -> None:
    case["model_calls"].append({k: response[k] for k in (
        "finish_reason", "input_tokens", "cached_tokens", "output_tokens", "seconds", "error")} | {"node": node})


def intake(case: dict, service: PaymentService, seed: int, path: Path) -> str:
    messages = [{"role": "system", "content": PROMPTS["intake_system"]}, {"role": "user", "content": case["message"]}]
    response = llm.chat(messages, [], seed, LIMITS["intake_max_tokens"], schema=PROMPTS["intake_schema"])
    _model_call(case, "intake", response)
    try:
        parsed = json.loads(response["content"])
        kind, invoice_id, month = parsed["kind"], str(parsed["invoice_id"]).strip(), str(parsed["month"]).strip()
    except (ValueError, KeyError, TypeError):
        case["request"] = {"raw": response["content"], "readable": False}
        case["handoff_reason"] = "the request could not be classified"
        return "handoff"
    if kind not in ("duplicate_charge", "other_refund", "other"):
        kind = "unreadable"
    if month and not re.fullmatch(r"\d{4}-\d{2}", month):
        month = ""  # an unusable month is treated as no month
    case["request"] = {"raw": response["content"], "readable": True, "kind": kind, "invoice_id": invoice_id,
                       "month": month}
    if kind != "duplicate_charge":
        case["handoff_reason"] = f"request kind {kind}: outside what the system may resolve on its own"
        return "handoff"
    return "verify"


def verify(case: dict, service: PaymentService, seed: int, path: Path) -> str:
    request = case["request"]
    invoices = service.invoices()
    payments = service.payments()
    by_id = {i["id"]: i for i in invoices}
    if request["invoice_id"] in by_id:
        scope, candidates = f"the invoice the customer named ({request['invoice_id']})", [by_id[request["invoice_id"]]]
    elif request["month"]:
        scope = f"the month the customer named ({request['month']})"
        candidates = [i for i in invoices if i["date"].startswith(request["month"])]
    else:
        scope, candidates = "the customer's account", invoices
    rows = []
    for inv in candidates:
        charges = [p for p in payments if p["invoice"] == inv["id"] and p["type"] == "charge"
                   and p["status"] == "succeeded"]
        refunds = [p for p in payments if p["type"] == "refund" and p.get("refund_of") in {c["id"] for c in charges}]
        rows.append({"invoice": inv, "charges": charges, "refunds": refunds})
    duplicates = [r for r in rows if len(r["charges"]) >= 2]
    v = {"scope": scope, "candidates": [r["invoice"]["id"] for r in rows], "rows": rows,
         "duplicate_invoice": None, "succeeded_charges": [], "refund_payment_id": None, "refunded_by": None}
    if len(duplicates) > 1:
        v["outcome"] = "ambiguous"
        case["verification"] = v
        case["handoff_reason"] = "more than one invoice was charged more than once"
        return "handoff"
    if not duplicates:
        v["outcome"] = "no_duplicate"
        case["verification"] = v
        return "reply"
    dup = duplicates[0]
    v["duplicate_invoice"] = dup["invoice"]["id"]
    v["succeeded_charges"] = [c["id"] for c in dup["charges"]]
    if dup["refunds"]:
        v["outcome"] = "already_refunded"
        v["refunded_by"] = dup["refunds"][0]["id"]
        case["verification"] = v
        return "reply"
    v["outcome"] = "refund"
    v["refund_payment_id"] = max(v["succeeded_charges"])  # the last successful charge is the extra one
    case["verification"] = v
    return "refund"


def refund(case: dict, service: PaymentService, seed: int, path: Path) -> str:
    if case["refund"] is None:
        payment_id = case["verification"]["refund_payment_id"]
        case["refund"] = {"payment_id": payment_id, "idempotency_key": f"{case['case_id']}:{payment_id}",
                          "status": "requested", "attempts": [], "result": None}
        checkpoint(path, case)  # record the intent before the side effect
    r = case["refund"]
    while True:
        result = service.refund(r["payment_id"], idempotency_key=r["idempotency_key"])
        r["attempts"].append({"result": result})
        if "error" not in result:
            r["status"], r["result"] = "succeeded", result
            checkpoint(path, case)
            return "reply"
        if result.get("retryable") and len(r["attempts"]) < MAX_ATTEMPTS:
            checkpoint(path, case)
            time.sleep(RETRY_WAITS[len(r["attempts"]) - 1])
            continue
        r["status"] = "failed"
        case["handoff_reason"] = f"the refund failed: {result['error']}"
        checkpoint(path, case)
        return "handoff"


def facts(case: dict) -> str:
    """The case facts the reply model gets, written by code from the record."""
    v = case["verification"]

    def describe(row: dict) -> str:
        inv = row["invoice"]
        charges = ", ".join(f"{c['id']} on {c['date']}" for c in row["charges"]) or "none"
        n = len(row["charges"])
        times = {0: "never charged successfully", 1: "charged successfully once", 2: "charged successfully twice"}
        return (f"- Invoice {inv['id']} ({inv['description']}, ${inv['amount']}, dated {inv['date']}) was "
                f"{times.get(n, f'charged successfully {n} times')}: {charges}.")

    if v["outcome"] == "refund":
        row = next(r for r in v["rows"] if r["invoice"]["id"] == v["duplicate_invoice"])
        res = case["refund"]["result"]
        return (describe(row) + f"\n- The system refunded the extra charge {res['refund_of']} (${res['amount']}) "
                f"today, {TODAY}, as refund {res['refund_id']}. It goes back to the original payment method "
                "within 5 business days.")
    if v["outcome"] == "already_refunded":
        row = next(r for r in v["rows"] if r["invoice"]["id"] == v["duplicate_invoice"])
        earlier = row["refunds"][0]
        return (describe(row) + f"\n- The extra charge {earlier['refund_of']} was already refunded on "
                f"{earlier['date']} as refund {earlier['id']}. No new refund was made, and nothing more is owed "
                "back for this invoice.")
    lines = [describe(r) for r in v["rows"]] or ["- No invoice was found."]
    return "\n".join(lines + [f"- No invoice in {v['scope']} was charged more than once, so there is no duplicate "
                              "charge to refund. No refund was made."])


def reply(case: dict, service: PaymentService, seed: int, path: Path) -> str:
    system = PROMPTS["reply_system"].format(system=PROMPTS["system"], facts=facts(case))
    messages = [{"role": "system", "content": system}, {"role": "user", "content": case["message"]}]
    response = llm.chat(messages, [], seed, LIMITS["max_tokens_per_call"])
    _model_call(case, "reply", response)
    case["reply"] = response["content"] if not response["error"] else None
    return "done"


def handoff(case: dict, service: PaymentService, seed: int, path: Path) -> str:
    if not any(h["by"] == case["case_id"] for h in service.load()["handoffs"]):  # safe to run twice
        service.hand_off(case["handoff_reason"] or "", by=case["case_id"])
    case["reply"] = HANDOFF_REPLY
    return "done"


NODES = {"intake": intake, "verify": verify, "refund": refund, "reply": reply, "handoff": handoff}


def run(run_dir: Path, case_id: str, message: str, seed: int, resume: bool = False,
        service: PaymentService | None = None) -> dict:
    path = Path(run_dir) / "case.json"
    service = service or PaymentService(Path(run_dir) / "ledger.json")
    if resume:
        case = json.loads(path.read_text())
        case["resumed"] = True
        case["log"].append(f"resumed at {case['node']}")
    else:
        case = new_case(case_id, message)
    checkpoint(path, case)
    while case["node"] != "done":
        node = case["node"]
        nxt = NODES[node](case, service, seed, path)
        try:
            transition(case, nxt)
        except InvalidTransition as exc:
            case["invalid_transitions"].append({"from": node, "to": nxt, "problem": str(exc)})
            case["handoff_reason"] = f"invalid transition: {exc}"
            case["log"].append(f"{node} -> handoff (invalid transition to {nxt})")
            case["node"] = "handoff"
        checkpoint(path, case)
    return case
