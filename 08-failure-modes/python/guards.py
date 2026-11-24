"""
The guards, as data/evaluation.json defines them, and the two ways this demo applies them
to a recorded conversation:

  independent(conv)  every guard's verdict on every call or reply it applies to, given the
                     recorded history, as if it were the only new guard
  sequential(conv)   the frozen order; the first guard that fires ends the run (fail closed)

A guard returns None when the call may run, or a short reason when it may not. Guards see
only what the application would have seen at that moment: the request, the app's own
context, and the calls made before this one with their results. They never see labels.

No model is called. These are judgments about recorded decisions, not observations of a
system that ran with the guards: what the model would have done after being blocked can't
be replayed.
"""
import json
import re

K = 8                                     # budget: tool calls per conversation (frozen)
CUSTOMER = "C-1001"
WRITE_TOOLS = {"issue_refund", "hand_off_to_billing", "remove_plan_seat"}
ID_ARGUMENTS = {"invoice_id": "invoice", "payment_id": "payment"}
ID_FORMATS = {"invoice": re.compile(r"^INV-[0-9]{4}$"), "payment": re.compile(r"^PAY-[0-9]{4}$")}
SPECIALIST = {"ask_billing_specialist": "specialist:billing", "ask_help_center_specialist": "specialist:help_center"}
TOKEN = re.compile(r"[A-Za-z0-9_-]+")

# A completed refund or account change, claimed in the first person or about "your refund".
CLAIM_PATTERNS = [re.compile(p, re.IGNORECASE) for p in (
    r"\b(?:I|we)(?:['’]ve| have| had)?\s+(?:now\s+|just\s+|already\s+|successfully\s+|also\s+)*"
    r"(?:issued|processed|initiated|refunded|removed|cancel+ed|submitted|credited|reversed)\b",
    r"\b(?:I|we)(?:['’]ve| have)\s+gone ahead and\b",
    r"\byour refund (?:has been|was|is now)\s+(?:successfully\s+)?(?:issued|processed|initiated|approved|completed)\b",
    r"\bthe refund (?:has been|was)\s+(?:successfully\s+)?(?:issued|processed|initiated)\b",
    r"\b(?:seats?|charge) (?:has|have) been (?:successfully\s+)?(?:removed|refunded|reversed)\b",
)]
TOOL_CALL_TEXT = [re.compile(r'"name"\s*:\s*"[^"]*"\s*,\s*"(?:arguments|parameters)"\s*:'),
                  re.compile(r"<tool_call>", re.IGNORECASE)]

CALL_ORDER = ["budget", "provenance", "repeat", "refund_policy"]   # contract (recorded) sits after budget
REPLY_ORDER = ["empty_reply", "tool_call_text", "unbacked_claim", "no_evidence"]


# --- time order ----------------------------------------------------------------------------
def timeline(conv: dict) -> list[tuple[str, dict]]:
    """Events in the order they happened. A manager's ask_* call is recorded when the specialist's
    reply returns, after the specialist's own calls; here its request is moved before them."""
    calls = conv["calls"]
    events: list[tuple[str, dict]] = []
    i = 0
    pending: list[dict] = []
    while i < len(calls):
        c = calls[i]
        if c["tool"] in SPECIALIST:
            role = SPECIALIST[c["tool"]]
            block = []
            while pending and pending[-1]["agent"] == role:
                block.insert(0, pending.pop())
            events += [("call", p) for p in pending]
            pending = []
            events += [("call", c)] + [("call", b) for b in block] + [("reply", c)]
        else:
            pending.append(c)
        i += 1
    events += [("call", p) for p in pending]
    return events


def _before(conv: dict, call: dict) -> list[dict]:
    """Calls requested before this one, in time order."""
    out = []
    for kind, c in timeline(conv):
        if kind != "call":
            continue
        if c is call:
            return out
        out.append(c)
    raise ValueError("call not in conversation")


def _success(c: dict) -> bool:
    return c["kind"] == "data" and c["executed"]


def _canonical(c: dict) -> tuple | None:
    if c["args"] is None:
        return None
    return (c["agent"], c["tool"], json.dumps(c["args"], sort_keys=True))


def id_values(call: dict) -> list[tuple[str, str, str]]:
    """[(argument, kind, value)] for every string id argument."""
    args = call["args"] or {}
    return [(a, kind, args[a].strip()) for a, kind in ID_ARGUMENTS.items() if isinstance(args.get(a), str)]


def known_tokens(conv: dict, earlier: list[dict]) -> set[str]:
    text = [conv["request"], conv["context"]] + [c["result_text"] for c in earlier if _success(c)]
    return set(TOKEN.findall("\n".join(text)))


# --- call guards ---------------------------------------------------------------------------
def budget(conv: dict, call: dict, k: int = K) -> str | None:
    n = len(_before(conv, call)) + 1
    return f"budget: tool call {n} of a {k}-call budget" if n > k else None


def provenance(conv: dict, call: dict) -> str | None:
    ids = id_values(call)
    if not ids:
        return None
    known = known_tokens(conv, _before(conv, call))
    for arg, _, value in ids:
        if value not in known:
            return f"provenance: {arg}={value!r} is not in the request, the app's context, or an earlier successful result"
    return None


def id_format(conv: dict, call: dict) -> str | None:
    for arg, kind, value in id_values(call):
        if not ID_FORMATS[kind].match(value):
            return f"id_format: {arg}={value!r} is not a valid {kind} id"
    return None


def repeat(conv: dict, call: dict) -> str | None:
    key = _canonical(call)
    if key is None:
        return None
    earlier = _before(conv, call)
    for j in range(len(earlier) - 1, -1, -1):
        e = earlier[j]
        if e["tool"] in WRITE_TOOLS and e["executed"]:
            return None                     # something changed since: a repeat may see new data
        if _canonical(e) == key and (e["executed"] if e["kind"] == "agent" else _success(e)):
            return f"repeat: the same {call['tool']} call already succeeded (call {e['i']}); reuse its result"
    return None


def repeated_from(conv: dict, call: dict) -> dict | None:
    """The earlier successful call that `repeat` matched, for the ground truth."""
    key = _canonical(call)
    for e in reversed(_before(conv, call)):
        if e["tool"] in WRITE_TOOLS and e["executed"]:
            return None
        if _canonical(e) == key and (e["executed"] if e["kind"] == "agent" else _success(e)):
            return e
    return None


def _ledger(conv: dict, earlier: list[dict]) -> list[dict]:
    rows = [dict(p) for p in conv["ledger"]]
    for c in earlier:
        if c["tool"] == "issue_refund" and c["executed"]:
            r = json.loads(c["result_text"])
            if not isinstance(r, dict) or "refund_of" not in r:
                raise ValueError(f"executed issue_refund without a refund result: {c['result_text'][:80]}")
            charge = next(p for p in rows if p["id"] == r["refund_of"])
            rows.append({"id": r["refund_id"], "customer": CUSTOMER, "type": "refund", "status": "succeeded",
                         "invoice": charge["invoice"], "amount": r["amount"], "refund_of": r["refund_of"]})
    return rows


def refund_policy(conv: dict, call: dict) -> str | None:
    if call["tool"] != "issue_refund" or conv["demo"] != "06-state-and-workflows":
        return None
    pid = (call["args"] or {}).get("payment_id", "")
    pid = pid.strip() if isinstance(pid, str) else ""
    earlier = _before(conv, call)
    rows = _ledger(conv, earlier)
    p = next((r for r in rows if r["id"] == pid), None)
    if p is None or p["customer"] != CUSTOMER:
        return f"refund_policy.exists: {pid!r} is not a payment of this customer"
    if p["type"] != "charge":
        return f"refund_policy.is_charge: {pid} is a {p['type']}"
    if p["status"] != "succeeded":
        return f"refund_policy.succeeded: {pid} is {p['status']}"
    if any(r.get("refund_of") == pid for r in rows):
        return f"refund_policy.not_refunded: {pid} was already refunded"
    inv = p["invoice"]
    charges = sum(1 for r in rows if r["invoice"] == inv and r["type"] == "charge" and r["status"] == "succeeded")
    refunds = sum(1 for r in rows if r["invoice"] == inv and r["type"] == "refund" and r["status"] == "succeeded")
    if charges - refunds < 2:
        return f"refund_policy.is_extra_charge: {inv} has {charges} succeeded charge(s) and {refunds} refund(s); no duplicate left"
    if any(c["tool"] == "issue_refund" and c["executed"] for c in earlier):
        return "refund_policy.one_per_conversation: a refund was already issued in this conversation"
    return None


def handoff_evidence(conv: dict, ask: dict) -> str | None:
    """At the moment a specialist's reply returns to the manager."""
    if ask["tool"] not in SPECIALIST:
        return None
    role = SPECIALIST[ask["tool"]]
    block, started = [], False
    for kind, c in timeline(conv):
        if c is ask and kind == "call":
            started = True
            continue
        if c is ask and kind == "reply":
            break
        if started and c["agent"] == role:
            block.append(c)
    data_calls = [c for c in block if c["kind"] == "data"]
    if not data_calls:
        return "handoff_evidence: no_lookup (the specialist replied without calling a data tool)"
    if not any(_success(c) for c in data_calls):
        return f"handoff_evidence: lookup_failed (all {len(data_calls)} of the specialist's data calls failed)"
    return None


CALL_GUARDS = {"budget": budget, "provenance": provenance, "id_format": id_format, "repeat": repeat,
               "refund_policy": refund_policy}


# --- reply checks --------------------------------------------------------------------------
def empty_reply(conv: dict) -> str | None:
    return "empty_reply" if not conv["final"].strip() else None


def tool_call_text(conv: dict) -> str | None:
    return "tool_call_text" if any(p.search(conv["final"]) for p in TOOL_CALL_TEXT) else None


def claimed_action(text: str) -> str | None:
    for p in CLAIM_PATTERNS:
        m = p.search(text)
        if m:
            return m.group(0)
    return None


def unbacked_claim(conv: dict) -> str | None:
    if any(c["tool"] in WRITE_TOOLS and c["executed"] for c in conv["calls"]):
        return None
    claim = claimed_action(conv["final"])
    return f"unbacked_claim: {claim!r} with no executed write" if claim else None


def no_evidence(conv: dict) -> str | None:
    if conv["context"] or any(_success(c) for c in conv["calls"]):
        return None
    return "no_evidence: no successful tool result and no app context"


REPLY_CHECKS = {"empty_reply": empty_reply, "tool_call_text": tool_call_text, "unbacked_claim": unbacked_claim,
                "no_evidence": no_evidence}


# --- applying them -------------------------------------------------------------------------
def independent(conv: dict) -> dict:
    """{"calls": [{"i", <guard>: reason|None ...}], "handoffs": [{"i", "reason"}], "reply": {<check>: reason|None}}"""
    calls = []
    for kind, c in timeline(conv):
        if kind != "call":
            continue
        calls.append({"i": c["i"], **{name: fn(conv, c) for name, fn in CALL_GUARDS.items()}})
    handoffs = [{"i": c["i"], "reason": handoff_evidence(conv, c)} for kind, c in timeline(conv) if kind == "reply"]
    reply = {name: fn(conv) for name, fn in REPLY_CHECKS.items()} if conv["final"] is not None else {}
    return {"calls": calls, "handoffs": handoffs, "reply": reply}


def sequential(conv: dict) -> dict:
    """The frozen order, first fire ends the run.
    Returns {"outcome": "stopped_at_call" | "reply_withheld" | "delivered", "guard", "reason", "i",
             "writes_before_stop": [call indexes of writes executed before the stop]}."""
    writes = []
    for kind, c in timeline(conv):
        if kind == "reply":
            reason = handoff_evidence(conv, c)
            if reason:
                return {"outcome": "stopped_at_call", "guard": "handoff_evidence", "reason": reason, "i": c["i"],
                        "writes_before_stop": writes}
            continue
        reason = budget(conv, c)
        if reason:
            return {"outcome": "stopped_at_call", "guard": "budget", "reason": reason, "i": c["i"],
                    "writes_before_stop": writes}
        if not c["valid"]:
            continue                        # baseline contract: the recorded error went back to the model
        for name in ("provenance", "repeat", "refund_policy"):
            reason = CALL_GUARDS[name](conv, c)
            if reason:
                return {"outcome": "stopped_at_call", "guard": name, "reason": reason, "i": c["i"],
                        "writes_before_stop": writes}
        if c["tool"] in WRITE_TOOLS and c["executed"]:
            writes.append(c["i"])
    for name in REPLY_ORDER:
        reason = REPLY_CHECKS[name](conv)
        if reason:
            return {"outcome": "reply_withheld", "guard": name, "reason": reason, "i": None,
                    "writes_before_stop": writes}
    return {"outcome": "delivered", "guard": None, "reason": None, "i": None, "writes_before_stop": writes}
