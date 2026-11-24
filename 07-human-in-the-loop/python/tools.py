"""
The five read-only tools from 02-agentic-workflows and 04-agent-memory, plus the schema of
the one action tool, issue_refund (defined in data/policy.json).

The read tools are the same functions with the same descriptions, with one change: they
also see the refunds this system has issued (passed in as `issued`), so that after an
approved refund, get_payments and get_refund_status show it. issue_refund is never run
from here: agent.py turns a call to it into an approval request.
"""
import json
import re
from datetime import date, timedelta
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
ACCOUNT = json.loads((DATA / "account.json").read_text())
DOCUMENTS = json.loads((DATA / "help_center.json").read_text())["documents"]
POLICY = json.loads((DATA / "policy.json").read_text())
TODAY = date.fromisoformat(ACCOUNT["today"])
WINDOW = ACCOUNT["refund_window_days"]
CUSTOMER = ACCOUNT["customer"]["id"]
INVOICE_FIELDS = ("id", "date", "amount", "description", "status")
PAYMENT_FIELDS = ("id", "date", "type", "amount", "invoice", "status", "note")

STOPWORDS = {"a", "an", "and", "are", "can", "do", "does", "for", "how", "i", "if", "in", "is", "it",
             "my", "of", "on", "or", "the", "to", "what", "when", "where", "which", "with", "you", "your"}


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required,
                       "additionalProperties": False}}}


READ_TOOLS = [
    _tool("search_help_center", "Search the help center articles. Returns the three most relevant articles.",
          {"query": {"type": "string", "description": "What to search for."}}, ["query"]),
    _tool("get_invoices", "List the signed-in customer's invoices.", {}, []),
    _tool("get_refund_status",
          "For each of the signed-in customer's invoices: whether it can be refunded today, why, and its refund "
          "deadline.",
          {"invoice_id": {"type": "string",
                          "description": "Optional. One invoice id from get_invoices. Leave it out to get every invoice."}},
          []),
    _tool("get_payments",
          "List the signed-in customer's card payments and refunds: date, amount, invoice, and whether each succeeded.",
          {}, []),
    _tool("get_plan_history", "List the changes to the signed-in customer's plan, with dates.", {}, []),
]
REFUND_TOOL = POLICY["refund_tool"]
TOOLS = READ_TOOLS + [REFUND_TOOL]
BY_NAME = {t["function"]["name"]: t for t in TOOLS}


def _terms(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if w not in STOPWORDS}


def _search(query: str) -> list[dict]:
    """Keyword search, as in Articles #8 and #23: the three articles sharing the most terms (ties keep file order)."""
    query_terms = _terms(query)
    scored = [(len(query_terms & _terms(text)), i, doc_id) for i, (doc_id, text) in enumerate(DOCUMENTS.items())]
    best = sorted((s for s in scored if s[0] > 0), key=lambda s: (-s[0], s[1]))[:3]
    return [{"id": doc_id, "text": DOCUMENTS[doc_id]} for _, _, doc_id in best]


def invoices() -> list[dict]:
    return [{k: i[k] for k in INVOICE_FIELDS} for i in ACCOUNT["invoices"] if i["customer"] == CUSTOMER]


def payments(issued: list[dict]) -> list[dict]:
    """The account's payments, then the refunds this system issued (each {"refund_id", "invoice_id", "amount"})."""
    rows = [{k: p[k] for k in PAYMENT_FIELDS if k in p} for p in ACCOUNT["payments"] if p["customer"] == CUSTOMER]
    rows += [{"id": r["refund_id"], "date": TODAY.isoformat(), "type": "refund", "amount": r["amount"],
              "invoice": r["invoice_id"], "status": "succeeded", "note": f"Issued after review ({r['request_id']})"}
             for r in issued]
    return rows


def cents(value) -> int:
    return round(float(value) * 100)


def left_to_refund_cents(invoice_id: str, issued: list[dict]) -> int:
    """Succeeded charges minus succeeded refunds on one invoice, including refunds this system issued."""
    total = 0
    for p in payments(issued):
        if p["invoice"] == invoice_id and p["status"] == "succeeded":
            total += cents(p["amount"]) if p["type"] == "charge" else -cents(p["amount"])
    return total


def _refund_status(invoice: dict, issued: list[dict]) -> dict:
    invoice_date = date.fromisoformat(invoice["date"])
    age = (TODAY - invoice_date).days
    if age < 0:
        raise ValueError(f"{invoice['id']} is dated after today")
    row = {**invoice, "days_since_invoice": age, "refund_deadline": (invoice_date + timedelta(days=WINDOW)).isoformat()}
    if invoice["status"] == "refunded" or left_to_refund_cents(invoice["id"], issued) <= 0:
        refundable, reason = False, "already refunded"
    elif age > WINDOW:
        refundable, reason = False, f"outside the {WINDOW}-day refund window"
    else:
        refundable, reason = True, f"within the {WINDOW}-day refund window"
    return {**row, "refundable_today": refundable, "reason": reason}


def _schema_problem(name: str, args) -> str | None:
    schema = BY_NAME[name]["function"]["parameters"]
    if not isinstance(args, dict):
        return "arguments must be a JSON object"
    missing = [k for k in schema["required"] if k not in args]
    if missing:
        return f"missing required argument(s): {', '.join(missing)}"
    extra = [k for k in args if k not in schema["properties"]]
    if extra:
        return f"unknown argument(s): {', '.join(extra)}"
    for key, value in args.items():
        kind = schema["properties"][key]["type"]
        if kind == "string" and not isinstance(value, str):
            return f"argument '{key}' must be a string"
        if kind == "number" and (isinstance(value, bool) or not isinstance(value, (int, float))):
            return f"argument '{key}' must be a number"
    return None


def parse(name: str, arguments: str) -> tuple[dict | None, str | None]:
    """Parses and schema-checks one requested call. Returns (args, None) or (None, error message)."""
    if name not in BY_NAME:
        return None, f"Unknown tool: {name}"
    try:
        args = json.loads(arguments) if arguments.strip() else {}
    except json.JSONDecodeError:
        return None, "Arguments are not valid JSON."
    problem = _schema_problem(name, args)
    return (None, f"Invalid arguments: {problem}.") if problem else (args, None)


def _execute(name: str, args: dict, issued: list[dict]):
    if name == "search_help_center":
        return _search(args["query"])
    if name == "get_invoices":
        return invoices()
    if name == "get_refund_status":
        rows = invoices()
        if "invoice_id" in args:
            rows = [r for r in rows if r["id"] == args["invoice_id"].strip()]
            if not rows:
                return {"error": f"No invoice {args['invoice_id']} was found for this customer."}
        return [_refund_status(r, issued) for r in rows]
    if name == "get_payments":
        return payments(issued)
    if name == "get_plan_history":
        return [{"date": h["date"], "change": h["change"]} for h in ACCOUNT["plan_history"] if h["customer"] == CUSTOMER]
    raise ValueError(name)


def run_read(name: str, arguments: str, issued: list[dict]) -> dict:
    """Runs one read-only call. Returns a record: the call, whether it was valid and executed, and the result text."""
    record = {"tool": name, "arguments": arguments, "valid": False, "executed": False}
    args, error = parse(name, arguments)
    if error:
        result = {"error": error}
    else:
        record["valid"] = True
        result = _execute(name, args, issued)
        record["executed"] = not (isinstance(result, dict) and "error" in result)
    return {**record, "result_text": json.dumps(result, ensure_ascii=False)}
