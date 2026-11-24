"""
The five tools, shared by all three systems. The workflow calls these functions
directly; the agent requests them by name. Either way, the same code runs, so the
systems differ only in who chooses and sequences the calls.

Every tool only reads. Requested calls are parsed and checked against the tool's
schema before anything runs.

The descriptions apply what 01-tool-design (Article #23) found: no example values
that look like real ids, and an optional argument says what leaving it out returns.
"""
import json
import re
from datetime import date, timedelta
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
ACCOUNT = json.loads((DATA / "account.json").read_text())
DOCUMENTS = json.loads((DATA / "help_center.json").read_text())["documents"]
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


TOOLS = [
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


def _invoices() -> list[dict]:
    return [{k: i[k] for k in INVOICE_FIELDS} for i in ACCOUNT["invoices"] if i["customer"] == CUSTOMER]


def _refund_status(invoice: dict) -> dict:
    invoice_date = date.fromisoformat(invoice["date"])
    age = (TODAY - invoice_date).days
    if age < 0:
        raise ValueError(f"{invoice['id']} is dated after today")
    row = {**invoice, "days_since_invoice": age, "refund_deadline": (invoice_date + timedelta(days=WINDOW)).isoformat()}
    if invoice["status"] == "refunded":
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
        if schema["properties"][key]["type"] == "string" and not isinstance(value, str):
            return f"argument '{key}' must be a string"
    return None


def _execute(name: str, args: dict):
    if name == "search_help_center":
        return _search(args["query"])
    if name == "get_invoices":
        return _invoices()
    if name == "get_refund_status":
        rows = _invoices()
        if "invoice_id" in args:
            rows = [r for r in rows if r["id"] == args["invoice_id"].strip()]
            if not rows:
                return {"error": f"No invoice {args['invoice_id']} was found for this customer."}
        return [_refund_status(r) for r in rows]
    if name == "get_payments":
        return [{k: p[k] for k in PAYMENT_FIELDS if k in p} for p in ACCOUNT["payments"] if p["customer"] == CUSTOMER]
    if name == "get_plan_history":
        return [{"date": h["date"], "change": h["change"]} for h in ACCOUNT["plan_history"] if h["customer"] == CUSTOMER]
    raise ValueError(name)


def run(name: str, arguments: str) -> dict:
    """
    Runs one call, requested by the agent or made by a workflow. Returns a record:
    the call, whether it was valid and executed, and the exact text the model receives.
    """
    record = {"tool": name, "arguments": arguments, "valid": False, "executed": False}
    if name not in BY_NAME:
        result = {"error": f"Unknown tool: {name}"}
    else:
        try:
            args = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            args = None
        if args is None:
            result = {"error": "Arguments are not valid JSON."}
        else:
            problem = _schema_problem(name, args)
            if problem:
                result = {"error": f"Invalid arguments: {problem}."}
            else:
                record["valid"] = True
                result = _execute(name, args)
                record["executed"] = not (isinstance(result, dict) and "error" in result)
    return {**record, "result_text": json.dumps(result, ensure_ascii=False)}
