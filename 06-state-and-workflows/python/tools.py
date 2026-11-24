"""
L's five tools (definitions in data/designs.json, 'loop_tools'), run against one
conversation's payment service. Requested calls are parsed and checked against the
tool's schema before anything runs, as in the earlier demos.

search_help_center and get_invoices behave as in 02-agentic-workflows. get_payments
reads the ledger, so refunds made during the conversation appear in it. issue_refund
goes straight to the payment service, without an idempotency key. hand_off_to_billing
records a handoff in the ledger.
"""
import json
import re

from payments import DATA, PaymentService

TOOLS = json.loads((DATA / "designs.json").read_text())["loop_tools"]["tools"]
BY_NAME = {t["function"]["name"]: t for t in TOOLS}
DOCUMENTS = json.loads((DATA / "help_center.json").read_text())["documents"]
INVOICE_FIELDS = ("id", "date", "amount", "description", "status")
PAYMENT_FIELDS = ("id", "date", "type", "amount", "invoice", "status", "note", "refund_of")

STOPWORDS = {"a", "an", "and", "are", "can", "do", "does", "for", "how", "i", "if", "in", "is", "it",
             "my", "of", "on", "or", "the", "to", "what", "when", "where", "which", "with", "you", "your"}


def _terms(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if w not in STOPWORDS}


def _search(query: str) -> list[dict]:
    """Keyword search, as in Articles #8 and #23: the three articles sharing the most terms (ties keep file order)."""
    query_terms = _terms(query)
    scored = [(len(query_terms & _terms(text)), i, doc_id) for i, (doc_id, text) in enumerate(DOCUMENTS.items())]
    best = sorted((s for s in scored if s[0] > 0), key=lambda s: (-s[0], s[1]))[:3]
    return [{"id": doc_id, "text": DOCUMENTS[doc_id]} for _, _, doc_id in best]


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


class Toolbox:
    def __init__(self, service: PaymentService):
        self.service = service

    def _execute(self, name: str, args: dict):
        if name == "search_help_center":
            return _search(args["query"])
        if name == "get_invoices":
            return [{k: i[k] for k in INVOICE_FIELDS} for i in self.service.invoices()]
        if name == "get_payments":
            return [{k: p[k] for k in PAYMENT_FIELDS if k in p} for p in self.service.payments()]
        if name == "issue_refund":
            return self.service.refund(args["payment_id"])
        if name == "hand_off_to_billing":
            return self.service.hand_off(args["summary"], by="L")
        raise ValueError(name)

    def run(self, name: str, arguments: str) -> dict:
        """Runs one requested call. Returns the call, whether it was valid and executed, and the text the model gets."""
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
                    result = self._execute(name, args)
                    record["executed"] = not (isinstance(result, dict) and "error" in result)
        return {**record, "result_text": json.dumps(result, ensure_ascii=False)}
