"""
The three toolsets compared in this demo. Everything else (model, prompt, data,
loop, sampling) is the same for all three.

  A  endpoint-shaped: Article #8's two read tools. get_invoices returns the
     invoice rows as stored; errors are short.
  B  the same two tools, names, and descriptions, with richer responses:
     get_invoices adds each invoice's age and refund deadline, an empty search
     says what to try, and errors say how to call the tool correctly.
     (Two changes at once: richer results and better errors.)
  C  toolset A plus one task-shaped tool, get_refund_status, which works out
     from the account data, at call time, whether each invoice can be refunded.

Every tool here only reads. Whatever the model asks for, the code below decides
what runs: it parses and validates the arguments before executing anything.
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
INVOICE_FIELDS = ("id", "date", "amount", "description", "status")

STOPWORDS = {"a", "an", "and", "are", "can", "do", "does", "for", "how", "i", "if", "in", "is", "it",
             "my", "of", "on", "or", "the", "to", "what", "when", "where", "which", "with", "you", "your"}


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required,
                       "additionalProperties": False}}}


SEARCH = _tool("search_help_center", "Search the help center articles. Returns the three most relevant articles.",
               {"query": {"type": "string", "description": "What to search for."}}, ["query"])
GET_INVOICES = _tool("get_invoices", "List the signed-in customer's invoices.", {}, [])
GET_REFUND_STATUS = _tool(
    "get_refund_status",
    "For each of the signed-in customer's invoices: whether it can be refunded today, why, and its refund "
    "deadline. Pass invoice_id to get one invoice only.",
    {"invoice_id": {"type": "string", "description": "Optional. One invoice id, for example INV-0000."}}, [])


# ---- the data behind the tools (shared by all three toolsets) --------------------------------

def _terms(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if w not in STOPWORDS}


def _search(query: str) -> list[dict]:
    """Keyword search, as in Article #8: the three articles sharing the most terms (ties keep file order)."""
    query_terms = _terms(query)
    scored = [(len(query_terms & _terms(text)), i, doc_id) for i, (doc_id, text) in enumerate(DOCUMENTS.items())]
    best = sorted((s for s in scored if s[0] > 0), key=lambda s: (-s[0], s[1]))[:3]
    return [{"id": doc_id, "text": DOCUMENTS[doc_id]} for _, _, doc_id in best]


def _invoices() -> list[dict]:
    """The signed-in customer's invoices, as stored."""
    customer = ACCOUNT["customer"]["id"]
    return [{k: i[k] for k in INVOICE_FIELDS} for i in ACCOUNT["invoices"] if i["customer"] == customer]


def _with_window(invoice: dict) -> dict:
    """The stored invoice plus its age and refund deadline, computed now from its date."""
    invoice_date = date.fromisoformat(invoice["date"])
    age = (TODAY - invoice_date).days
    if age < 0:
        raise ValueError(f"{invoice['id']} is dated after today")
    return {**invoice, "days_since_invoice": age, "refund_deadline": (invoice_date + timedelta(days=WINDOW)).isoformat()}


def _refund_status(invoice: dict) -> dict:
    row = _with_window(invoice)
    if invoice["status"] == "refunded":
        refundable, reason = False, "already refunded"
    elif row["days_since_invoice"] > WINDOW:
        refundable, reason = False, f"outside the {WINDOW}-day refund window"
    else:
        refundable, reason = True, f"within the {WINDOW}-day refund window"
    return {**row, "refundable_today": refundable, "reason": reason}


# ---- argument checks ----------------------------------------------------------------------------

def _schema_problem(tool: dict, args) -> str | None:
    """Checks arguments against the tool's JSON schema (the subset these tools use). Returns a problem or None."""
    schema = tool["function"]["parameters"]
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


class Toolset:
    """One toolset: the tool definitions the model sees, and the code that runs a requested call."""

    def __init__(self, name: str, tools: list[dict]):
        self.name = name
        self.tools = tools
        self.by_name = {t["function"]["name"]: t for t in tools}

    def run(self, name: str, arguments: str) -> dict:
        """
        Handles one requested call. Returns a record: the call, whether it was
        valid, whether it executed, and the exact text sent back to the model.
        """
        record = {"tool": name, "arguments": arguments, "valid": False, "executed": False}
        if name not in self.by_name:
            result = {"error": self.unknown_tool(name)}
        else:
            try:
                args = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError:
                args = None
            if args is None:
                result = {"error": self.invalid_json(name)}
            else:
                problem = _schema_problem(self.by_name[name], args)
                if problem:
                    result = {"error": self.invalid_arguments(name, problem)}
                else:
                    record["valid"] = True
                    result = self.execute(name, args)
                    record["executed"] = "error" not in result
        return {**record, "result_text": json.dumps(result, ensure_ascii=False)}

    # Short errors, as in Article #8. Toolset B overrides these.
    def unknown_tool(self, name: str) -> str:
        return f"Unknown tool: {name}"

    def invalid_json(self, name: str) -> str:
        return "Arguments are not valid JSON."

    def invalid_arguments(self, name: str, problem: str) -> str:
        return f"Invalid arguments: {problem}."

    def execute(self, name: str, args: dict):
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
        raise ValueError(name)


class RicherResponses(Toolset):
    """Toolset B: the same tools as A; richer results, and errors that say what to do next."""

    def unknown_tool(self, name: str) -> str:
        return f"Unknown tool: {name}. The available tools are {' and '.join(self.by_name)}."

    def invalid_json(self, name: str) -> str:
        return f"Arguments are not valid JSON. {self._usage(name)}"

    def invalid_arguments(self, name: str, problem: str) -> str:
        return f"Invalid arguments for {name}: {problem}. {self._usage(name)}"

    def _usage(self, name: str) -> str:
        if name == "get_invoices":
            return "get_invoices takes no arguments: call it with {}."
        return 'search_help_center takes one string, query: for example {"query": "your search words"}.'

    def execute(self, name: str, args: dict):
        if name == "search_help_center":
            results = _search(args["query"])
            if results:
                return {"results": results}
            return {"results": [], "hint": "No article matched these words. Try a shorter query with one or two "
                                          "key words."}
        if name == "get_invoices":
            return {"today": TODAY.isoformat(),
                    "refund_window": f"{WINDOW} days from the invoice date, unless already refunded",
                    "invoices": [_with_window(i) for i in _invoices()]}
        raise ValueError(name)


TOOLSETS = {
    "A": Toolset("A", [SEARCH, GET_INVOICES]),
    "B": RicherResponses("B", [SEARCH, GET_INVOICES]),
    "C": Toolset("C", [SEARCH, GET_INVOICES, GET_REFUND_STATUS]),
}
