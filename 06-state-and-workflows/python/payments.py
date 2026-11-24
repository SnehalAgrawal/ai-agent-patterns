"""
A simulated payment service: the persisted application data both designs act on.

Each conversation gets its own ledger file, a copy of data/account.json plus the
scenario's earlier refunds. The service enforces its own data rules, as a real payment
API would: a refund must name an existing, successful charge of this customer that
hasn't been refunded. It knows nothing about the support team's refund policy (which
charges *should* be refunded); that is the caller's job.

It also supports idempotency keys, in the way common payment APIs describe them: the
first successful result for a key is saved, and a repeat with the same key returns that
saved result instead of refunding again. Errors are not saved, so a failed request can
be retried with the same key.

Faults, for the experiments:
  transient_failures  the first N refund requests fail with a retryable error and change nothing
  crash_after_commit  the process exits right after the first refund is written to the ledger,
                      before the caller gets the result
"""
import json
import os
from datetime import date
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
ACCOUNT = json.loads((DATA / "account.json").read_text())
TODAY = ACCOUNT["today"]
CUSTOMER = ACCOUNT["customer"]["id"]
CRASH_EXIT_CODE = 75


def _write(path: Path, data: dict) -> None:
    """Atomic on one machine: a reader sees the old file or the new one, never half of one."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    os.replace(tmp, path)


class PaymentService:
    def __init__(self, ledger: Path, crash=None):
        self.path = Path(ledger)
        self.crash = crash or (lambda: os._exit(CRASH_EXIT_CODE))  # tests replace this with an exception

    @classmethod
    def create(cls, ledger: Path, prior_refunds: list[dict], fault: dict, crash=None) -> "PaymentService":
        data = {"invoices": ACCOUNT["invoices"], "payments": ACCOUNT["payments"] + prior_refunds,
                "initial_payment_ids": [p["id"] for p in ACCOUNT["payments"] + prior_refunds],
                "idempotency": {}, "requests": [], "handoffs": [],
                "fault": {"transient_failures": fault.get("transient_failures", 0),
                          "crash_after_commit": bool(fault.get("crash_after_commit", False))}}
        _write(Path(ledger), data)
        return cls(ledger, crash)

    def load(self) -> dict:
        return json.loads(self.path.read_text())

    # --- reads -----------------------------------------------------------------------------
    def payments(self) -> list[dict]:
        return [p for p in self.load()["payments"] if p["customer"] == CUSTOMER]

    def invoices(self) -> list[dict]:
        return [i for i in self.load()["invoices"] if i["customer"] == CUSTOMER]

    # --- writes ----------------------------------------------------------------------------
    def refund(self, payment_id: str, idempotency_key: str | None = None) -> dict:
        data = self.load()
        data["requests"].append({"payment_id": payment_id, "idempotency_key": idempotency_key})
        payment_id = (payment_id or "").strip()

        if idempotency_key and idempotency_key in data["idempotency"]:
            saved = data["idempotency"][idempotency_key]
            _write(self.path, data)
            if saved["payment_id"] != payment_id:
                return {"error": f"Idempotency key {idempotency_key} was already used for {saved['payment_id']}.",
                        "retryable": False}
            return {**saved["response"], "replayed": True}

        if data["fault"]["transient_failures"] > 0:
            data["fault"]["transient_failures"] -= 1
            _write(self.path, data)
            return {"error": "The payment service is temporarily unavailable (HTTP 503). No refund was made. "
                             "The request can be retried.", "retryable": True}

        problem = self._problem(data, payment_id)
        if problem:
            _write(self.path, data)
            return {"error": problem, "retryable": False}

        charge = next(p for p in data["payments"] if p["id"] == payment_id)
        refund_id = self._next_id(data)
        data["payments"].append({"id": refund_id, "customer": CUSTOMER, "date": TODAY, "type": "refund",
                                 "amount": charge["amount"], "invoice": charge["invoice"], "status": "succeeded",
                                 "refund_of": payment_id})
        response = {"refund_id": refund_id, "refund_of": payment_id, "amount": charge["amount"],
                    "invoice": charge["invoice"], "status": "succeeded",
                    "note": "Returned to the original payment method within 5 business days."}
        if idempotency_key:
            data["idempotency"][idempotency_key] = {"payment_id": payment_id, "response": response}
        crash = data["fault"]["crash_after_commit"]
        data["fault"]["crash_after_commit"] = False  # the crash happens once; recovery uses the same ledger
        _write(self.path, data)
        if crash:
            self.crash()
        return response

    def hand_off(self, summary: str, by: str) -> dict:
        data = self.load()
        data["handoffs"].append({"summary": summary, "by": by, "date": TODAY})
        _write(self.path, data)
        return {"status": "handed off", "message": "A person from the billing team will reply to the customer by email."}

    # --- helpers ---------------------------------------------------------------------------
    @staticmethod
    def _problem(data: dict, payment_id: str) -> str | None:
        payment = next((p for p in data["payments"] if p["id"] == payment_id), None)
        if payment is None or payment["customer"] != CUSTOMER:
            return f"No payment {payment_id} was found for this customer."
        if payment["type"] != "charge":
            return f"{payment_id} is a {payment['type']}, not a charge."
        if payment["status"] != "succeeded":
            return f"{payment_id} did not succeed ({payment['status']}), so there is nothing to refund."
        earlier = next((p for p in data["payments"] if p.get("refund_of") == payment_id), None)
        if earlier:
            return f"{payment_id} was already refunded on {earlier['date']} ({earlier['id']})."
        return None

    @staticmethod
    def _next_id(data: dict) -> str:
        ids = {p["id"] for p in data["payments"]}
        n = 5008
        while f"PAY-{n}" in ids:
            n += 1
        return f"PAY-{n}"


def new_refunds(ledger: dict) -> list[dict]:
    """Refunds added to the ledger after it was created."""
    initial = set(ledger["initial_payment_ids"])
    return [p for p in ledger["payments"] if p["type"] == "refund" and p["id"] not in initial]


def invoice_month(invoice: dict) -> str:
    return date.fromisoformat(invoice["date"]).strftime("%Y-%m")
