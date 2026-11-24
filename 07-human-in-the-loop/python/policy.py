"""
The approval policy from data/policy.json, as plain functions. No model is involved here.

  hard_checks()  what code can decide alone: ownership, a valid amount, money left to refund
  tier()         who may approve an action that passed them: within_policy or exception
  may_decide()   whether a reviewer's role is allowed for a tier
  action_sha()   the fingerprint that binds a decision to the exact action the reviewer saw
"""
import hashlib
import json
from datetime import date
from decimal import Decimal

from tools import ACCOUNT, CUSTOMER, POLICY, TODAY, WINDOW, cents, left_to_refund_cents

TIERS = POLICY["tiers"]
TTL_HOURS = POLICY["approval_ttl_hours"]


def _invoice(invoice_id: str) -> dict | None:
    return next((i for i in ACCOUNT["invoices"] if i["id"] == invoice_id.strip()), None)


def amount_is_valid(amount) -> bool:
    if isinstance(amount, bool) or not isinstance(amount, (int, float)):
        return False
    value = Decimal(str(amount))
    return value > 0 and value == value.quantize(Decimal("0.01"))


def hard_checks(invoice_id: str, amount, issued: list[dict]) -> list[dict]:
    """Every check, in order, with whether it passed and what it compared. Stops at the first failure."""
    results = []
    invoice = _invoice(invoice_id)
    owned = invoice is not None and invoice["customer"] == CUSTOMER
    results.append({"id": "owned", "passed": owned,
                    "detail": f"{invoice_id} belongs to the signed-in customer" if owned
                    else f"{invoice_id} is not one of the signed-in customer's invoices"})
    if not owned:
        return results
    valid = amount_is_valid(amount)
    results.append({"id": "amount_valid", "passed": valid,
                    "detail": f"${float(amount):.2f} is a valid amount" if valid else f"{amount!r} is not a valid amount"})
    if not valid:
        return results
    left = left_to_refund_cents(invoice["id"], issued)
    available = cents(amount) <= left
    results.append({"id": "amount_available", "passed": available,
                    "detail": f"${float(amount):.2f} of ${left / 100:.2f} left to refund on {invoice['id']}" if available
                    else f"${float(amount):.2f} requested, but only ${left / 100:.2f} is left to refund on {invoice['id']}"})
    return results


def failed(checks: list[dict]) -> str | None:
    return next((c["detail"] for c in checks if not c["passed"]), None)


def tier(invoice_id: str, amount) -> dict:
    """within_policy: at most WINDOW days old and at most the invoice amount. Otherwise exception."""
    invoice = _invoice(invoice_id)
    age = (TODAY - date.fromisoformat(invoice["date"])).days
    in_window = age <= WINDOW
    in_amount = cents(amount) <= cents(invoice["amount"])
    name = "within_policy" if in_window and in_amount else "exception"
    return {"tier": name, "may_decide": TIERS[name]["may_decide"],
            "facts": {"days_since_invoice": age, "refund_window_days": WINDOW, "invoice_amount": invoice["amount"],
                      "within_window": in_window, "at_most_invoice_amount": in_amount}}


def may_decide(tier_name: str, role: str) -> bool:
    return role in TIERS[tier_name]["may_decide"]


def action_sha(action: dict) -> str:
    """SHA-256 of the canonical action: tool, invoice, and amount in cents. The reason doesn't change the action."""
    canonical = {"tool": "issue_refund", "invoice_id": action["invoice_id"].strip(), "amount_cents": cents(action["amount"])}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


def half_rounded_down(amount) -> float:
    """The measured run's scripted modification: half the proposed amount, rounded down to the cent."""
    return (cents(amount) // 2) / 100
