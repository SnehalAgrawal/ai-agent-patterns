"""
Tests that need no model: the transition table and guards, the payment service's rules
and idempotency, the retry policy, and recovery after a crash for both designs. The
model is replaced by scripted replies.

  python -m unittest test_workflow.py
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import agent_loop
import workflow
from payments import PaymentService, new_refunds

DATA = Path(__file__).resolve().parent.parent / "data"
SCENARIOS = {s["id"]: s for s in json.loads((DATA / "scenarios.json").read_text())["scenarios"]}


class Crash(Exception):
    pass


def crash():
    raise Crash()


def reply(content="", tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = [{"function": {"name": n, "arguments": a}} for n, a in tool_calls]
    return {"content": content, "tool_calls": [{"name": n, "arguments": json.dumps(a)} for n, a in tool_calls or []],
            "finish_reason": "stop", "input_tokens": 1, "cached_tokens": 0, "output_tokens": 1, "seconds": 0.0,
            "error": None, "message": message}


def scripted(*replies):
    queue = list(replies)
    return mock.patch("llm.chat", side_effect=lambda *a, **k: queue.pop(0))


def service_for(scenario: str, tmp: Path, crash_fn=None, **fault) -> PaymentService:
    s = SCENARIOS[scenario]
    return PaymentService.create(tmp / "ledger.json", s["prior_refunds"], {**s["fault"], **fault}, crash=crash_fn)


INTAKE_OCT = reply('{"kind": "duplicate_charge", "invoice_id": "", "month": "2026-10"}')


class TransitionTests(unittest.TestCase):
    def test_table_rejects_skipping_verification(self):
        case = workflow.new_case("x", "m")
        case["request"] = {"kind": "duplicate_charge"}
        with self.assertRaises(workflow.InvalidTransition):
            workflow.transition(case, "refund")

    def test_guard_rejects_refund_without_a_duplicate(self):
        case = workflow.new_case("x", "m")
        case["node"] = "verify"
        case["verification"] = {"outcome": "no_duplicate", "refund_payment_id": None}
        with self.assertRaises(workflow.InvalidTransition):
            workflow.transition(case, "refund")

    def test_guard_rejects_a_second_refund(self):
        case = workflow.new_case("x", "m")
        case["node"] = "verify"
        case["verification"] = {"outcome": "refund", "refund_payment_id": "PAY-5006",
                                "succeeded_charges": ["PAY-5005", "PAY-5006"], "refunded_by": None}
        case["refund"] = {"status": "succeeded"}
        with self.assertRaises(workflow.InvalidTransition):
            workflow.transition(case, "refund")

    def test_reply_needs_a_succeeded_refund(self):
        case = workflow.new_case("x", "m")
        case["node"] = "refund"
        case["refund"] = {"status": "requested"}
        with self.assertRaises(workflow.InvalidTransition):
            workflow.transition(case, "reply")


class PaymentServiceTests(unittest.TestCase):
    def test_data_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = service_for("D1", Path(tmp))
            self.assertIn("did not succeed", svc.refund("PAY-5004")["error"])
            self.assertIn("No payment", svc.refund("PAY-6001")["error"])  # another customer's
            self.assertEqual(svc.refund("PAY-5006")["refund_id"], "PAY-5008")
            self.assertIn("already refunded", svc.refund("PAY-5006")["error"])
            # the service doesn't know the policy: a non-duplicate charge can be refunded
            self.assertEqual(svc.refund("PAY-5003")["status"], "succeeded")

    def test_idempotency_key_replays_the_first_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = service_for("D1", Path(tmp))
            first = svc.refund("PAY-5006", idempotency_key="k")
            again = svc.refund("PAY-5006", idempotency_key="k")
            self.assertTrue(again["replayed"])
            self.assertEqual(again["refund_id"], first["refund_id"])
            self.assertEqual(len(new_refunds(svc.load())), 1)
            self.assertIn("already used", svc.refund("PAY-5005", idempotency_key="k")["error"])


class WorkflowTests(unittest.TestCase):
    def run_s(self, scenario, intake, svc_kwargs=None):
        with tempfile.TemporaryDirectory() as tmp:
            svc = service_for(scenario, Path(tmp), **(svc_kwargs or {}))
            with scripted(intake, reply("reply text")), mock.patch("time.sleep"):
                case = workflow.run(Path(tmp), "c", SCENARIOS[scenario]["message"], 1, service=svc)
            return case, svc.load()

    def test_duplicate_is_refunded_once(self):
        case, ledger = self.run_s("D1", INTAKE_OCT)
        self.assertEqual([r["refund_of"] for r in new_refunds(ledger)], ["PAY-5006"])
        self.assertEqual(case["log"], ["intake -> verify", "verify -> refund", "refund -> reply", "reply -> done"])

    def test_already_refunded_makes_no_refund(self):
        case, ledger = self.run_s("R1", INTAKE_OCT)
        self.assertEqual(new_refunds(ledger), [])
        self.assertEqual(case["verification"]["outcome"], "already_refunded")

    def test_no_duplicate_in_november(self):
        case, ledger = self.run_s("N1", reply('{"kind": "duplicate_charge", "invoice_id": "", "month": "2026-11"}'))
        self.assertEqual(new_refunds(ledger), [])
        self.assertEqual(case["verification"]["outcome"], "no_duplicate")

    def test_other_refund_is_handed_off(self):
        case, ledger = self.run_s("O1", reply('{"kind": "other_refund", "invoice_id": "", "month": "2026-09"}'))
        self.assertEqual(new_refunds(ledger), [])
        self.assertEqual(len(ledger["handoffs"]), 1)
        self.assertEqual(case["reply"], workflow.HANDOFF_REPLY)

    def test_transient_failure_is_retried_with_the_same_key(self):
        case, ledger = self.run_s("T1", INTAKE_OCT)
        attempts = case["refund"]["attempts"]
        self.assertEqual(len(attempts), 2)
        self.assertTrue(attempts[0]["result"]["retryable"])
        self.assertEqual(len(new_refunds(ledger)), 1)
        self.assertEqual({r["idempotency_key"] for r in ledger["requests"]}, {"c:PAY-5006"})

    def test_outage_goes_to_handoff_after_three_attempts(self):
        case, ledger = self.run_s("D1", INTAKE_OCT, {"transient_failures": 5})
        self.assertEqual(len(case["refund"]["attempts"]), 3)
        self.assertEqual(case["refund"]["status"], "failed")
        self.assertEqual(len(ledger["handoffs"]), 1)

    def test_crash_after_commit_then_resume_refunds_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = service_for("D1", Path(tmp), crash_fn=crash, crash_after_commit=True)
            with scripted(INTAKE_OCT), self.assertRaises(Crash):
                workflow.run(Path(tmp), "c", SCENARIOS["D1"]["message"], 1, service=svc)
            saved = json.loads((Path(tmp) / "case.json").read_text())
            self.assertEqual((saved["node"], saved["refund"]["status"]), ("refund", "requested"))
            with scripted(reply("reply text")):
                case = workflow.run(Path(tmp), "c", "", 1, resume=True, service=PaymentService(Path(tmp) / "ledger.json"))
            ledger = json.loads((Path(tmp) / "ledger.json").read_text())
            self.assertEqual(len(new_refunds(ledger)), 1)
            self.assertTrue(case["refund"]["attempts"][-1]["result"]["replayed"])


class LoopTests(unittest.TestCase):
    def test_resume_reruns_the_unanswered_refund_call(self):
        """L's resume runs the pending call again; without a key, the service reports it already refunded."""
        with tempfile.TemporaryDirectory() as tmp:
            svc = service_for("D1", Path(tmp), crash_fn=crash, crash_after_commit=True)
            with scripted(reply(tool_calls=[("issue_refund", {"payment_id": "PAY-5006"})])), self.assertRaises(Crash):
                agent_loop.run(Path(tmp), SCENARIOS["D1"]["message"], 1, service=svc)
            with scripted(reply("final")):
                state = agent_loop.run(Path(tmp), "", 1, resume=True, service=PaymentService(Path(tmp) / "ledger.json"))
            self.assertIn("already refunded", state["messages"][-2]["content"])
            self.assertEqual(state["final"], "final")
            self.assertEqual(len(new_refunds(json.loads((Path(tmp) / "ledger.json").read_text()))), 1)

    def test_step_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = service_for("D1", Path(tmp))
            calls = [reply(tool_calls=[("get_invoices", {})]) for _ in range(agent_loop.MAX_MODEL_CALLS)]
            with scripted(*calls):
                state = agent_loop.run(Path(tmp), "m", 1, service=svc)
            self.assertEqual(state["outcome"], "step_limit")


if __name__ == "__main__":
    unittest.main()
