"""
The approval lifecycle, tested without a model: a scripted stand-in replaces Ollama, so these
tests check the code around the model (pause, decide, expire, resume, execute, audit), not
the model. Run from this folder:

  python -m unittest test_lifecycle -v
"""
import json
import os
import tempfile
import unittest

from agent import Agent
from store import Store

T0 = "2026-11-12T10:00:00Z"   # request created
T1 = "2026-11-12T12:00:00Z"   # decision, within the 24-hour expiry
T2 = "2026-11-13T10:00:01Z"   # one second after expiry


def reply(text: str) -> dict:
    return {"content": text, "tool_calls": [], "finish_reason": "stop", "input_tokens": 0, "cached_tokens": 0,
            "output_tokens": 0, "seconds": 0.0, "error": None, "message": {"role": "assistant", "content": text}}


def calls(*pairs) -> dict:
    raw = [{"function": {"name": n, "arguments": a}} for n, a in pairs]
    return {"content": "", "tool_calls": [{"name": n, "arguments": json.dumps(a)} for n, a in pairs],
            "finish_reason": "stop", "input_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "seconds": 0.0,
            "error": None, "message": {"role": "assistant", "content": "", "tool_calls": raw}}


def refund(invoice: str, amount, reason: str = "Duplicate charge.") -> tuple:
    return ("issue_refund", {"invoice_id": invoice, "amount": amount, "reason": reason})


class Script:
    """Returns the scripted responses in order, and keeps every message list it was sent."""

    def __init__(self, *responses):
        self.responses, self.sent = list(responses), []

    def __call__(self, messages, tools, seed, max_tokens, schema=None):
        self.sent.append([dict(m) for m in messages])
        return self.responses.pop(0)


class Lifecycle(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "hitl.sqlite")
        self.store = Store(self.path)

    def tearDown(self):
        self.store.close()
        self.dir.cleanup()

    def agent(self, *responses) -> Agent:
        return Agent(self.store, seed=1, chat=Script(*responses))

    def pause_duplicate_refund(self, amount=25) -> str:
        agent = self.agent(calls(("get_payments", {})), calls(refund("INV-1002", amount)))
        record = agent.start("c1", "I was charged twice for October. Please refund the extra charge.", T0)
        self.assertEqual(record["outcome"], "paused")
        return record["request_id"]

    def test_pause_saves_everything_and_executes_nothing(self):
        request_id = self.pause_duplicate_refund()
        req = self.store.get(request_id)
        self.assertEqual(req["status"], "pending")
        self.assertEqual(req["tier"], "within_policy")
        self.assertEqual(req["expires_at"], "2026-11-13T10:00:00Z")
        self.assertEqual(self.store.issued(), [])
        view = req["view"]
        self.assertEqual(view["action"], {"tool": "issue_refund", "invoice_id": "INV-1002", "amount": 25})
        self.assertEqual([c["id"] for c in view["checks"]], ["owned", "amount_valid", "amount_available"])
        self.assertEqual(view["evidence"][0]["tool"], "get_payments")
        self.assertEqual(view["agent_reason"], "Duplicate charge.")

    def test_resume_from_the_database_alone_after_approval(self):
        request_id = self.pause_duplicate_refund()
        sha = self.store.get(request_id)["action_sha256"]
        self.store.close()
        # Nothing kept in memory: a new connection to the same file and a new agent, as another process would have.
        self.store = Store(self.path)
        script = Script(reply("Your refund of $25 has been issued."))
        agent = Agent(self.store, chat=script)
        self.assertTrue(agent.decide(request_id, "approve", "rev-1", "support", sha, T1)["ok"])
        record = agent.resume(request_id, T1)
        self.assertEqual(record["outcome"], "answered")
        self.assertEqual(record["tool_result"]["status"], "approved")
        self.assertEqual(len(self.store.issued()), 1)
        # The model saw its earlier messages, then the decision as the issue_refund result.
        last = script.sent[0][-1]
        self.assertEqual(last["role"], "tool")
        self.assertEqual(json.loads(last["content"])["refund_issued"], True)

    def test_hard_checks_stop_bad_calls_without_a_person(self):
        for invoice, amount, expected in [("INV-2042", 25, "not one of the signed-in customer's invoices"),
                                          ("INV-0998", 25, "only $0.00 is left"),
                                          ("INV-1002", 60, "only $50.00 is left"),
                                          ("INV-1002", 0.001, "not a valid amount")]:
            with self.subTest(invoice=invoice, amount=amount):
                script = Script(calls(refund(invoice, amount)), reply("Sorry."))
                record = Agent(self.store, chat=script).start(f"c-{invoice}-{amount}", "Refund.", T0)
                self.assertEqual(record["outcome"], "answered")
                self.assertIsNone(record["request_id"])
                self.assertIn(expected, json.loads(script.sent[1][-1]["content"])["error"])
        self.assertEqual(self.store.requests(), [])

    def test_roles_follow_the_tier(self):
        agent = self.agent(calls(refund("INV-1002", 50)))
        request_id = agent.start("c1", "Refund both October charges.", T0)["request_id"]
        req = self.store.get(request_id)
        self.assertEqual(req["tier"], "exception")
        refused = agent.decide(request_id, "approve", "rev-1", "support", req["action_sha256"], T1)
        self.assertFalse(refused["ok"])
        self.assertTrue(agent.decide(request_id, "approve", "rev-2", "billing_lead", req["action_sha256"], T1)["ok"])

    def test_decision_is_bound_to_the_action_shown(self):
        request_id = self.pause_duplicate_refund()
        result = Agent(self.store).decide(request_id, "approve", "rev-1", "support", "0" * 64, T1)
        self.assertFalse(result["ok"])
        self.assertEqual(self.store.get(request_id)["status"], "pending")

    def test_only_the_first_decision_counts(self):
        request_id = self.pause_duplicate_refund()
        sha = self.store.get(request_id)["action_sha256"]
        other = Store(self.path)  # a second reviewer, on another connection
        try:
            first = Agent(self.store).decide(request_id, "approve", "rev-1", "support", sha, T1)
            second = Agent(other).decide(request_id, "reject", "rev-2", "support", sha, T1, reason="No.")
        finally:
            other.close()
        self.assertTrue(first["ok"])
        self.assertFalse(second["ok"])
        self.assertEqual(self.store.get(request_id)["status"], "approved")

    def test_two_reviewers_at_the_same_moment(self):
        # Both reviewers loaded the request while it was pending; the first decision lands
        # between the second reviewer's read and write. Only the compare-and-set stops it.
        request_id = self.pause_duplicate_refund()
        sha = self.store.get(request_id)["action_sha256"]
        other = Store(self.path)
        try:
            stale = other.get(request_id)
            other.get = lambda _id: stale
            self.assertTrue(Agent(self.store).decide(request_id, "approve", "rev-1", "support", sha, T1)["ok"])
            second = Agent(other).decide(request_id, "reject", "rev-2", "support", sha, T1, reason="No.")
        finally:
            other.close()
        self.assertFalse(second["ok"])
        self.assertIn("someone else first", second["error"])
        self.assertEqual(self.store.get(request_id)["decision"]["reviewer"], "rev-1")

    def test_expired_requests_cannot_be_approved_and_resume_as_expired(self):
        request_id = self.pause_duplicate_refund()
        sha = self.store.get(request_id)["action_sha256"]
        agent = Agent(self.store, chat=Script(reply("Your request expired; nothing was refunded.")))
        self.assertFalse(agent.decide(request_id, "approve", "rev-1", "support", sha, T2)["ok"])
        self.assertEqual(self.store.get(request_id)["status"], "expired")
        record = agent.resume(request_id, T2)
        self.assertEqual(record["tool_result"]["status"], "expired")
        self.assertEqual(self.store.issued(), [])

    def test_resume_before_a_decision_is_refused(self):
        request_id = self.pause_duplicate_refund()
        self.assertEqual(Agent(self.store).resume(request_id, T1)["outcome"], "refused")

    def test_modified_actions_are_checked_again(self):
        request_id = self.pause_duplicate_refund()
        sha = self.store.get(request_id)["action_sha256"]
        agent = Agent(self.store)
        self.assertFalse(agent.decide(request_id, "modify", "rev-1", "support", sha, T1, amount=60)["ok"])
        # $40 passes the hard checks ($50 left) but is more than the invoice: exception tier.
        self.assertFalse(agent.decide(request_id, "modify", "rev-1", "support", sha, T1, amount=40)["ok"])
        self.assertTrue(agent.decide(request_id, "modify", "rev-1", "support", sha, T1, amount=12.5)["ok"])
        req = self.store.get(request_id)
        self.assertEqual(req["final_action"], {"invoice_id": "INV-1002", "amount": 12.5})
        self.assertEqual(req["action"]["amount"], 25)

    def test_records_changed_while_waiting(self):
        # Two refunds of $25 on INV-1002 ($50 charged), each approved; then a third.
        first = self.pause_duplicate_refund()
        agent = self.agent(calls(refund("INV-1002", 25)))
        second = agent.start("c2", "Refund the extra October charge.", T0)["request_id"]
        third = self.agent(calls(refund("INV-1002", 25))).start("c3", "Refund October.", T0)["request_id"]
        for request_id in (first, second, third):
            sha = self.store.get(request_id)["action_sha256"]
            self.assertTrue(Agent(self.store).decide(request_id, "approve", "rev-1", "support", sha, T1)["ok"])
        for request_id in (first, second):
            Agent(self.store, chat=Script(reply("Done."))).resume(request_id, T1)
        # All three were approved while $50 was left; by the time the third runs, nothing is.
        record = Agent(self.store, chat=Script(reply("It couldn't be refunded."))).resume(third, T1)
        self.assertEqual(record["tool_result"]["status"], "approved_not_executed")
        self.assertEqual(len(self.store.issued()), 2)

    def test_execution_happens_once(self):
        request_id = self.pause_duplicate_refund()
        sha = self.store.get(request_id)["action_sha256"]
        agent = Agent(self.store, chat=Script({**reply(""), "error": "model server down"}, reply("Issued.")))
        agent.decide(request_id, "approve", "rev-1", "support", sha, T1)
        # The refund runs, then the model call fails: the resume didn't finish.
        crashed = agent.resume(request_id, T1)
        self.assertEqual(crashed["outcome"], "error")
        self.assertEqual(len(self.store.issued()), 1)
        # Retrying the resume doesn't refund again.
        retried = agent.resume(request_id, T1)
        self.assertEqual(retried["outcome"], "answered")
        self.assertEqual(len(self.store.issued()), 1)
        # A finished resume can't be repeated.
        self.assertEqual(agent.resume(request_id, T1)["outcome"], "refused")
        kinds = [e["type"] for e in self.store.events(request_id=request_id)]
        self.assertEqual(kinds.count("refund_executed"), 1)
        self.assertEqual(kinds.count("execution_skipped"), 1)

    def test_rejected_refund_cannot_be_requested_again(self):
        request_id = self.pause_duplicate_refund()
        sha = self.store.get(request_id)["action_sha256"]
        script = Script(calls(refund("INV-1002", 25, "Please reconsider.")), reply("It was declined."))
        agent = Agent(self.store, chat=script)
        agent.decide(request_id, "reject", "rev-1", "support", sha, T1, reason="Not a duplicate.")
        record = agent.resume(request_id, T1)
        self.assertEqual(record["calls"][0]["refused"], "already_decided")
        self.assertEqual(len(self.store.requests()), 1)

    def test_clarify_returns_the_question_and_a_new_request_supersedes(self):
        request_id = self.pause_duplicate_refund()
        sha = self.store.get(request_id)["action_sha256"]
        script = Script(calls(("get_payments", {})), calls(refund("INV-1002", 25, "PAY-5006 duplicates PAY-5005.")))
        agent = Agent(self.store, chat=script)
        agent.decide(request_id, "clarify", "rev-1", "support", sha, T1, question="Which payment is the duplicate?")
        record = agent.resume(request_id, T1)
        self.assertEqual(json.loads(script.sent[0][-1]["content"])["status"], "needs_clarification")
        self.assertEqual(record["outcome"], "paused")
        new = self.store.get(record["request_id"])
        self.assertEqual(new["supersedes"], request_id)
        self.assertEqual(new["view"]["agent_reason"], "PAY-5006 duplicates PAY-5005.")

    def test_a_changed_prompt_or_toolset_blocks_resume(self):
        request_id = self.pause_duplicate_refund()
        req = self.store.get(request_id)
        checkpoint = req["checkpoint"] | {"version": "an-older-deploy"}
        self.store.db.execute("UPDATE requests SET checkpoint = ? WHERE id = ?", (json.dumps(checkpoint), request_id))
        agent = Agent(self.store)
        agent.decide(request_id, "approve", "rev-1", "support", req["action_sha256"], T1)
        self.assertEqual(agent.resume(request_id, T1)["outcome"], "refused")
        self.assertEqual(self.store.issued(), [])

    def test_audit_trail_tells_the_whole_story(self):
        request_id = self.pause_duplicate_refund()
        sha = self.store.get(request_id)["action_sha256"]
        agent = Agent(self.store, chat=Script(reply("Your refund of $25 has been issued.")))
        agent.decide(request_id, "approve", "rev-1", "support", "0" * 64, T1)  # refused: wrong action
        agent.decide(request_id, "approve", "rev-1", "support", sha, T1)
        agent.resume(request_id, T1)
        kinds = [(e["actor"], e["type"]) for e in self.store.events(conversation_id="c1")]
        self.assertEqual(kinds, [
            ("customer", "customer_message"), ("agent", "tool_call"), ("agent", "refund_proposed"),
            ("system", "approval_requested"), ("rev-1", "decision_refused"), ("rev-1", "decision_recorded"),
            ("system", "refund_executed"), ("system", "resumed"), ("agent", "reply")])


if __name__ == "__main__":
    unittest.main()
