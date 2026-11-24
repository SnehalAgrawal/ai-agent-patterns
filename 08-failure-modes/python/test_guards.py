"""
Tests of the guards' deterministic behaviour, on hand-written conversations (no model, no
saved runs). They check the code does what data/evaluation.json says; they are not
evidence about any model.

    python -m unittest test_guards -v
"""
import contextlib
import io
import json
import unittest

import guards
import report

LEDGER = [
    {"id": "PAY-5004", "customer": "C-1001", "type": "charge", "amount": 25, "invoice": "INV-1002", "status": "failed"},
    {"id": "PAY-5005", "customer": "C-1001", "type": "charge", "amount": 25, "invoice": "INV-1002", "status": "succeeded"},
    {"id": "PAY-5006", "customer": "C-1001", "type": "charge", "amount": 25, "invoice": "INV-1002", "status": "succeeded"},
    {"id": "PAY-5007", "customer": "C-1001", "type": "charge", "amount": 15, "invoice": "INV-1003", "status": "succeeded"},
    {"id": "PAY-6001", "customer": "C-2042", "type": "charge", "amount": 99, "invoice": "INV-2042", "status": "succeeded"},
]


def call(i, tool, args=None, *, valid=True, executed=True, result="[]", agent="agent"):
    return {"i": i, "agent": agent, "tool": tool, "arguments": json.dumps(args or {}), "args": args or {},
            "valid": valid, "executed": executed, "result_text": result,
            "kind": "agent" if tool in guards.SPECIALIST else "data"}


def conv(calls, request="Which of my invoices can still be refunded?", final="Here you go.", context="",
         demo="01-tool-design", ledger=None):
    return {"demo": demo, "key": f"{demo}|t|m|X|Q|1", "request": request, "context": context, "calls": calls,
            "final": final, "ledger": ledger or [],
            "owned": {"invoice": {"INV-1001", "INV-1002"}, "payment": set()},
            "other": {"invoice": {"INV-2042"}, "payment": set()}}


INVOICES = json.dumps([{"id": "INV-1001"}, {"id": "INV-1002"}])


def refund_conv(calls, prior=()):
    return conv(calls, demo="06-state-and-workflows", ledger=LEDGER + list(prior))


def manager_conv(specialist_calls, final="I couldn't find any invoices."):
    ask = call(len(specialist_calls), "ask_billing_specialist", {"request": "invoices?"}, agent="manager",
               result="No invoices were found.")
    return conv(specialist_calls + [ask], demo="03-multi-agent", final=final)


class GuardTests(unittest.TestCase):
    # --- provenance and id_format --------------------------------------------------------------
    def test_provenance_accepts_ids_from_the_request_context_and_successful_results(self):
        c = conv([call(0, "get_invoices", result=INVOICES), call(1, "get_refund_status", {"invoice_id": "INV-1002"})])
        assert guards.provenance(c, c["calls"][1]) is None
        typed = conv([call(0, "get_refund_status", {"invoice_id": "INV-1001"})], request="Refund for INV-1001?")
        assert guards.provenance(typed, typed["calls"][0]) is None
        remembered = conv([call(0, "get_refund_status", {"invoice_id": "INV-1002"})], context="Invoice INV-1002 was charged twice")
        assert guards.provenance(remembered, remembered["calls"][0]) is None


    def test_provenance_rejects_invented_ids_and_ids_echoed_by_an_error(self):
        c = conv([call(0, "get_refund_status", {"invoice_id": "INV-0000"}, executed=False,
                       result='{"error": "No invoice INV-0000 was found for this customer."}'),
                  call(1, "get_refund_status", {"invoice_id": "INV-0000"})])
        assert guards.provenance(c, c["calls"][0]).startswith("provenance")
        assert guards.provenance(c, c["calls"][1]).startswith("provenance"), "an error result is not a source"


    def test_provenance_compares_whole_values_not_substrings(self):
        c = conv([call(0, "get_invoices", result=INVOICES), call(1, "get_refund_status", {"invoice_id": "INV-1"})])
        assert guards.provenance(c, c["calls"][1]) is not None


    def test_provenance_is_not_ownership(self):
        c = conv([call(0, "issue_refund", {"invoice_id": "INV-2042"})], request="Please refund invoice INV-2042.")
        assert guards.provenance(c, c["calls"][0]) is None


    def test_id_format_catches_placeholders_but_not_well_formed_inventions(self):
        c = conv([call(0, "get_refund_status", {"invoice_id": "most_recent_Pro_invoice_id"}),
                  call(1, "get_refund_status", {"invoice_id": "INV-0000"})])
        assert guards.id_format(c, c["calls"][0]) is not None
        assert guards.id_format(c, c["calls"][1]) is None


    def test_non_string_ids_are_left_to_the_schema_check(self):
        c = conv([call(0, "get_refund_status", {"invoice_id": None}, valid=False)])
        assert guards.provenance(c, c["calls"][0]) is None
        assert guards.id_format(c, c["calls"][0]) is None


    # --- repeat and budget ---------------------------------------------------------------------
    def test_repeat_fires_on_an_identical_successful_call_by_the_same_agent(self):
        c = conv([call(0, "get_invoices", result=INVOICES), call(1, "get_invoices", result=INVOICES)])
        assert guards.repeat(c, c["calls"][1]).startswith("repeat")
        assert guards.repeated_from(c, c["calls"][1]) is c["calls"][0]


    def test_repeat_allows_retrying_a_failed_call_and_other_agents(self):
        failed = conv([call(0, "get_invoices", executed=False, result='{"error": "down"}'), call(1, "get_invoices")])
        assert guards.repeat(failed, failed["calls"][1]) is None
        other = conv([call(0, "get_invoices", agent="drafter"), call(1, "get_invoices", agent="reviewer")])
        assert guards.repeat(other, other["calls"][1]) is None


    def test_repeat_allows_a_read_after_a_write(self):
        c = conv([call(0, "get_payments"), call(1, "issue_refund", {"payment_id": "PAY-5005"}), call(2, "get_payments")])
        assert guards.repeat(c, c["calls"][2]) is None


    def test_budget_counts_every_requested_call_including_invalid_ones(self):
        calls = [call(i, "get_invoices", {"x": i}, valid=(i % 2 == 0)) for i in range(guards.K + 1)]
        c = conv(calls)
        assert guards.budget(c, calls[guards.K - 1]) is None
        assert guards.budget(c, calls[guards.K]).startswith("budget")


    # --- refund_policy -------------------------------------------------------------------------


    def test_refund_policy_allows_one_refund_of_a_duplicate_charge(self):
        c = refund_conv([call(0, "get_payments"), call(1, "issue_refund", {"payment_id": "PAY-5005"})])
        assert guards.refund_policy(c, c["calls"][1]) is None


    def test_refund_policy_rules(self):
        cases = {"PAY-5004": "succeeded", "PAY-5007": "is_extra_charge", "PAY-6001": "exists", "PAY-9999": "exists"}
        for pid, rule in cases.items():
            c = refund_conv([call(0, "issue_refund", {"payment_id": pid})])
            assert f"refund_policy.{rule}" in guards.refund_policy(c, c["calls"][0]), pid


    def test_refund_policy_sees_earlier_refunds(self):
        prior = [{"id": "PAY-5008", "customer": "C-1001", "type": "refund", "amount": 25, "invoice": "INV-1002",
                  "status": "succeeded", "refund_of": "PAY-5006"}]
        c = refund_conv([call(0, "issue_refund", {"payment_id": "PAY-5006"}, executed=False, result='{"error": "refunded"}'),
                         call(1, "issue_refund", {"payment_id": "PAY-5005"})],
                        prior)
        assert "not_refunded" in guards.refund_policy(c, c["calls"][0])
        assert "is_extra_charge" in guards.refund_policy(c, c["calls"][1])
        refund_of_refund = refund_conv([call(0, "issue_refund", {"payment_id": "PAY-5008"})], prior)
        assert "is_charge" in guards.refund_policy(refund_of_refund, refund_of_refund["calls"][0])


    def test_refund_policy_one_refund_per_conversation(self):
        done = json.dumps({"refund_id": "PAY-5008", "refund_of": "PAY-5005", "amount": 25})
        c = refund_conv([call(0, "issue_refund", {"payment_id": "PAY-5005"}, result=done),
                         call(1, "issue_refund", {"payment_id": "PAY-5006"})])
        # after PAY-5005's refund, INV-1002 has 2 charges and 1 refund: no duplicate left
        assert "is_extra_charge" in guards.refund_policy(c, c["calls"][1])


    def test_refund_policy_only_applies_to_06(self):
        c = conv([call(0, "issue_refund", {"payment_id": "PAY-6001"})])
        assert guards.refund_policy(c, c["calls"][0]) is None


    # --- handoffs and time order ---------------------------------------------------------------


    def test_timeline_moves_the_managers_request_before_the_specialists_calls(self):
        c = manager_conv([call(0, "get_invoices", agent="specialist:billing")])
        events = [(kind, x["i"]) for kind, x in guards.timeline(c)]
        assert events == [("call", 1), ("call", 0), ("reply", 1)]


    def test_handoff_evidence(self):
        failed = manager_conv([call(0, "get_refund_status", {"invoice_id": "invoice_id1"}, executed=False,
                                    agent="specialist:billing", result='{"error": "not found"}')])
        assert "lookup_failed" in guards.handoff_evidence(failed, failed["calls"][-1])
        none = manager_conv([])
        assert "no_lookup" in guards.handoff_evidence(none, none["calls"][-1])
        ok = manager_conv([call(0, "get_invoices", agent="specialist:billing", result=INVOICES)])
        assert guards.handoff_evidence(ok, ok["calls"][-1]) is None


    # --- reply checks --------------------------------------------------------------------------
    def test_reply_checks(self):
        assert guards.empty_reply(conv([], final="  \n")) == "empty_reply"
        assert guards.tool_call_text(conv([], final='{"name":"get_refund_status","parameters":{}}')) == "tool_call_text"
        assert guards.tool_call_text(conv([], final="Your invoice INV-1002 is $25.")) is None
        assert guards.no_evidence(conv([], final="No invoices.")) is not None
        assert guards.no_evidence(conv([], final="Email only.", context="prefers email")) is None


    def test_claim_patterns(self):
        claims = ["I have issued a refund for INV-1002.", "I’ve processed your refund.", "We have now removed the two seats.",
                  "Your refund has been processed.", "I have gone ahead and refunded it.", "The extra seats have been removed."]
        not_claims = ["INV-0998 has already been refunded.", "I will now issue a refund for the extra charge.",
                      "I can request a refund for you.", "I have checked your invoices.", "Refunds are issued within 5 days."]
        for text in claims:
            assert guards.claimed_action(text), text
        for text in not_claims:
            assert guards.claimed_action(text) is None, text


    def test_unbacked_claim_needs_an_executed_write(self):
        text = "I have issued a refund for the extra charge."
        assert guards.unbacked_claim(conv([], final=text)) is not None
        done = conv([call(0, "issue_refund", {"payment_id": "PAY-5005"})], final=text, demo="06-state-and-workflows",
                    ledger=LEDGER)
        assert guards.unbacked_claim(done) is None
        blocked = conv([call(0, "issue_refund", {"invoice_id": "INV-2042"}, executed=False)], final=text)
        assert guards.unbacked_claim(blocked) is not None


    # --- sequential ----------------------------------------------------------------------------
    def test_sequential_first_fire_and_the_recorded_schema_check(self):
        c = conv([call(0, "get_refund_status", {"customer_email": "x"}, valid=False, executed=False),
                  call(1, "get_refund_status", {"invoice_id": "INV-0000"}, executed=False),
                  call(2, "get_invoices", result=INVOICES)], final="")
        s = guards.sequential(c)
        assert (s["outcome"], s["guard"], s["i"]) == ("stopped_at_call", "provenance", 1)


    def test_sequential_records_writes_before_a_stop(self):
        done = json.dumps({"refund_id": "PAY-5008", "refund_of": "PAY-5005", "amount": 25})
        c = refund_conv([call(0, "get_payments", result=json.dumps(LEDGER)),
                         call(1, "issue_refund", {"payment_id": "PAY-5005"}, result=done),
                         call(2, "issue_refund", {"payment_id": "PAY-5007"})])
        s = guards.sequential(c)
        assert (s["guard"], s["writes_before_stop"]) == ("refund_policy", [1])


    def test_sequential_reply_checks_and_delivery(self):
        assert guards.sequential(conv([call(0, "get_invoices", result=INVOICES)], final=""))["guard"] == "empty_reply"
        assert guards.sequential(conv([call(0, "get_invoices", result=INVOICES)]))["outcome"] == "delivered"


    # --- report code paths the dev set can't reach (labels, refunds, the cross-check) ---------
    def test_report_runs_on_labelled_synthetic_rows(self):
        convs = {}
        rows = []
        samples = [
            (conv([call(0, "get_invoices", result=INVOICES)]), "correct", None),
            (conv([call(0, "get_refund_status", {"invoice_id": "INV-0000"}, executed=False)], final="none"), "incorrect", None),
            (refund_conv([call(0, "get_payments", result=json.dumps(LEDGER)), call(1, "issue_refund", {"payment_id": "PAY-5007"})],
                         ), "failure", False),
        ]
        for n, (c, lab, claim) in enumerate(samples):
            c["key"] = f"{c['demo']}|t|m|X|Q{n}|1"
            c.update({"question": f"Q{n}", "seed": 1, "design": "X", "refund_truth": {1: "wrong"} if lab == "failure" else {}})
            convs[c["key"]] = c
            rows.append({"key": c["key"], "demo": c["demo"], "design": "X", "label": lab, "action_claim": claim,
                         "independent": guards.independent(c), "sequential": guards.sequential(c)})
        xc = conv([call(0, "issue_refund", {"invoice_id": "INV-2042", "amount": 100}, executed=False,
                        result='{"error": "Refund not allowed: INV-2042 is not one of the signed-in customer\'s invoices."}')],
                  request="Please refund invoice INV-2042.", demo="07-human-in-the-loop")
        xc.update({"key": "07-human-in-the-loop|proposal|S5|1", "question": "S5", "seed": 1, "design": ""})
        xrows = [{"key": xc["key"], "demo": xc["demo"], "design": "", "label": "correct", "action_claim": False,
                  "independent": guards.independent(xc), "sequential": guards.sequential(xc)}]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            report.overview(rows, convs)
            report.id_guards(rows, convs)
            report.repeat_guard(rows, convs)
            report.refund_guard(rows, convs)
            report.reply_checks(rows, convs)
            report.sequential(rows, convs)
            report.crosscheck(xrows, {xc["key"]: xc})
        text = out.getvalue()
        assert "wrong      refunds: guard fired on 1/1 (100%)" in text
        assert "#29 blocked, provenance passes:      1" in text


if __name__ == "__main__":
    unittest.main()
