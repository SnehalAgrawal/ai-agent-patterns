"""
The support agent with a human approval checkpoint.

The loop is the one from Articles #8, #23-#26: the model calls tools until it replies. The
read-only tools run immediately. A call to issue_refund never runs when the model makes it:

  1. code parses it and applies the hard checks; a failure goes back to the model as an error
  2. otherwise the run PAUSES: an approval request is saved with everything the reviewer
     needs and the agent's checkpoint (its messages so far), and the process can exit
  3. a reviewer decides (approve, modify, reject, clarify), or the request expires
  4. resume() loads the checkpoint, executes the approved action once (re-checking the
     current records first), gives the model the outcome as the tool result, and continues
     the loop so the model can reply to the customer

Every step is written to the audit trail in store.py.
"""
import hashlib
import json

from llm import chat as ollama_chat
from llm import tool_message
from policy import TTL_HOURS, action_sha, failed, hard_checks, may_decide, tier
from store import Store, plus_hours
from tools import DATA, POLICY, TOOLS, parse, run_read

SYSTEM_PROMPT = json.loads((DATA / "prompt.json").read_text())["system"]
RESULTS = json.loads((DATA / "decisions.json").read_text())["results_sent_to_the_model"]
LIMITS = POLICY["agent_limits"]
MAX_STEPS = LIMITS["max_model_calls_per_segment"]
MAX_TOKENS = LIMITS["max_tokens_per_call"]
# A checkpoint can only be resumed by code that would have produced it: same prompt, same tools.
VERSION = hashlib.sha256((SYSTEM_PROMPT + json.dumps(TOOLS, sort_keys=True)).encode()).hexdigest()[:16]
DECIDED = ("approved", "approved_with_changes", "rejected", "expired")


def _fill(template: dict, **values) -> dict:
    out = {}
    for key, value in template.items():
        if isinstance(value, str) and value.startswith("<") and value.endswith(">"):
            out[key] = values[key]
        else:
            out[key] = value
    return out


def _text(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False)


class Agent:
    def __init__(self, store: Store, seed: int = 1, chat=ollama_chat):
        self.store, self.seed, self.chat = store, seed, chat

    # --- 1. start a conversation -------------------------------------------------------------
    def start(self, conversation_id: str, message: str, now: str) -> dict:
        self.store.event(now, "customer", "customer_message", {"text": message}, conversation_id)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": message}]
        context = {"conversation_id": conversation_id, "customer_message": message, "evidence": [], "supersedes": None}
        return self._segment(messages, context, now)

    # --- the loop, from the start or from a checkpoint, until a reply or a pause ---------------
    def _segment(self, messages: list[dict], context: dict, now: str) -> dict:
        conversation_id = context["conversation_id"]
        record = {"model_calls": [], "calls": [], "final": None, "outcome": None, "request_id": None}
        for _ in range(MAX_STEPS):
            response = self.chat(messages, TOOLS, self.seed, MAX_TOKENS)
            record["model_calls"].append({k: response[k] for k in (
                "finish_reason", "input_tokens", "cached_tokens", "output_tokens", "seconds", "error")}
                | {"tool_calls": len(response["tool_calls"])})
            if response["error"]:
                record["outcome"] = "error"
                return record
            if not response["tool_calls"]:
                record["final"], record["outcome"] = response["content"], "answered"
                self.store.event(now, "agent", "reply", {"text": response["content"],
                                 "sha256": hashlib.sha256(response["content"].encode()).hexdigest()}, conversation_id)
                return record
            messages.append(response["message"])
            pending = None
            for call in response["tool_calls"]:
                if call["name"] != "issue_refund":
                    result = run_read(call["name"], call["arguments"], self.store.issued())
                    record["calls"].append(result)
                    context["evidence"].append({"tool": call["name"], "arguments": call["arguments"],
                                                "result": result["result_text"]})
                    self.store.event(now, "agent", "tool_call", {"tool": call["name"], "arguments": call["arguments"],
                                     "executed": result["executed"]}, conversation_id)
                    messages.append(tool_message(call["name"], result["result_text"]))
                    continue
                entry, result = self._refund_call(call["arguments"], pending, context, now)
                if result is None:  # a valid proposal: its result arrives when the run resumes
                    pending = entry | {"insert_at": len(messages)}
                    record["calls"].append(entry)
                    continue
                record["calls"].append(entry | {"result_text": _text(result)})
                messages.append(tool_message("issue_refund", _text(result)))
            if pending:
                record["request_id"] = self._pause(pending, messages, context, now)
                record["outcome"] = "paused"
                return record
        record["outcome"] = "step_limit"
        return record

    def _refund_call(self, arguments: str, pending: dict | None, context: dict, now: str):
        """Returns (record, result). result None means: this call becomes an approval request."""
        conversation_id = context["conversation_id"]
        entry = {"tool": "issue_refund", "arguments": arguments, "valid": False, "executed": False}
        self.store.event(now, "agent", "refund_proposed", {"arguments": arguments}, conversation_id)
        if pending:
            entry["refused"] = "one_at_a_time"
            return entry, RESULTS["one_at_a_time"]
        args, error = parse("issue_refund", arguments)
        if error:
            entry["refused"] = "invalid"
            return entry, {"error": error}
        entry |= {"valid": True, "invoice_id": args["invoice_id"].strip(), "amount": args["amount"],
                  "reason": args["reason"]}
        for earlier in self.store.requests(conversation_id):
            if earlier["action"]["invoice_id"] != entry["invoice_id"]:
                continue
            if earlier["status"] in DECIDED:
                entry["refused"] = "already_decided"
                self.store.event(now, "system", "refund_refused", {"why": "already_decided", "request": earlier["id"]},
                                 conversation_id, earlier["id"])
                return entry, {"error": RESULTS["already_decided"]["error"].replace("<id>", earlier["id"]).replace(
                    "<invoice>", entry["invoice_id"]).replace("<status>", earlier["status"])}
            if earlier["status"] == "pending":
                entry["refused"] = "already_pending"
                return entry, {"error": f"The refund request {earlier['id']} for {entry['invoice_id']} is already "
                                        "waiting for review. Do not request it again."}
        checks = hard_checks(entry["invoice_id"], entry["amount"], self.store.issued())
        entry["checks"] = checks
        problem = failed(checks)
        if problem:
            entry["refused"] = "hard_check"
            self.store.event(now, "system", "refund_blocked", {"check": problem}, conversation_id)
            return entry, {"error": RESULTS["blocked"]["error"].replace("<the failed hard check>", problem)}
        entry["tier"] = tier(entry["invoice_id"], entry["amount"])
        return entry, None

    def _pause(self, pending: dict, messages: list[dict], context: dict, now: str) -> str:
        request_id = self.store.next_request_id()
        action = {"invoice_id": pending["invoice_id"], "amount": pending["amount"]}
        sha = action_sha(action)
        view = {
            "request_id": request_id, "created_at": now, "expires_at": plus_hours(now, TTL_HOURS),
            "customer_message": context["customer_message"],
            "action": {"tool": "issue_refund", **action}, "action_sha256": sha,
            "agent_reason": pending["reason"],
            "checks": pending["checks"], "tier": pending["tier"]["tier"],
            "may_decide": pending["tier"]["may_decide"], "policy_facts": pending["tier"]["facts"],
            "evidence": context["evidence"], "supersedes": context["supersedes"],
        }
        checkpoint = {"version": VERSION, "seed": self.seed, "messages": messages, "insert_at": pending["insert_at"],
                      "context": context}
        self.store.create_request({"id": request_id, "conversation_id": context["conversation_id"],
                                   "created_at": now, "expires_at": view["expires_at"], "tier": view["tier"],
                                   "action": action, "action_sha256": sha, "view": view, "checkpoint": checkpoint,
                                   "supersedes": context["supersedes"]})
        self.store.event(now, "system", "approval_requested", {"action": action, "action_sha256": sha,
                         "tier": view["tier"], "expires_at": view["expires_at"]}, context["conversation_id"], request_id)
        return request_id

    # --- 3. a reviewer decides -----------------------------------------------------------------
    def decide(self, request_id: str, decision: str, reviewer: str, role: str, shown_sha: str, now: str,
               amount=None, reason: str | None = None, question: str | None = None) -> dict:
        req = self.store.get(request_id)
        if req is None:
            return {"ok": False, "error": f"No request {request_id}."}

        def refuse(why: str) -> dict:
            self.store.event(now, reviewer, "decision_refused", {"decision": decision, "role": role, "why": why},
                             req["conversation_id"], request_id)
            return {"ok": False, "error": why}

        if req["status"] != "pending":
            return refuse(f"{request_id} is not pending (status: {req['status']}).")
        if now > req["expires_at"]:
            self._expire(req, now)
            return refuse(f"{request_id} expired at {req['expires_at']}.")
        if shown_sha != req["action_sha256"]:
            return refuse("The action you were shown is not the action in this request.")
        if not may_decide(req["tier"], role):
            return refuse(f"Role '{role}' may not decide a request in tier '{req['tier']}'.")
        record = {"decision": decision, "reviewer": reviewer, "role": role, "at": now}
        final_action, status = None, None
        if decision == "approve":
            final_action, status = req["action"], "approved"
        elif decision == "modify":
            if amount is None:
                return refuse("modify needs a new amount.")
            final_action = {"invoice_id": req["action"]["invoice_id"], "amount": amount}
            problem = failed(hard_checks(final_action["invoice_id"], amount, self.store.issued()))
            if problem:
                return refuse(f"The modified action fails a hard check: {problem}.")
            new_tier = tier(final_action["invoice_id"], amount)["tier"]
            if not may_decide(new_tier, role):
                return refuse(f"The modified action is in tier '{new_tier}', which role '{role}' may not decide.")
            record |= {"requested_amount": req["action"]["amount"], "amount": amount, "tier": new_tier,
                       "note": reason or "The reviewer approved a partial refund."}
            status = "approved_with_changes"
        elif decision == "reject":
            if not reason:
                return refuse("reject needs a reason.")
            record["reason"], status = reason, "rejected"
        elif decision == "clarify":
            if not question:
                return refuse("clarify needs a question.")
            record["question"], status = question, "clarification_requested"
        else:
            return refuse(f"Unknown decision '{decision}'.")
        if final_action:
            record["final_action_sha256"] = action_sha(final_action)
        if not self.store.transition(request_id, status, record, final_action):
            return refuse(f"{request_id} was decided by someone else first.")
        self.store.event(now, reviewer, "decision_recorded", record, req["conversation_id"], request_id)
        return {"ok": True, "status": status}

    def _expire(self, req: dict, now: str) -> bool:
        changed = self.store.transition(req["id"], "expired", {"decision": "expire", "reviewer": "system",
                                                                "at": now, "expired_at": req["expires_at"]})
        if changed:
            self.store.event(now, "system", "expired", {"expires_at": req["expires_at"]}, req["conversation_id"],
                             req["id"])
        return changed

    def expire_due(self, now: str) -> list[str]:
        return [r["id"] for r in self.store.requests(status="pending") if now > r["expires_at"] and self._expire(r, now)]

    # --- 4. resume -------------------------------------------------------------------------------
    def _execute(self, req: dict, now: str) -> dict:
        """Runs the approved refund at most once, after re-checking the current records."""
        final = req["final_action"]
        base = {"request_id": req["id"], "invoice_id": final["invoice_id"]}
        self.store.db.execute("BEGIN IMMEDIATE")  # no other writer between the re-check and the insert
        try:
            refund = self.store.refund_for(req["id"])
            if refund is None:
                problem = failed(hard_checks(final["invoice_id"], final["amount"], self.store.issued()))
                if problem:
                    self.store.event(now, "system", "execution_refused", {"check": problem}, req["conversation_id"],
                                     req["id"])
                    self.store.db.execute("COMMIT")
                    return base | {"status": "approved_not_executed", "refund_issued": False,
                                   "requested_amount": final["amount"],
                                   "reason": f"The records changed before the refund ran: {problem}. Nothing was "
                                             "refunded."}
                refund, _ = self.store.insert_refund(req["id"], final["invoice_id"], final["amount"], now)
                self.store.event(now, "system", "refund_executed", refund, req["conversation_id"], req["id"])
            else:
                self.store.event(now, "system", "execution_skipped", {"refund_id": refund["refund_id"],
                                 "why": "already executed for this request"}, req["conversation_id"], req["id"])
            self.store.db.execute("COMMIT")
        except Exception:
            self.store.db.execute("ROLLBACK")
            raise
        if req["status"] == "approved":
            return _fill(RESULTS["approve"], **base, refund_id=refund["refund_id"], amount=final["amount"])
        return _fill(RESULTS["modify"], **base, refund_id=refund["refund_id"], amount=final["amount"],
                     requested_amount=req["action"]["amount"], note=req["decision"]["note"])

    def outcome_result(self, req: dict, now: str) -> dict:
        """The issue_refund tool result for a decided request (executing it first if approved)."""
        base = {"request_id": req["id"], "invoice_id": req["action"]["invoice_id"],
                "requested_amount": req["action"]["amount"]}
        if req["status"] in ("approved", "approved_with_changes"):
            return self._execute(req, now)
        if req["status"] == "rejected":
            return _fill(RESULTS["reject"], **base, reason=req["decision"]["reason"])
        if req["status"] == "expired":
            return _fill(RESULTS["expire"], **base)
        if req["status"] == "clarification_requested":
            return _fill(RESULTS["clarify"], request_id=req["id"], question=req["decision"]["question"])
        raise ValueError(req["status"])

    def resume(self, request_id: str, now: str) -> dict:
        req = self.store.get(request_id)
        if req is None:
            return {"outcome": "refused", "error": f"No request {request_id}."}
        if req["status"] == "pending":
            if now <= req["expires_at"]:
                return {"outcome": "refused", "error": f"{request_id} is still waiting for a decision."}
            self._expire(req, now)
            req = self.store.get(request_id)
        if req["completed_at"]:
            return {"outcome": "refused", "error": f"{request_id} was already resumed and answered."}
        checkpoint = req["checkpoint"]
        if checkpoint["version"] != VERSION:
            return {"outcome": "refused", "error": "The prompt or tools changed since this run paused; it can't be "
                                                   "resumed safely. Start a new conversation."}
        result = self.outcome_result(req, now)
        self.store.event(now, "system", "resumed", {"result": result}, req["conversation_id"], request_id)
        record = self.continue_with(req, result, now)
        if record["outcome"] != "error":
            self.store.mark_completed(request_id, now)
        return record | {"tool_result": result}

    def continue_with(self, req: dict, result: dict, now: str) -> dict:
        """Continues a paused run from its checkpoint with the given issue_refund result."""
        checkpoint = req["checkpoint"]
        messages = [dict(m) for m in checkpoint["messages"]]
        messages.insert(checkpoint["insert_at"], tool_message("issue_refund", _text(result)))
        context = dict(checkpoint["context"])
        context["evidence"] = list(context["evidence"])
        if req["status"] == "clarification_requested":
            context["supersedes"] = req["id"]
        seed_before, self.seed = self.seed, checkpoint["seed"]
        try:
            return self._segment(messages, context, now)
        finally:
            self.seed = seed_before
