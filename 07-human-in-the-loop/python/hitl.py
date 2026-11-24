"""
Walk through the approval lifecycle by hand. Every command is a separate process; the paused
run lives in the SQLite file (default ../state/hitl.sqlite), not in memory.

  python hitl.py ask "I was charged twice for my October invoice. Please refund the extra charge."
  python hitl.py pending
  python hitl.py show APR-0001
  python hitl.py decide APR-0001 approve --reviewer ana --role support --sha <action_sha256 from show>
  python hitl.py decide APR-0001 modify  --reviewer ana --role support --sha <...> --amount 12.50 --reason "..."
  python hitl.py decide APR-0001 reject  --reviewer ana --role support --sha <...> --reason "..."
  python hitl.py decide APR-0001 clarify --reviewer ana --role support --sha <...> --question "..."
  python hitl.py resume APR-0001
  python hitl.py expire                     # expire every pending request past its deadline
  python hitl.py audit [APR-0001]

--now overrides the clock (ISO 8601 UTC, e.g. 2026-11-13T12:00:00Z), to try expiry without waiting.
Needs Ollama with qwen2.5:3b for ask and resume (OLLAMA_CHAT_MODEL to change it).
"""
import argparse
import json
import uuid
from pathlib import Path

from agent import Agent
from store import Store, utc_now

DEFAULT_DB = Path(__file__).resolve().parent.parent / "state" / "hitl.sqlite"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--db", default=str(DEFAULT_DB))
parser.add_argument("--now", default=None, help="override the clock (ISO 8601 UTC)")
parser.add_argument("--seed", type=int, default=1)
sub = parser.add_subparsers(dest="command", required=True)
ask = sub.add_parser("ask")
ask.add_argument("message")
sub.add_parser("pending")
show = sub.add_parser("show")
show.add_argument("request_id")
decide = sub.add_parser("decide")
decide.add_argument("request_id")
decide.add_argument("decision", choices=["approve", "modify", "reject", "clarify"])
decide.add_argument("--reviewer", required=True)
decide.add_argument("--role", required=True, choices=["support", "billing_lead"])
decide.add_argument("--sha", required=True, help="the action_sha256 you were shown")
decide.add_argument("--amount", type=float)
decide.add_argument("--reason")
decide.add_argument("--question")
resume = sub.add_parser("resume")
resume.add_argument("request_id")
sub.add_parser("expire")
audit = sub.add_parser("audit")
audit.add_argument("request_id", nargs="?")
args = parser.parse_args()

Path(args.db).parent.mkdir(parents=True, exist_ok=True)
store = Store(args.db)
agent = Agent(store, seed=args.seed)
now = args.now or utc_now()


def show_request(request_id: str) -> None:
    req = store.get(request_id)
    if req is None:
        raise SystemExit(f"No request {request_id}.")
    print(json.dumps(req["view"] | {"status": req["status"], "decision": req["decision"]}, indent=1,
                     ensure_ascii=False))


def show_record(record: dict) -> None:
    for call in record.get("calls", []):
        note = call.get("refused") or ("executed" if call.get("executed") else "waiting for review")
        print(f"  tool  {call['tool']} {call['arguments']}  ({note})")
    if record.get("tool_result"):
        print(f"  issue_refund result: {json.dumps(record['tool_result'])}")
    if record["outcome"] == "paused":
        print(f"\nPaused: {record['request_id']} is waiting for a reviewer. Nothing has been refunded.")
        print(f"  python hitl.py show {record['request_id']}")
    elif record.get("final") is not None:
        print(f"\nReply to the customer:\n{record['final']}")
    else:
        print(f"\nOutcome: {record['outcome']} {record.get('error', '')}")


if args.command == "ask":
    show_record(agent.start(f"conv-{uuid.uuid4().hex[:8]}", args.message, now))
elif args.command == "pending":
    for req in store.requests(status="pending"):
        print(f"{req['id']}  {req['tier']:<13}  {req['action']['invoice_id']}  ${req['action']['amount']}  "
              f"expires {req['expires_at']}")
elif args.command == "show":
    show_request(args.request_id)
elif args.command == "decide":
    result = agent.decide(args.request_id, args.decision, args.reviewer, args.role, args.sha, now,
                          amount=args.amount, reason=args.reason, question=args.question)
    print(f"Recorded: {result['status']}" if result["ok"] else f"Refused: {result['error']}")
elif args.command == "resume":
    show_record(agent.resume(args.request_id, now))
elif args.command == "expire":
    expired = agent.expire_due(now)
    print("Expired: " + (", ".join(expired) if expired else "none"))
elif args.command == "audit":
    events = store.events(request_id=args.request_id) if args.request_id else store.events()
    for e in events:
        print(f"{e['seq']:>4}  {e['at']}  {e['request_id'] or '-':<9}  {e['actor']:<14}  {e['type']:<20}  "
              f"{json.dumps(e['data'], ensure_ascii=False)[:120]}")
store.close()
