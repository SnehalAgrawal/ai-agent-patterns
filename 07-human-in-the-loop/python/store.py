"""
Durable state for paused runs, in one SQLite file (Python's standard library, no server).

  requests  one row per approval request: the action, the evidence the reviewer sees, the
            agent's checkpoint (its messages up to the pause), the status, and the decision
  refunds   one row per executed refund; UNIQUE(request_id) makes execution idempotent
  events    the audit trail: append-only, one row per thing that happened, in order

A paused run lives here, not in a process. Any process that opens the same file can show the
request, record a decision, or resume the run. Times are ISO 8601 UTC strings ("...Z"), so
they compare correctly as text.
"""
import json
import sqlite3
from datetime import datetime, timedelta, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  status TEXT NOT NULL,              -- pending, approved, approved_with_changes, rejected, expired,
                                     -- clarification_requested
  tier TEXT NOT NULL,
  action TEXT NOT NULL,              -- JSON {"invoice_id", "amount"}
  action_sha256 TEXT NOT NULL,
  view TEXT NOT NULL,                -- JSON: everything the reviewer is shown
  checkpoint TEXT NOT NULL,          -- JSON: the agent's messages and where the result goes
  supersedes TEXT,
  decision TEXT,                     -- JSON, set once
  final_action TEXT,                 -- JSON: the action as approved (after any modification)
  completed_at TEXT                  -- set when the resumed run has replied to the customer
);
CREATE TABLE IF NOT EXISTS refunds (
  refund_id TEXT PRIMARY KEY,
  request_id TEXT NOT NULL UNIQUE,
  invoice_id TEXT NOT NULL,
  amount REAL NOT NULL,
  executed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  at TEXT NOT NULL,
  conversation_id TEXT,
  request_id TEXT,
  actor TEXT NOT NULL,
  type TEXT NOT NULL,
  data TEXT NOT NULL
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def plus_hours(iso: str, hours: float) -> str:
    moment = datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return (moment + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


class Store:
    def __init__(self, path: str):
        self.db = sqlite3.connect(path, isolation_level=None)  # autocommit; transactions are explicit
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    # --- audit trail -----------------------------------------------------------------------
    def event(self, at: str, actor: str, kind: str, data: dict, conversation_id=None, request_id=None) -> None:
        self.db.execute("INSERT INTO events (at, conversation_id, request_id, actor, type, data) VALUES (?,?,?,?,?,?)",
                        (at, conversation_id, request_id, actor, kind, json.dumps(data, ensure_ascii=False)))

    def events(self, request_id: str | None = None, conversation_id: str | None = None) -> list[dict]:
        sql, params = "SELECT * FROM events", []
        if request_id:
            sql, params = sql + " WHERE request_id = ?", [request_id]
        elif conversation_id:
            sql, params = sql + " WHERE conversation_id = ?", [conversation_id]
        rows = self.db.execute(sql + " ORDER BY seq", params).fetchall()
        return [dict(r) | {"data": json.loads(r["data"])} for r in rows]

    # --- requests --------------------------------------------------------------------------
    def next_request_id(self) -> str:
        count = self.db.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        return f"APR-{count + 1:04d}"

    def create_request(self, row: dict) -> None:
        self.db.execute(
            "INSERT INTO requests (id, conversation_id, created_at, expires_at, status, tier, action, action_sha256,"
            " view, checkpoint, supersedes) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (row["id"], row["conversation_id"], row["created_at"], row["expires_at"], "pending", row["tier"],
             json.dumps(row["action"]), row["action_sha256"], json.dumps(row["view"], ensure_ascii=False),
             json.dumps(row["checkpoint"], ensure_ascii=False), row.get("supersedes")))

    def get(self, request_id: str) -> dict | None:
        row = self.db.execute("SELECT * FROM requests WHERE id = ?", (request_id,)).fetchone()
        if row is None:
            return None
        out = dict(row)
        for key in ("action", "view", "checkpoint", "decision", "final_action"):
            out[key] = json.loads(out[key]) if out[key] else None
        return out

    def requests(self, conversation_id: str | None = None, status: str | None = None) -> list[dict]:
        sql, params, where = "SELECT id FROM requests", [], []
        if conversation_id:
            where.append("conversation_id = ?")
            params.append(conversation_id)
        if status:
            where.append("status = ?")
            params.append(status)
        if where:
            sql += " WHERE " + " AND ".join(where)
        return [self.get(r["id"]) for r in self.db.execute(sql + " ORDER BY id", params).fetchall()]

    def transition(self, request_id: str, new_status: str, decision: dict | None = None,
                   final_action: dict | None = None) -> bool:
        """
        pending -> new_status, only if the request is still pending. Returns False if another
        decision (or the expiry) got there first: a compare-and-set, so two reviewers deciding
        at the same moment can't both win.
        """
        cursor = self.db.execute(
            "UPDATE requests SET status = ?, decision = ?, final_action = ? WHERE id = ? AND status = 'pending'",
            (new_status, json.dumps(decision, ensure_ascii=False) if decision else None,
             json.dumps(final_action) if final_action else None, request_id))
        return cursor.rowcount == 1

    def mark_completed(self, request_id: str, at: str) -> None:
        self.db.execute("UPDATE requests SET completed_at = ? WHERE id = ?", (at, request_id))

    # --- refunds ---------------------------------------------------------------------------
    def issued(self) -> list[dict]:
        return [dict(r) | {"amount": r["amount"]} for r in
                self.db.execute("SELECT refund_id, request_id, invoice_id, amount FROM refunds ORDER BY refund_id")]

    def refund_for(self, request_id: str) -> dict | None:
        row = self.db.execute("SELECT * FROM refunds WHERE request_id = ?", (request_id,)).fetchone()
        return dict(row) if row else None

    def insert_refund(self, request_id: str, invoice_id: str, amount: float, at: str) -> tuple[dict, bool]:
        """Records a refund once per request. Returns (refund, created): created is False if it already existed."""
        refund_id = "RF-" + request_id.removeprefix("APR-")
        try:
            self.db.execute("INSERT INTO refunds (refund_id, request_id, invoice_id, amount, executed_at)"
                            " VALUES (?,?,?,?,?)", (refund_id, request_id, invoice_id, amount, at))
            return self.refund_for(request_id), True
        except sqlite3.IntegrityError:
            return self.refund_for(request_id), False
