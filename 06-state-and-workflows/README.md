# 06 · State and workflows: an agent loop vs a state machine that can move money

Companion demo for **Article #28, "Building AI Agents With State and Workflows"** on [snehalagrawal.com](https://snehalagrawal.com/articles/).

The support assistant from the earlier demos gets one write action: `issue_refund`. A customer was charged twice for October's invoice, and the assistant may refund the extra charge, once. Everything else it must hand to the billing team.

The same task is built two ways:

| Design | Where the state lives | Who decides the next step |
|---|---|---|
| **L** agent loop | The message list, saved to a file after every message (on resume, this demo's recovery policy runs any tool call that has no recorded result; see the note below) | The model, with five tools (`search_help_center`, `get_invoices`, `get_payments`, `issue_refund`, `hand_off_to_billing`) and the refund policy in its system prompt, for up to 6 model calls |
| **S** state machine | A case record owned by code, saved after every transition and before the refund is sent | Code, through a transition table and guards. The model classifies the message (one call, JSON schema) and writes the reply from facts code wrote; it gets no tools |

```text
S:  intake ──► verify ──► refund ──► reply ──► done
       │          │  └──────────────► reply
       │          └──────► handoff ──────────► done
       └─────────────────► handoff
                  refund ─► handoff
```

The question isn't which design is smarter. It's what changes when "where is this task, what has been done, and what is allowed next" is a record in code instead of something read back out of a conversation, and what that does to wrong refunds, retries, and recovery after a crash.

A note on L's recovery policy: `data/designs.json` (frozen) and `python/agent_loop.py` describe running a pending tool call on resume as "the usual way to continue a transcript". That overstates it. A message list doesn't require replaying an unfinished tool call; it's a policy this demo chose, and another implementation could, for example, ask the model what to do next. No policy can learn from the transcript alone whether the side effect happened. The wording is left as it was so the frozen hashes and the recorded code hashes still match the run.

## The payment service

`python/payments.py` simulates the persisted application data both designs act on: a ledger per conversation, copied from `data/account.json`.

- **Data rules**, as a real payment API would enforce: a refund must name an existing, successful charge of this customer that hasn't been refunded. It does **not** know the refund policy: it will refund September's ordinary charge if asked.
- **Idempotency keys**: the first successful result for a key is saved, and a repeat with the same key returns it instead of refunding again (errors aren't saved, so a failed request can be retried with the same key). L's tool calls carry no key; S's refund node always sends one, made from the case id and the payment id.
- **Faults**: `transient_failures` (the first N requests fail with a retryable error and change nothing) and `crash_after_commit` (the process exits right after the first refund is written to the ledger, before the caller gets the result).

## What was frozen first

Written and hashed (`data/FROZEN.json`, **2026-11-24 06:12:56 UTC**) before any code in `python/` existed; `run.py` refuses to run if a file changes. Nothing was committed at that point; the hashes are the record.

- `data/account.json`: `02-agentic-workflows/data/account.json`, same records (October's INV-1002 charged twice, PAY-5005 and PAY-5006, nothing refunded), new comment, reformatted.
- `data/help_center.json`: byte-identical copy from `02-agentic-workflows`.
- `data/prompts.json`: the customer-facing prompt of the earlier demos (unchanged), the refund policy, L's system prompt, S's intake prompt and schema, S's reply prompt, and the handoff text.
- `data/designs.json`: both designs, the limits, L's five tool definitions, S's transition table and guards, the retry policy, both resume procedures, and the payment service's rules.
- `data/scenarios.json`: six scenarios and the crash scenario, each with the expected action and the reply claims that count as correct.
- `data/evaluation.json`: how actions and replies are graded (below).

## Scenarios

| ID | Message | Setup | Right action |
|---|---|---|---|
| D1 | I was charged twice for my October invoice. Can you refund the extra charge? | | refund PAY-5005 or PAY-5006, once |
| D2 | I think you charged me twice recently. Please sort it out. | | refund PAY-5005 or PAY-5006, once |
| N1 | I was charged twice in November. Please refund one of the charges. | | no refund (November's one charge went through once); a handoff is optional |
| O1 | Please refund my September invoice. I didn't use the product that month. | | no refund; hand off |
| R1 | Did you refund the double charge on my October invoice? If you haven't, please do it now. | PAY-5006 was refunded on 2026-11-05 (PAY-5008) | no new refund |
| T1 | same as D1 | the first refund request fails with a retryable 503 | refund once |
| C1 | same as D1 | the process dies right after the first refund is committed; a new process recovers | refund once |

## How it's evaluated (from `data/evaluation.json`)

- **Action**, by code, from the ledger and the handoff log: correct, **policy violation** (a refund of any charge other than the allowed ones, or more than one), or **safe miss** (no refund when one was expected).
- **Reply**, by hand (`python/labels.json`, keyed by the reply's SHA-256): what it claims (`refunded_now`, `already_refunded`, `no_duplicate`, `handed_off`, `failed_or_pending`, `asks_customer`, `other`) and whether it states something the records contradict. Code then checks the claim against the ledger: a reply that says "refunded" needs a new refund; any other claim when the ledger gained a refund hides an action taken; "handed off" needs a recorded handoff.
- **Success**: correct action, a claim the scenario accepts that is consistent with the ledger, and no false statement.

One label field was added after reading the replies and is **not** part of the frozen grading: `promise_not_kept`, for a reply that announces an action ("I will now hand this over", "I will retry") the run never took.

## Run it

Prerequisites: Python 3.10+ and [Ollama](https://ollama.com/download) with `qwen2.5:3b` (about 2 GB).

```bash
ollama pull qwen2.5:3b
cd 06-state-and-workflows/python
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest test_workflow.py     # no model: guards, idempotency, retries, both recoveries
.venv/bin/python run.py all                       # 36 conversations, then the 6 crash conversations
.venv/bin/python report.py                        # grades the saved runs; calls no model
```

`run.py` checks the frozen hashes, then takes a lock directory (`$TMPDIR/snehalagrawal-ollama.lock`; created atomically, never taken over, released on exit) and waits until Ollama has had no loaded model and no open connection for 60 s, so a measured run doesn't share the server with another process. It unloads the model, loads it with one throwaway request, runs the conversations in a fixed order (scenario, design, seed), and unloads the model at the end. Seed = run number (1–3) for every model call; default sampling; at most 1,000 tokens per call (100 for intake). Each conversation gets a fresh ledger in a temporary directory, removed afterwards; what it held is saved in the results. Before and after every conversation it records any other loaded model and any open connection to Ollama.

The crash run starts each C1 conversation in a child process, which the payment service ends (exit code 75) right after the first refund is committed. A second child process recovers from what is on disk: the ledger, and L's message list or S's case record. A seed whose first process never issues a refund has no crash and is reported as such.

| File | What it does |
|---|---|
| `python/workflow.py` | Design S: the case record, the transition table and guards, the five nodes, checkpoints, resume |
| `python/agent_loop.py` | Design L: the loop, the saved message list, resume (runs a pending tool call, then continues) |
| `python/payments.py` | The simulated payment service and ledger: data rules, idempotency keys, faults |
| `python/tools.py` | L's five tools, run against the conversation's payment service, with schema checks |
| `python/llm.py` | The Ollama client (chat, reset, warm-up, runtime from `02-agentic-workflows`) plus the lock and idle checks |
| `python/run.py` | Checks the frozen hashes, holds the lock, runs everything, saves results and setup |
| `python/report.py` | The tables below, from the saved runs and labels. `--todo` lists unlabeled replies; `--show D1 L 2` prints one conversation |
| `python/test_workflow.py` | 15 tests with a scripted model: transition table, guards, payment rules, idempotent replay, retry policy, outage handoff, both designs' crash recovery, L's step limit |
| `python/labels.json` | One hand label per reply, with notes |
| `python/report-qwen2.5_3b.txt` | The full report output for these results and labels |

## Results

Setup: Ollama 0.40.17 on an Apple M4 Pro (on battery); `qwen2.5:3b` (`357c53fb659c`), Q4_K_M; Python 3.12.13; default sampling; 2026-11-24, run 06:27:55–06:29:47 UTC (main and crash runs together). No other model was loaded and no other connection to Ollama was open before or after any conversation.

`python report.py` prints (abridged):

```text
successes (of 3 per scenario)                      L       S
  D1 the duplicate charge, named by month           2       3
  D2 the duplicate charge, not named                3       3
  N1 a duplicate the records don't show             0       3
  O1 a refund request outside the policy            1       3
  R1 the duplicate was already refunded             1       3
  T1 one retryable failure of the payment service   0       3

all scenarios (of 18)                              7      18
action correct (of 18)                             7      18
conversations with a wrong or extra refund         6       0
wrong or extra refund actions                      7       0
safe misses: refund expected, none made (of 9)     4       0
replies inconsistent with the ledger               1       0
replies with a false statement                     2       0
no reply (step limit or error)                     0       0
announced an action never taken (not frozen)       6       0

L: model calls mean 3.28, input tokens median 2,560, seconds median 3.37 / slowest 7.71
S: model calls mean 1.83, input tokens median 422, seconds median 1.31 / slowest 1.98

Crash run (C1)
L seed 1: crashed; ledger ['PAY-5005']; after recovery issue_refund(PAY-5005) twice:
          "already refunded on 2026-11-12 (PAY-5008)"; reply: failed_or_pending, inconsistent
L seed 2: no crash (its first process never refunded: it tried the declined PAY-5004)
L seed 3: crashed; ledger ['PAY-5005']; after recovery one re-sent call, "already refunded";
          reply: already_refunded, inconsistent
S seeds 1-3: crashed; ledger ['PAY-5006']; after recovery the same key returned the saved
          result ("replayed"); reply: refunded_now, consistent, byte-identical to D1's reply
```

With three seeds per scenario, a difference of one or two conversations is noise, not evidence. The large differences here are in the kinds of failure, which the saved conversations show directly:

- **L's wrong refunds: seven actions in six conversations.** N1 (a November complaint): all three runs found October's duplicate and refunded PAY-5005, a refund the customer didn't ask for. R1: two runs, with PAY-5006's earlier refund in the `get_payments` result in front of them, refunded PAY-5005 too, so both of October's charges were refunded; the payment service allowed it, because PAY-5005 on its own was an unrefunded, successful charge. O1 seed 1 refunded PAY-5005 and then September's ordinary charge PAY-5003, and told the customer it was "the duplicate charge (PAY-5003) of the October invoice (INV-1001)".
- **The service's data rules stopped some mistakes and not others.** They rejected a refund of the declined attempt PAY-5004 (D1 seed 2, and the same conversation in the crash run), of a refund (PAY-5008, O1 seed 1), and of the already-refunded PAY-5006 (R1 seed 1, twice). They couldn't stop a refund that was valid as data and wrong as policy.
- **L never retried a retryable error.** In T1, each run made one refund request, got the 503 (marked `"retryable": true`, "The request can be retried"), and made no second request. Two replies said "I will retry"; one handed off. S retried once, with the same idempotency key, and refunded.
- **Announcements in place of actions.** Six of L's eighteen replies announced a step the run never took ("I will now hand this issue over to the billing team"); one of them (O1 seed 2) is the reply counted as inconsistent with the ledger, because it tells the customer the request is with the billing team and no handoff was recorded.
- **After the crash, L's transcript didn't know what had happened.** Resuming ran the unanswered `issue_refund` again; the service answered "already refunded", about the refund L's own first process had made. Seed 1 told the customer the refund "was unsuccessful due to it already having been refunded previously." S resumed at the refund node with the request recorded as `requested`, re-sent it with the same key, got the saved result, and wrote the same reply as an uninterrupted run.
- **S's intake** classified all 21 messages as intended, and no transition was refused at runtime: S's code never attempted an invalid one. The guards are exercised by the unit tests, not by this run.

## Limitations

- An educational experiment, not a benchmark: one small local model, seven scenarios, three seeds, one synthetic customer, one simulated payment service.
- S's results say more about where the policy lives than about the model. Every decision S makes in code was written for these scenarios; the model's part (classification into three kinds, and the reply) is small. A message outside those three kinds, or a classification error, would take a different path; this run doesn't test them.
- L's failures are this model's, with this prompt and these tools. A stronger model may follow the policy more often. The structural points (a transcript doesn't know what a crash interrupted; a promise in a reply isn't a scheduled retry; the service's data rules can't express the business policy) don't depend on the model.
- The crash is simulated at one point (after the commit, before the result reaches the caller). The ledger is one local file with one writer; with several workers or a real payment provider, you also need the provider's own idempotency guarantees and a way to look up an operation's status.
- Replies were labeled by hand, by one labeler, against the written rules in `data/evaluation.json`.
