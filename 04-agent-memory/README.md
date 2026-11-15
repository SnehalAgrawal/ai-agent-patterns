# 04 · Agent memory: what should an agent carry between conversations?

Companion demo for **Article #26, "Memory in AI Agents: What Does an Agent Actually Need to Remember?"** on [snehalagrawal.com](https://snehalagrawal.com/articles/) (not published yet; this README will link to it when it is).

> **Status: design frozen, not yet implemented or run.** This README describes the experiment as frozen in `data/`. The system code, the run, and the results come after the frozen design is committed.

The support assistant from [`02-agentic-workflows`](../02-agentic-workflows/) and [`03-multi-agent`](../03-multi-agent/) has talked to the same customer three times before. In a new conversation, it gets the customer's question and, depending on the design, some memory of those earlier conversations. It can always look up the customer's current records with the same five read-only tools.

The question isn't whether memory helps. It's what should be carried forward, what should be looked up, and what happens when something remembered disagrees with the current records.

## The four memory designs

| Design | What carries over from the earlier conversations |
|---|---|
| **N** no memory | Nothing: the new conversation starts fresh |
| **T** transcripts | The three earlier conversations, word for word |
| **S** summaries (Σ in the article) | One summary per earlier conversation, written by the model after it |
| **K** structured notes | One JSON object per earlier conversation (preferences, commitments, open items, facts), extracted by the model |

These are four designs in this experiment, not a catalogue of memory architectures. In every design, the memory block is placed in the same position (after the customer-facing prompt), the tools are the same, and the account records are the current source of truth.

## What was frozen first

Written and hashed (`data/FROZEN.json`, 2026-11-15 14:02:46 UTC) before any system code existed; `run.py` will refuse to run if any of these change:

- `data/conversations.json`: the three earlier conversations (2026-10-30, 2026-11-06, 2026-11-10), written by hand as test material. They establish a duplicate-charge case (CASE-311) with its refund **pending**, an **email-only** contact preference, the customer's **intention** to remove an extra seat next month, and a request to be reminded of INV-1002's refund deadline (2026-11-27).
- `data/account.json`: `02-agentic-workflows/data/account.json` with one added line, payment `PAY-5008`: the refund of the duplicate charge, succeeded on 2026-11-05. So what the 2026-10-30 conversation said ("pending") is stale by the new conversation on 2026-11-12. The data used by Articles #24 and #25 is unchanged.
- `data/help_center.json`, `data/prompt.json`: byte-identical copies of `02-agentic-workflows/data`.
- `data/questions.json`: ten questions, each with its expected answer, the memory facts it needs, and the acceptable evidence from current records (with alternatives where more than one source can support the answer).
- `data/memory.json`: the four designs, the memory-block formats, and the prompts and JSON schema for creating summaries and notes.
- `data/evaluation.json`: how every conversation is evaluated (below).

## The ten questions

| ID | Question | Needs |
|---|---|---|
| A1 | How did I ask you to contact me? | memory |
| A2 | What's the case number for my double-charge report? | memory |
| B1 | Has the refund for my double charge come through yet? | current records; memory is stale |
| B2 | Did I already remove the extra seat I mentioned? | current records; memory holds an intention, not an action |
| C1 | Last time you told me about a deadline. What was it, and is it still open? | memory and current records |
| C2 | How many extra seats do I have now, and what do they cost per month? | current records |
| D1 | How much was my most recent invoice? | current records |
| D2 | Until what date can I ask for a refund on INV-1003? | current records |
| E1 | Did I give you my VAT number in one of our earlier conversations? | the complete history (the answer is no) |
| E2 | What did we agree about my September invoice last time? | the complete history (nothing was) |

## How it's evaluated (from `data/evaluation.json`)

Three dimensions, recorded separately for every conversation:

- **Answer label**, by hand: `correct`, `incorrect`, or `no_answer`. Claims about earlier conversations are judged against `data/conversations.json`. A separate **invented-memory** flag marks replies that claim something was said or agreed earlier when it wasn't.
- **Evidence availability:** whether each needed memory fact reached the model (N: absent by construction; T: present by construction; S and K: hand-labeled per seed as present, misstated, or absent), whether the current records were looked up (the needed text in an executed tool result), and, for E1 and E2, whether the design carried the complete history.
- **Outcome**, exactly one: **substantive success** (correct, evidence available), **appropriate abstention** (no answer when the needed memory wasn't available, as for N on A1, A2, C1, E1, E2), **correct but unsupported**, or **failure**.

When memory and a current record disagree, the record wins. B1 succeeds on current records alone; four sub-scores are recorded in addition (current record retrieved; current status given; "pending" not repeated as current; the change from pending noted, where the old status was available). For B2, a reply that reports the remembered intention as a completed removal is flagged.

Cost counts every call: the memory-creation calls for S and K per seed, the input tokens each conversation spends on carried memory versus the rest, output tokens, model calls, latency, and experiment totals including memory creation.

## Setup (planned)

- Ollama with `qwen2.5:3b` only, for the same reasons as `02-agentic-workflows` and `03-multi-agent`.
- 4 designs × 10 questions × 3 seeds = **120 conversations**, plus 6 memory-creation calls per seed (3 for S, 3 for K). Seed = run number (1–3) for every model call; at most 5 model calls per conversation, 1,000 tokens per call, 400 per memory-creation call.
- Fixed run order; model unloaded before and after and loaded with one throwaway request first; the run waits until no other process has used Ollama for 60 s, and records for each conversation whether another model was loaded.

## Limitations

- An educational experiment, not a benchmark: one small local model, ten questions, three seeds, one synthetic customer with three short earlier conversations.
- One design of each memory strategy, written before the run. Other prompts, longer histories, retrieval over a memory store, or memory the agent writes itself during a conversation are not tested.
- Answers and memory facts are labeled by hand, by one person, against the written rules in `data/evaluation.json`.
