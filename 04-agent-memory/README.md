# 04 · Agent memory: what should an agent carry between conversations?

Companion demo for **Article #26, ["Memory in AI Agents: What Does an Agent Actually Need to Remember?"](https://snehalagrawal.com/articles/memory-in-ai-agents-what-does-an-agent-actually-need-to-remember/)** on snehalagrawal.com.

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

## Run it

Prerequisites: Python 3.10+ and [Ollama](https://ollama.com/download) with `qwen2.5:3b` (about 2 GB).

```bash
ollama pull qwen2.5:3b
cd 04-agent-memory/python
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run.py                                   # memory for 4 designs x 3 seeds, then 120 conversations
.venv/bin/python report.py results-qwen2.5_3b.jsonl
```

`run.py` checks the frozen hashes, unloads the model, loads it with one throwaway request, creates every design's memory for every seed (saved to `results-<model>-memory.json`), then runs the conversations in a fixed order (design, question, seed) and unloads the model. Seed = run number (1–3) for every model call; at most 5 model calls per conversation, 1,000 tokens per call, 400 per memory-creation call. It records, for every conversation and memory creation, whether any other model was loaded in Ollama, and writes `results-<model>-setup.json` (runtime, model digest, frozen and code hashes) and the model's chat template.

| File | What it does |
|---|---|
| `python/systems.py` | The four designs, built from `data/memory.json`: transcript rendering, summary and notes creation, prompt placement, and the agent loop |
| `python/tools.py`, `python/llm.py` | Byte-identical copies from `02-agentic-workflows`: the five data tools and the Ollama client |
| `python/run.py` | Checks the frozen hashes, creates the memory, runs every conversation, saves everything |
| `python/report.py` | Grades saved conversations with the rules in `data/evaluation.json`; calls no model. `--todo` lists replies and memory blocks that still need a hand label |
| `python/labels.json` | Hand labels for every reply: answer label, invented-memory flag, note, and the B1 and B2 sub-scores |
| `python/memory_labels.json` | Hand labels for every S and K memory block, per memory fact |

## Results

Setup: Ollama 0.40.16 on an Apple M4 Pro; `qwen2.5:3b` (`357c53fb659c`), Q4_K_M; default sampling; 2026-11-15, run started 15:16:11 UTC. Memory creation and the 120 conversations took 209 s. No other model was loaded in Ollama during any conversation or memory creation.

`python report.py results-qwen2.5_3b.jsonl` prints (abridged):

```text
questions          design   n  correct  incorrect  no answer  invented  success  abstain  unsupp.  failure
memory (A1, A2)    N        6        0          1          5         0        0        5        0        1
memory (A1, A2)    T        6        2          4          0         1        2        0        0        4
memory (A1, A2)    S        6        2          3          1         1        2        0        0        4
memory (A1, A2)    K        6        2          3          1         1        2        0        0        4
memory vs records  N        6        2          2          2         0        2        0        0        4
memory vs records  T        6        3          2          1         0        3        0        0        3
memory vs records  S        6        1          5          0         1        1        0        0        5
memory vs records  K        6        1          4          1         0        0        0        1        5
memory + records   N        3        0          2          1         0        0        1        0        2
memory + records   T        3        2          1          0         0        1        0        1        1
memory + records   S        3        0          3          0         0        0        0        0        3
memory + records   K        3        0          3          0         0        0        0        0        3
records only       N        9        6          3          0         0        6        0        0        3
records only       T        9        7          1          1         0        7        0        0        2
records only       S        9        7          2          0         0        7        0        0        2
records only       K        9        6          3          0         0        6        0        0        3
never said         N        6        0          3          3         2        0        3        0        3
never said         T        6        1          1          4         0        1        0        0        5
never said         S        6        1          2          3         2        1        0        0        5
never said         K        6        3          1          2         1        3        0        0        3
all                N       30        8         11         11         2        8        9        0       13
all                T       30       15          9          6         1       14        0        1       15
all                S       30       11         15          4         4       11        0        0       19
all                K       30       12         14          4         2       11        0        1       18

B1, conversations per design:   retrieved PAY-5008   current status   no stale claim   change noted
N                                          3                3                3              n/a
T                                          2                2                2           0 of 3
S                                          0                2                1           0 of 3
K                                          0                2                3           0 of 3

design  creation calls  creation in/out tok  memory tok  calls  input tok  of which memory  output tok  median s  max s  total in tok
N                    0                0 / 0           0   1.63        940                0          79      1.04   3.50        24,580
T                    0                0 / 0         385   1.83      1,721              770         108      1.38   5.22        49,909
S                    9           1,791 / 409        182   1.87      1,372              364          86      1.37   3.85        40,489
K                    9          2,187 / 1,038       436   1.77      1,824              872          88      1.26   4.55        52,003
```

Columns: `correct`, `incorrect`, `no answer` are hand labels. `invented`: replies that claim something was said or agreed earlier when it wasn't. `success`: correct and the evidence available (needed memory facts in the carried memory, needed records looked up). `abstain`: no answer when the needed memory wasn't available. `unsupp.`: correct without the evidence. `memory tok`: median memory-block tokens per model call. `input tok`, `of which memory`, `output tok`: median per conversation. `total in tok`: every conversation plus memory creation, whole experiment.

With three seeds per question, a difference of one or two conversations is noise, not evidence.

What the recorded conversations show (every reply and every S and K memory block was read and labeled by hand):

- **Memory that was carried wasn't reliably used.** T carried the verbatim transcripts, so CASE-311 was in front of it every time, and it answered A2 correctly in none of three runs: twice it looked in the records, which have no cases, and said there was no case number; the third gave CASE-311 along with false claims about the refund. Across A1 and A2, each memory design succeeded 2 of 6 times. N abstained on 5 of 6, as it should.
- **Stale memory pulled the lookup in the wrong direction.** On B1, all three N runs called `get_payments` and found the refund (PAY-5008); S and K, whose memory said "refund pending", never retrieved it in six runs (they read the invoices or the refund status instead). Successes: N 2, T 2, S 0, K 0 of 3. No design that carried the old status said it had changed (0 of 9), and three replies repeated "pending" as current.
- **Memory designs claimed to have no memory.** On "Did I give you my VAT number?", 7 of the 9 replies from designs that carried the history were no answer, and 6 of those said they had no access to earlier conversations, which they had. Only S and K, once each, answered no.
- **Invented memories appeared with and without memory.** On "What did we agree about my September invoice?", N invented an earlier discussion in two of three runs ("last time we discussed your September invoice, we decided to add 2 extra seats"), as did S twice and K once; T didn't. One S reply to B2 reported the customer's intention as an action ("since you mentioned you removed one seat").
- **Summaries and notes kept most facts, not all.** Of 30 memory facts (5 per block, 6 blocks), 28 were kept correctly: K seed 2 dropped the case number, and S seed 2 turned the reminder request into "Reminder sent." K seed 1 also recorded an INV-1002 date that appears nowhere (2026-11-01).
- **Memory didn't help the records-only questions, and cost more.** On C2, D1, and D2, every design succeeded 6 or 7 times of 9 (N and K 6, T and S 7). Carrying memory raised the median input per conversation from 940 tokens (N) to 1,372 (S), 1,721 (T), and 1,824 (K). K's notes (436 tokens per call) were longer than the full transcripts (385); with creation included, K used the most input tokens overall (52,003 against N's 24,580).
- **Overall:** substantive successes N 8, T 14, S 11, K 11 of 30; N's 9 appropriate abstentions are reported separately and aren't successes. T's lead describes this run, not a general ranking.
- **Negative evidence (E1, E2).** The frozen rules count a design as having the complete history when its memory was created from all three conversations (an operational definition; it doesn't mean the summaries or notes preserve everything said), so S's and K's "no" answers count as successes. A stricter reading would accept only the verbatim transcripts as proof that something was never said; under it, those 4 answers (S 1, K 3) would be correct but unsupported, giving S 10 and K 8 successes. The frozen scoring is reported above; this reading is noted, not substituted.

One label was revised after review: N's seed-3 reply to C2 was first labeled correct and is now incorrect (it prices a "seat included in the Pro plan" that the data doesn't have); the revision is recorded in its note. `python/report-qwen2.5_3b.txt` is the full report output for these labels.

The check run (`python/results-check.jsonl`, 16 conversations, seed 1) was made before the full run with the same code; it is kept as a record and is not part of these numbers. Its seed-1 memory blocks are identical to the full run's.

## Limitations

- An educational experiment, not a benchmark: one small local model, ten questions, three seeds, one synthetic customer with three short earlier conversations.
- One design of each memory strategy, written before the run. Other prompts, longer histories, retrieval over a memory store, or memory the agent writes itself during a conversation are not tested.
- Answers and memory facts are labeled by hand, by one person, against the written rules in `data/evaluation.json`. One label was revised after review (see Results).
- Token counts are Ollama's `prompt_eval_count` and `eval_count` for this model's tokenizer; memory-block tokens are measured as the difference from design N's first call; latencies are for this machine with nothing else running.
