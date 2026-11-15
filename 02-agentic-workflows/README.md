# 02 · Agentic workflows: who should control the flow?

Companion demo for **Article #24, "Agentic Workflows: When Should an LLM Control the Flow?"** on [snehalagrawal.com](https://snehalagrawal.com/articles/) (not published yet; this README will link to it when it is).

The same support assistant answers the same twelve questions three ways. The model, the customer-facing system prompt, the data, and the five tools are identical. W and H add a router call (its own classification instructions and a JSON schema) before the assistant runs; A does not. The control flow is the variable being compared: who decides which tools run, and in what order.

It follows [`01-tool-design`](../01-tool-design/) (Article #23): the same Ollama setup, and the same agent loop for A and for H's fallback; the router and the fixed workflows are new here. The tool descriptions apply what `01-tool-design` found.

## The three systems

| System | Who decides the steps |
|---|---|
| **W** workflow | A **router** (one model call, structured JSON output) picks a known intent or `other`. For a known intent, **code** runs that intent's fixed tool calls, and the model writes the reply from their results. `other` is **handed to a person** (a fixed message; no human answer is simulated). |
| **A** agent | The **model** gets all five tools and chooses what to call and in what order, for up to 5 model calls (the loop from Articles #8 and #23). |
| **H** hybrid | The same router. Known intents take W's fixed path; `other` goes to A's agent loop, with the same cap. |

| Known intent | Fixed steps (W and H) |
|---|---|
| `refund_status` | `get_refund_status()` |
| `invoice_lookup` | `get_invoices()` |
| `help_center` | `search_help_center(<the customer's question>)` |

All three systems call the same tool functions (`python/tools.py`): `search_help_center`, `get_invoices`, `get_refund_status`, `get_payments`, `get_plan_history`. All only read. The workflow's results reach the model in the same tool-result format the agent sees, so the phrasing step differs from the agent's last step only in who chose the calls.

## What was frozen first

Written and hashed before any system code existed (2026-11-15 10:47:37 UTC); `run.py` refuses to run if any of these change:

- `data/account.json`: synthetic. The account from `01-tool-design`, plus card payments (October's invoice had a declined attempt, then was charged twice on the retry day) and a plan history (two extra seats added on 2026-11-05).
- `data/help_center.json`: the eight articles from `01-tool-design`, plus three on duplicate charges, failed payments, and seats.
- `data/prompt.json`: the customer-facing system prompt from Articles #8 and #23, unchanged.
- `data/questions.json`: twelve questions, each with the intent a correct router picks, the expected answer, and the facts it depends on, plus the grading rules.

Eight questions are **known-path** (refund status, a deadline, the latest invoice, a policy, a price, how to cancel). Four are **open-ended** (expected intent `other`): "Something looks off with my October bill. What happened?", "Why did I pay $15 on November 5? I didn't buy anything.", "Have you ever refunded me anything?", and "My card was declined in October. Do I still owe you anything?"

### Grading rules (from `questions.json`)

- **Routing** (W, H): the router's intent against the frozen intent. A wrong route is a routing error even if the reply after it happens to be acceptable.
- **Handoff** (W): its own outcome; never labeled, never a success.
- **Facts retrieved**: every fact the question depends on appears in a tool result that executed.
- **Answer label**, by hand, for every reply that isn't a handoff: `correct`, `incorrect`, or `no answer` (the interpretation approved for `01-tool-design`), stored in `python/labels.json` with the SHA-256 of the exact reply.
- **Success**: labeled correct **and** facts retrieved.
- **Cost**: model calls and input tokens include the router, agent, and phrasing calls. Latency is wall-clock time for the automated path only.

## Run it

Prerequisites: Python 3.10+ and [Ollama](https://ollama.com/download) with `qwen2.5:3b` (about 2 GB). `llama3.2:3b` isn't used: with the Ollama template recorded in `01-tool-design`, it gets no tools after the first tool result, which would make its agent results a test of the template, not of the control flow.

```bash
ollama pull qwen2.5:3b
cd 02-agentic-workflows/python
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run.py                                   # 3 systems x 12 questions x 3 seeds = 108 conversations
.venv/bin/python report.py results-qwen2.5_3b.jsonl
```

`run.py` unloads the model, loads it with one throwaway request (so no conversation includes load time), runs the conversations in a fixed order (system, question, seed), and unloads it. It records, for every conversation, whether any other model was loaded in Ollama, because another process using the machine changes the timings. It also writes `results-<model>-setup.json` and the model's chat template.

| File | What it does |
|---|---|
| `python/tools.py` | The five tools, shared by all systems |
| `python/systems.py` | The router, the fixed workflows, the agent loop, and the three systems |
| `python/llm.py` | Ollama `/api/chat` over plain HTTP, with tools or a JSON schema |
| `python/run.py` | Checks the frozen hashes, runs every conversation, saves them |
| `python/report.py` | Grades saved conversations and prints the comparison; calls no model |
| `python/show.py` | Prints conversations compactly, for hand labeling |

## Results

Setup: Ollama 0.40.16 on an Apple M4 Pro; `qwen2.5:3b` (`357c53fb659c`), Q4_K_M; default sampling, seed = run number (1–3), at most 5 model calls and 1,000 tokens per call in the agent loop; 2026-11-15, run started 11:41:01 UTC. The 108 conversations took 183 s. No other model was loaded in Ollama during any of them, so the timings are comparable.

`python report.py results-qwen2.5_3b.jsonl` prints:

```text
questions  system   n  route errors  handoff  correct  incorrect  no answer  facts  success  unsupp.  step limit  model calls  input tok  median s  max s
known (8)  W       24             3        0       18          6          0     21       18        0           0         2.00      622.0      0.92   2.98
known (8)  A       24             0        0       12         10          2     15       10        2           0         1.92      957.0      1.37   4.05
known (8)  H       24             3        0       17          7          0     21       17        0           0         2.00      622.0      0.79   2.30
open (4)   W       12            12        0        3          9          0      9        3        0           0         2.00      539.0      1.39   5.71
open (4)   A       12             0        0        2          7          3      6        2        0           0         2.08     1250.0      2.35   9.11
open (4)   H       12            12        0        3          9          0      9        3        0           0         2.00      539.0      1.33   5.62
all (12)   W       36            15        0       21         15          0     30       21        0           0         2.00      619.0      1.15   5.71
all (12)   A       36             0        0       14         17          5     21       12        2           0         1.97     1057.5      1.88   9.11
all (12)   H       36            15        0       20         16          0     30       20        0           0         2.00      619.0      1.04   5.62
```

Columns: `correct`, `incorrect`, and `no answer` are the hand labels (with `handoff`, they add up to n). `facts`: every required fact appeared in an executed tool result. `success`: correct **and** facts. `unsupp.`: correct without the facts (`success` + `unsupp.` = `correct`). `model calls`: mean per conversation. `input tok`: median per conversation, all model calls. `median s`, `max s`: wall-clock seconds for the automated path.

With three seeds per question, a difference of one or two conversations should be treated as noise rather than evidence of a real improvement.

What the recorded conversations show (every reply was read and labeled by hand):

- **The router never chose `other`.** In both W and H, all 12 open-ended runs (4 questions × 3 seeds) were routed to a known intent (O1 and O2 to `invoice_lookup`, O3 and O4 to `refund_status`), and so was every run of K6, a policy question (to `refund_status`): each routed system's 15 routing errors. So W never handed off, and H's agent path was never exercised: H took exactly W's routes in all 36 conversations. Its results are effectively W's; the one-conversation difference comes from replies that differed despite identical input and seed (33 of 36 W/H replies were identical).
- **On the known questions, the fixed path did better than the agent.** Successes: W 18 of 24, A 10 of 24. The agent picked the plan history for "How do I cancel?" and "How much does the Pro plan cost?", called no tool and asked the customer for an invoice id (twice), and announced a next step without taking it. It also used more input tokens (median 957 against 622) and took longer (median 1.37 s against 0.92 s; slowest 4.05 s against 2.98 s).
- **Lucky routes are still wrong routes.** W and H's three open-ended successes are all O2 ("Why did I pay $15 on November 5?"): the wrong route, `invoice_lookup`, happened to return an invoice described as "Extra seats add-on". On O1, the same route returned the invoice list, which can't show a duplicate charge, and every W and H reply said the October bill looked fine.
- **Tool access didn't make the agent investigate.** On O1 (the duplicate charge), the agent never called `get_payments`. On O4 ("Do I still owe you anything?"), it called `get_payments` in all three runs, so the payment record showing October was paid was in front of it, and every reply still stated an amount owed or a false cause ($25, $75). Its 7 `get_payments` calls were on O2 (3), O3 (1), and O4 (3). No system answered O1 or O4 correctly.
- **Routing was cheap.** The router call's median was 129 input tokens and 0.09 s.

Two agent replies to K7 gave the right Pro plan price from the plan history ("Started the Pro plan ($25 per month)"), not from the help center. The frozen facts for K7 accept only the help-center text, so they count as correct without facts (`unsupp.`), not as successes; the frozen facts were not changed.

The check run (`python/results-check.jsonl`, 6 conversations) was made before the full run, while another process was using Ollama; it is kept as a record and is not part of these numbers.

## Limitations

- An educational experiment, not a benchmark: one small local model, twelve questions, three seeds, one synthetic account.
- One router design, written before the run. The finding is that this router never chose `other`; a different prompt or model might. That would be a separate experiment, not a correction of this one.
- The facts check shows what reached the model, not what it understood. Answers are graded by hand, by one person, against written rules.
- Token counts are Ollama's `prompt_eval_count` for this model's tokenizer; latencies are for this machine with nothing else running.
