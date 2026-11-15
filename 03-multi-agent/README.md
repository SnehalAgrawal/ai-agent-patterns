# 03 · Multi-agent: what does a second agent add?

Companion demo for **Article #25, "Single-Agent vs Multi-Agent Systems"** on [snehalagrawal.com](https://snehalagrawal.com/articles/) (not published yet; this README will link to it when it is).

The same support assistant from [`02-agentic-workflows`](../02-agentic-workflows/) (Article #24) answers the same twelve questions three ways: as one agent, as a manager that delegates to two specialist agents, and as one agent whose draft is checked by a reviewer agent. The question isn't whether more agents are better. It's what capability each extra agent adds, and whether that is worth another context, another decision-maker, another handoff, and more model calls.

## The three systems

| System | Shape | What the extra agent is for |
|---|---|---|
| **S** single agent | Article #24's agent A, unchanged: one agent loop with the five data tools | (baseline) |
| **M** manager + specialists | A **manager** agent whose only tools are two agents: a **billing specialist** (invoices, refund status, payments, plan history) and a **help-center specialist** (help-center search). Each specialist runs its own tool loop on the manager's request and returns its reply as the tool result. | Specialized contexts: each agent has fewer tools and a narrower job |
| **R** agent + reviewer | S writes a draft. A **reviewer** agent with all five data tools checks it against the records, then replies `APPROVE` (the draft is sent) or with a replacement reply. One pass; no return to the drafter. | Independent verification |

Design choices, frozen in `data/agents.json`:

- **The manager uses the customer-facing prompt from Article #24, unchanged.** That prompt says only "Use the available tools to help the customer", so at the customer-facing level S and M differ only in their tools (five data tools, or two specialist agents). Behind those tools, M adds the specialists' own prompts, separate contexts, and handoffs.
- **Specialists see only the manager's request,** not the customer's message or the manager's conversation. The manager writes the task. This is a deliberate context boundary in this implementation, not a requirement of the "agents as tools" (manager) pattern: in OpenAI's Agents SDK, a tool-agent gets a single input string by default and doesn't inherit the parent run's conversation automatically, but the conversation can be shared. Sharing the full context with every agent is a different design, discussed in the article but not tested here.
- **The reviewer has the tools,** so it can check evidence rather than judge plausibility.
- **Fallbacks are fixed:** an invalid call to a specialist gets the same error text as an invalid data-tool call; a specialist that ends without a reply returns a fixed "couldn't finish" text; `APPROVE` counts only as the whole reply after a fixed normalization (not a reply that merely contains the word); a reviewer that ends without any reply leaves the draft in place, recorded as `no_verdict`.

## What was frozen first

Written and hashed (`data/FROZEN.json`, 2026-11-15 12:59:49 UTC) before any system code existed; `run.py` will refuse to run if any of these change:

- `data/account.json`, `data/help_center.json`, `data/prompt.json`, `data/questions.json`: byte-identical copies of `02-agentic-workflows/data` (same SHA-256 hashes). No questions were added or changed, so nothing in the test material was tuned to this experiment.
- `data/agents.json`: the three systems, the manager's two tools, every agent's prompt, the reviewer's message format, the verdict rule, and the fallbacks.
- `data/evaluation.json`: how every conversation is evaluated (below).

## How it's evaluated (from `data/evaluation.json`)

- **Answer label**, by hand, unchanged from Article #24: `correct`, `incorrect`, or `no_answer` for the reply the customer gets, stored with the SHA-256 of the exact text. In R, the draft is labeled too.
- **Facts at four levels.** The facts are each question's frozen evidence list.
  - **F1 retrieved:** some agent's executed tool results contain every fact.
  - **F2 passed on** (M): each specialist reply is hand-labeled per fact as present, misstated, or absent.
  - **F3 available to the writer:** the agent whose text the customer gets had every fact as evidence in its own context. A draft is a claim, not evidence.
  - **F4 used correctly:** the answer label.
- **Success** is correct **and** F3. Correct **and** F1 (Article #24's definition) is reported too; the difference is the *retrieved-to-writer availability gap*: facts that were retrieved somewhere in the system but weren't available to the agent that wrote the reply. It shows where information was, not what a handoff costs.
- **Reviewer outcomes** (R), mutually exclusive: approved correct, rewrote still correct, damaged, fixed, approved wrong, rewrote still wrong, no verdict. Per review: whether the reviewer checked anything, whether the evidence was in front of it, and approvals made without any check.
- **Where M failed:** one stage per M conversation that isn't correct, the first that fails: delegation → retrieval → handoff → synthesis, with each boundary defined in `evaluation.json`.
- **Cost, with nothing excluded:** every model call by every agent, input and output tokens, and end-to-end wall-clock time for the whole sequence of agents, also broken down by role.

## Run it

Prerequisites: Python 3.10+ and [Ollama](https://ollama.com/download) with `qwen2.5:3b` (about 2 GB). `llama3.2:3b` isn't used: with the Ollama template recorded in `01-tool-design`, it gets no tools after the first tool result.

```bash
ollama pull qwen2.5:3b
cd 03-multi-agent/python
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run.py                                   # 3 systems x 12 questions x 3 seeds = 108 conversations
.venv/bin/python report.py results-qwen2.5_3b.jsonl
```

`run.py` checks the frozen hashes, unloads the model, loads it with one throwaway request (so no conversation includes load time), runs the conversations in a fixed order (system, question, seed), and unloads it. Seed = run number (1–3) for every model call in a conversation; at most 5 model calls per agent loop and 1,000 tokens per call. It records, for every conversation, whether any other model was loaded in Ollama, and writes `results-<model>-setup.json` (runtime, model digest, frozen and code hashes) and the model's chat template.

| File | What it does |
|---|---|
| `python/systems.py` | The three systems, built from `data/agents.json`: the shared agent loop, the manager's specialist tools, the reviewer and its verdict rule |
| `python/tools.py`, `python/llm.py` | Byte-identical copies from `02-agentic-workflows`: the five data tools and the Ollama client |
| `python/run.py` | Checks the frozen hashes, runs every conversation, saves them |
| `python/report.py` | Grades saved conversations with the rules in `data/evaluation.json`; calls no model. `--todo` lists texts that still need a hand label |
| `python/labels.json` | Hand labels for every reply the customer got and every R draft, keyed by question and the SHA-256 of the exact text |
| `python/handoff_labels.json` | Hand labels for every specialist reply (per fact) and every manager request (per fact) |

## Results

Setup: Ollama 0.40.16 on an Apple M4 Pro; `qwen2.5:3b` (`357c53fb659c`), Q4_K_M; default sampling; 2026-11-15, run started 13:08:54 UTC. The 108 conversations took 652 s. No other model was loaded in Ollama during any of them.

`python report.py results-qwen2.5_3b.jsonl` prints (abridged):

```text
questions  system     n  correct  incorrect  no answer   F1   F3  success  succ. F1  unsupp.  step limit
known (8)  S         24       12         10          2   15   15       10        10        2           0
known (8)  M         24        8         10          6   13    9        4         8        4           0
known (8)  R         24       13         10          1   18   17       12        12        1           0
open (4)   S         12        2          7          3    6    6        2         2        0           0
open (4)   M         12        0          4          8    1    1        0         0        0           4
open (4)   R         12        1         11          0    6    3        0         1        1           0
all (12)   S         36       14         17          5   21   21       12        12        2           0
all (12)   M         36        8         14         14   14   10        4         8        4           4
all (12)   R         36       14         21          1   24   20       12        13        2           0

questions  system   model calls  input tok  output tok  median s   max s
all (12)   S               1.97     1057.5       134.5      1.92    9.20
all (12)   M               7.28     2206.0       319.5      5.69   43.53   (manager 2.67, billing 4.22, help center 0.39)
all (12)   R               4.42     2926.5       298.0      5.50   15.71   (drafter 1.97, reviewer 2.44)
```

Columns: `correct`, `incorrect`, `no answer` are the hand labels of the reply the customer got. `F1`: some agent's executed data-tool results contain every fact. `F3`: the agent that wrote the reply had every fact as evidence. `success`: correct **and** F3. `succ. F1`: correct **and** F1 (Article #24's definition). `unsupp.`: correct without F3. `step limit`: the manager used its 5 model calls without replying. `model calls`: mean per conversation, every agent. `input tok`, `output tok`: median per conversation, every call. `median s`, `max s`: end-to-end wall-clock seconds.

With three seeds per question, a difference of one or two conversations is noise, not evidence.

What the recorded conversations show (every text was read and labeled by hand; S's 36 replies are byte-identical to Article #24's agent A replies, so their labels were imported from `02-agentic-workflows` and recorded as such):

- **S is #24's agent A, reproduced.** All 36 replies were identical to the saved #24 replies, so the baseline is the same system on the same data, not a re-tuned one.
- **M succeeded on 4 of 36, S on 12.** The billing specialist made 130 data-tool calls and 98 were invalid: it added arguments the tools don't have (`customer_email` 56 times, `plan_name`, `customer`, `email`, `month`, `date_from`, ...) and retried. S, with the same five tools, made 36 calls and none was invalid. The specialist prompt names the customer's email, and 32 of the manager's 64 requests repeated it; the cause of the invented arguments wasn't tested. M's first failing stage, for its 28 conversations that weren't correct: retrieval 19, synthesis 6 (the manager was told every fact and still answered wrongly), delegation 3 (K8, "How do I cancel?", was sent to the billing specialist every time), handoff 0.
- **The narrow help-center specialist helped.** The manager sent it every run of K6 (the refund window) and K7 (the Pro plan price), and M had 4 successes of 6 on those two questions, against S's 0 of 6: with only the help-center search, the specialist couldn't answer from the invoices or the plan history, as S did.
- **The retrieved-to-writer availability gap was 4 conversations, all correct answers.** On K2 (one run) and K3 (all three), the specialist retrieved the invoice's date but passed on only the refund deadline; the manager answered correctly without the date. These count as correct without the facts, not as successes.
- **R's reviewer fixed as many answers as it broke.** Outcomes: approved correct 5, rewrote still correct 6, damaged 2, fixed 2, approved wrong 2, rewrote still wrong 18, no verdict 1 (a correct draft, sent unchanged). R's final replies were correct 14 times, exactly as often as its drafts. The reviewer made at least one tool call that executed in all 36 reviews and had every fact in its own results in 20; it never approved a draft without a tool call, and it rewrote 28 of 36 drafts.
- **A rewrite was often not a reply.** 11 of the 28 rewrites addressed the drafter rather than the customer ("The draft reply is incorrect...", "Please correct the draft and resend it"), and the frozen rule sends a rewrite as written. Three rewrites repeated the draft word for word. One reply was `_APPROVE`: under the frozen rule that is a rewrite, so the customer got "_APPROVE"; others began with "ADOPT", "ARDUE", "APOORV," or "Approve" followed by a reply.
- **Cost.** M made 3.7 times as many model calls as S (7.28 against 1.97 per conversation) and used about twice the input tokens (median 2,206 against 1,058) and 2.4 times the output tokens (median 320 against 135); its slowest conversation took 43.5 s. Its specialists made more model calls than the manager (4.61 against 2.67 per conversation), mostly the billing specialist retrying invalid calls. R made 2.2 times as many calls and used 2.8 times the input tokens (median 2,927) and 2.2 times the output tokens (median 298).
- **R's drafts were S's replies in 33 of 36 conversations.** In three (K4, O1, O4, seed 1), the same prompt, tools, and seed produced a different draft; their labels come from reading those drafts.

The check run (`python/results-check.jsonl`, 12 conversations, seed 1) was made before the full run, with the same code except that the setup record didn't yet include the code hashes; it is kept as a record and is not part of these numbers.

## Limitations

- An educational experiment, not a benchmark: one small local model, twelve questions, three seeds, one synthetic account.
- One design of each pattern, written before the run. Other prompts, other models, or full-context delegation might behave differently; that would be a separate experiment.
- Published multi-agent results (for example Anthropic's research system) come from frontier models on broad research tasks. This experiment doesn't test that setting and isn't meant to contradict it.
- Answers, handoff messages, and failure stages are labeled by hand, by one person, against the written rules in `data/evaluation.json`.
- The facts are each question's evidence list, frozen for Article #24. On K2 and K3 the manager answered correctly with only the refund deadline, which the frozen list doesn't accept on its own; those count as correct without the facts.
- Token counts are Ollama's `prompt_eval_count` and `eval_count` for this model's tokenizer; latencies are for this machine with nothing else running.
