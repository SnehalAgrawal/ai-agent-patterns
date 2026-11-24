# 05 · Planning: who decides the steps, and when?

Companion demo for **Article #27, ["Planning, Reasoning, and Execution in AI Agents: Who Decides the Steps, and When?"](https://snehalagrawal.com/articles/planning-reasoning-and-execution-in-ai-agents/)** on snehalagrawal.com.

The support assistant from [`02-agentic-workflows`](../02-agentic-workflows/), [`03-multi-agent`](../03-multi-agent/) and [`04-agent-memory`](../04-agent-memory/) gets requests that take more than one step: find an invoice, then check it; check for a duplicate charge, then for its refund; work out a date when the tool that knows it is down. Four designs answer them with the same model, prompt, tools, and data. They differ only in **who decides the steps, and when**.

## The four designs

| Design | Who decides the steps | When |
|---|---|---|
| **W** workflow | A developer, in code: one fixed plan per task | Before the request arrives |
| **L** loop (ReAct-style) | The model, one response at a time, with the tools | After every result |
| **P** plan, then execute | The model, in one planning call (JSON); code runs the steps. Deliberately simple: a step can't refer to an earlier step's result (no ReWOO-style `#E1` references or LLMCompiler-style dependency graph) | Once, before any tool runs |
| **R** plan, execute, replan | The model plans; code runs the steps; a replanning call sees the results and is done or plans more | Before, and after every batch of results |

W, P and R share the **answer step**: one model call without tools that sees the customer's message and every attempted step with its result, in the same format the loop sees them. So between W, P and R, the only difference is how the steps were chosen.

W knows which task it is answering (routing is assumed correct), so it is a **ceiling** for a fixed workflow, not a fair competitor: a real one needs a router, which is where [`02-agentic-workflows`](../02-agentic-workflows/) found its problems. Its value here is diagnostic: with the right steps given, W's failures are failures of the answer step.

Every design is held to **8 model calls** per conversation. L ends without a reply if its 8th call still requests tools (`step_limit`). R always reserves one call for the answer, so it replans at most 6 times; if the 6th replan still plans steps, they don't run (`replan_limit`). A plan runs at most 6 steps.

## What was frozen first

Written and hashed (`data/FROZEN.json`, 2026-11-24 06:06:39 UTC) before any system code existed; `run.py` refuses to run if any of these change:

- `data/account.json`, `data/help_center.json`, `data/prompt.json`: byte-identical copies of `04-agent-memory/data` (same SHA-256).
- `data/tasks.json`: nine requests, each with what it tests, the expected answer, what a correct reply must contain, the acceptable evidence from the records, the tools whose results are relevant, and any injected failure. The expected values were computed with `python/tools.py`, not by hand.
- `data/designs.json`: the four designs, every prompt and JSON schema, W's plans, and the limits.
- `data/evaluation.json`: how every conversation is evaluated (below).

`python/tools.py` and `python/llm.py` were copied byte-identical from `04-agent-memory/python` before the freeze. One infrastructure check came before the freeze: a planning-shaped JSON schema with a free-form `arguments` object, sent to `qwen2.5:3b` with an unrelated weather prompt, to confirm Ollama's structured output accepted it. No task was sent to a model before the freeze.

## The nine tasks

| ID | Request | Tests |
|---|---|---|
| Q1 | How do I cancel my subscription? | one step: planning adds only overhead |
| Q2 | Which of my invoices can I still get a refund for, until when, and how would the money come back to me? | independent lookups |
| Q3 | Can I still get a refund on my most recent Pro plan invoice? If so, until when? | a dependent step: which invoice is meant has to be found first |
| Q4 | Was I charged twice for my October invoice? If so, has the extra charge been refunded, and when? | a conditional chain |
| Q5 | What was I charged in November, and why that amount? | synthesis from three records |
| Q6 | How much have I paid you in total so far, after refunds? | reasoning over correct results (arithmetic) |
| Q7 | Until what date can I ask for a refund on INV-1002? | an execution failure: `get_refund_status` is down for the whole conversation; the date can still be worked out from the invoice date and the 30-day policy |
| Q8 | Please remove the two extra seats from my plan and refund the seats invoice. | a request no tool can carry out (every tool only reads) |
| Q9 | I was charged twice for October. How do I get the extra charge refunded? | new information: the records show it was already refunded |

## How it's evaluated (from `data/evaluation.json`)

- **Answer label**, by hand: `correct`, `incorrect`, or `no_answer`, against each task's `correct_requires`. A separate **fabricated-action** flag marks replies that claim the assistant did something (a refund, a seat change, a ticket), since no tool changes anything.
- **Evidence**, automatic: the needed values appear in the result of a tool call that executed (error results, including the injected failure, don't count). For P and R, also whether the **first plan alone** got the evidence.
- **Outcome**, exactly one: **success** (correct, evidence retrieved), **correct but unsupported**, **wrong with evidence** (the steps got the facts; the reply didn't use them), or **wrong without evidence** (the steps never got the facts). This split separates planning and execution from reasoning over the results.
- **Plans and calls**, automatic: steps per plan, unknown tools, invalid arguments, **unseen ids** (an invoice or payment id used before it appeared in the message or in any earlier result), repeated calls (same tool, same arguments), calls outside the task's useful tools, and limits reached.
- **Cost**: model calls, tool calls, input and output tokens (Ollama's `prompt_eval_count` and `eval_count`), the largest single call, input tokens by call position (context growth), and wall-clock seconds.

## Run it

Prerequisites: Python 3.10+ and [Ollama](https://ollama.com/download) with `qwen2.5:3b` (about 2 GB).

```bash
ollama pull qwen2.5:3b
cd 05-planning/python
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run.py                                   # 4 designs x 9 tasks x 3 seeds = 108 conversations
.venv/bin/python report.py results-qwen2.5_3b.jsonl
```

`run.py` checks the frozen hashes, unloads the model, loads it with one throwaway request, runs the conversations in a fixed order (design, task, seed), and unloads the model. Seed = run number (1–3) for every model call; at most 1,000 tokens per call. It records, for every conversation, whether any other model was loaded in Ollama, and writes `results-<model>-setup.json` (runtime, model digest, frozen and code hashes, power source) and the model's chat template.

| File | What it does |
|---|---|
| `python/systems.py` | The four designs, built from `data/designs.json`: W's fixed plans, the loop, planning, replanning, the shared answer step, and the injected failure |
| `python/tools.py`, `python/llm.py` | Byte-identical copies from `04-agent-memory`: the five read-only tools and the Ollama client |
| `python/run.py` | Checks the frozen hashes, runs every conversation, saves everything |
| `python/report.py` | Grades saved conversations with the rules in `data/evaluation.json`; calls no model. `--todo` lists replies that still need a hand label |
| `python/labels.json` | Hand labels for every distinct reply: answer label, fabricated-action flag, note |

## Results

Setup: Ollama 0.40.17 on an Apple M4 Pro (24 GiB), on battery; `qwen2.5:3b` (digest `357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b`), Q4_K_M; default sampling; 2026-11-24, run started 06:12:20 UTC, 108 conversations in 873 s. The run started after 60 s with no other connection to Ollama, and no other model was loaded during any conversation. Ollama's server log for the run's window has 329 `/api/chat` requests: this run's 328 model calls plus the warm-up. Another client made only status requests in that window (108 `/api/ps`, one `HEAD /`, one `/api/tags`), so no other model work overlapped the run. (Earlier demos in this repository used Ollama 0.40.16, with the same model digest.)

`python report.py results-qwen2.5_3b.jsonl` prints (saved as `python/report-qwen2.5_3b.txt`):

```text
design   n  correct  incorrect  no answer  fabricated  success  unsupp.  wrong+ev  wrong-ev  evidence
W       27        9         15          3           0        9        0        15         3        24
L       27        0         20          7           1        0        0         7        20         7
P       27        0         27          0           3        0        0         7        20         7
R       27        4         22          1           2        4        0         9        14        13

successes (evidence retrieved), per task, of 3 seeds
task           W         L         P         R   tests
Q1         3 (3)     0 (0)     0 (0)     2 (3)   one step
Q2         3 (3)     0 (0)     0 (0)     0 (1)   independent lookups
Q3         1 (3)     0 (0)     0 (0)     1 (1)   a dependent step
Q4         0 (3)     0 (0)     0 (0)     1 (1)   a conditional chain
Q5         0 (3)     0 (0)     0 (0)     0 (0)   synthesis
Q6         0 (3)     0 (3)     0 (3)     0 (3)   reasoning over correct results
Q7         0 (0)     0 (0)     0 (0)     0 (0)   an execution failure with another route
Q8         2 (3)     0 (3)     0 (3)     0 (3)   a request no tool can carry out
Q9         0 (3)     0 (1)     0 (1)     0 (1)   new information

post-hoc, not frozen: replies whose main answer was right, and of those, how many also stated something false
task           W         L         P         R
Q1         3 (0)     0 (0)     0 (0)     3 (1)
Q2         3 (0)     0 (0)     0 (0)     0 (0)
Q3         2 (1)     0 (0)     0 (0)     1 (0)
Q4         0 (0)     0 (0)     0 (0)     1 (0)
Q5         3 (3)     0 (0)     0 (0)     0 (0)
Q6         0 (0)     0 (0)     0 (0)     2 (2)
Q7         0 (0)     0 (0)     0 (0)     0 (0)
Q8         2 (0)     0 (0)     0 (0)     0 (0)
Q9         2 (2)     1 (1)     1 (1)     0 (0)
all       15 (6)     1 (1)     1 (1)     7 (3)

plans                         P        R
first plans                        27       27
first plan steps, mean           1.74     1.85
empty first plans                   0        0
unparseable plans (any)             0        0
first plan got the evidence         7        6
final evidence                      7       13
replanning calls, total             0      133
replans that added steps            0      106
replan_limit                        0       14

calls                        W       L       P       R
unknown tool                 0       0       0      13
invalid arguments            0       2       1      19
injected failure             3       3       3      15
repeated                     0       0       0     106
outside useful tools         0       9      12      72
unseen id in args            0       0       0       4
tool calls, total           36      36      47     216
step_limit (L)               0       0       0       0

Q7 (get_refund_status unavailable)        W     L     P     R
conversations                             3     3     3     3
hit the failure                           3     3     3     3
called it again after failing             0     0     0     3
got invoice date + policy                 0     0     0     0
success                                   0     0     0     0

cost per conversation   model calls  tool calls  input tok med/max  largest call  seconds med/max  total input tok
W                              1.00        1.33        589 / 805            805     3.76 / 8.6           12,705
L                              2.22        1.33      1,065 / 2,270        1,184     4.63 / 15.0          32,922
P                              2.00        1.74        719 / 1,238          911     4.00 / 10.8          21,222
R                              6.93        8.00      7,360 / 16,452       3,260    18.10 / 30.5         210,415

input tokens by model-call position (median; conversations reaching it)
L  1: 405 (27)   2: 531 (27)   3: 893 (6)
R  1: 343 (27)   2: 662 (27)   3: 993 (27)   4: 1,035 (27)   5: 1,303 (25)   6: 1,554 (21)   7: 1,626 (18)   8: 1,406 (15)
```

Columns: `correct`, `incorrect`, `no answer`, `fabricated` are hand labels. `success`: correct and the evidence retrieved. `unsupp.`: correct without it. `wrong+ev`: wrong or no answer although the evidence was retrieved (the reply didn't use it). `wrong-ev`: wrong or no answer, and the steps never retrieved the evidence. `evidence` counts Q8, which has no record evidence, as retrieved by definition. In the per-task table, each cell is successes (conversations with the evidence), of 3 seeds. The post-hoc table is explained below.

With three seeds per task, a difference of one or two conversations is noise, not evidence.

Input tokens are Ollama's `prompt_eval_count` for each model call's whole prompt, summed over the conversation: history re-sent in a later call is counted again, and the count includes any part Ollama reused from its prompt cache (every call reported `prompt_eval_cached_count` no larger than it), so it measures what was sent, not what was computed. Seconds are wall-clock time for the whole conversation (every model call and tool call, through the reply), measured by `run.py`; model loading and the isolation checks are outside the timer. The 8-call limit counts model calls, not tool calls: one plan can hold up to 6 tool calls.

What the recorded conversations show (every reply was read and labeled by hand):

- **Planning and reasoning failed separately.** Excluding Q8, the steps retrieved the evidence in W 21, L 4, P 4, R 10 of 24 conversations. W, whose steps a developer wrote, still answered wrong in 14 of the 21 conversations where its steps retrieved the evidence (Q3, Q4, Q5, Q6, Q9; Q8 excluded, as above); L and P answered all 4 of theirs wrong, R 6 of 10. W read the payment list as "no duplicate charge" (Q4), gave totals of $60, $90 and $30 instead of $65 (Q6), and added false claims about the refund to a correct seat charge (Q5). P's one-step plan on Q6 was W's plan, so their replies are byte-identical.
- **Placeholders for values the planner didn't have.** On Q3, all three L runs called `get_refund_status` with `most_recent_Pro_invoice_id` and never listed the invoices. Across all tasks, a refund-status call whose `invoice_id` wasn't a real invoice id appeared in L 4, P 8, R 8 conversations (W 0): `invoice123`, `REPLACE_WITH_INVOICE_ID`, `recent_invoice_id`, `<extra_charge_invoice_id>` and others. The frozen "unseen id" check matches only `INV-<digits>`/`PAY-<digits>`, so it counted 4 calls (all R); the conversation counts above come from reading every refund-status argument. P's plan reasoning sometimes described a dependent step that the plan then left out.
- **The loop stopped early; the replanner didn't stop.** L never used more than 3 model calls (21 of 27 conversations made one round of tool calls), and 3 replies ended by announcing a step that never ran. R hit its replan limit in 14 of 27 conversations; 106 of its 216 tool calls repeated an earlier identical call, 75 of them a call that had already succeeded, which its prompt said not to do (repeating a step that had returned an error was allowed); it made all 13 calls to tools that don't exist (`remove_plan_seat` 6, `calculate_total_payment_amount` 5, `get_plan_details`, `get_plan_status`); and 6 times it said it was done without the evidence.
- **Replanning added facts.** R's first plans retrieved the evidence in 3 of 24 conversations (all Q6); after replanning, 10. It gave the only correct Q4 answer, reaching the payment list in its fifth batch of steps.
- **No design rerouted around the failure (Q7).** Every conversation hit the outage; none read both the invoice date and the 30-day policy. W and L replied that the service was unavailable (6 no answers); P's three replies were wrong; R retried the failed tool in all three runs and hit the replan limit each time, after replanning reasoning such as "We need to use another method".
- **Plans for actions no tool supports (Q8).** Only W2 and W3 said the assistant couldn't do it. Six replies (L 1, P 3, R 2) said the seats were being removed or the refund started; one R reply invented a support email address and phone number.
- **Reasoning that contradicts the plan beside it.** For example, R on Q6: "the calculate_total_payment_amount tool is unknown and should not be used ... No further steps are needed", with `done: false` and that tool as the next step.
- **Cost.** Input tokens for the whole run: W 12,705, P 21,222, L 32,922, R 210,415 (16.6x W). Median seconds per conversation: W 3.8, P 4.0, L 4.6, R 18.1. The median input of R's model calls grew from 343 tokens at the first call to 1,626 at the seventh.

**Post-hoc reading (not part of the frozen evaluation).** The frozen rule counts a reply as correct only if every value or claim in it is true. After labeling, each reply was also marked `main_answer_right` in `labels.json`: whether it gives what the task's `correct_requires` asks for, even if it also states something false. W 15 (6 of them with something false), L 1 (1), P 1 (1), R 7 (3). This changes no outcome; it shows how much of the gap is false detail around a right answer. Two R replies reached Q6's $65 by wrong working (one summed only three charges; the other wrote "$115 ($25 + $25 + $25 + $15)") and are labeled incorrect.

The check run (`python/results-check.jsonl`, 12 conversations: seed 1, Q3, Q7, Q8, all designs) was made with the same code before the full run. It is kept as a record and is not part of these numbers.

## Limitations

- An educational experiment, not a benchmark: one small local model, nine tasks, three seeds, one synthetic account.
- One prompt per planner, written before the run. Other planner prompts, plan formats with references to earlier results (ReWOO, LLMCompiler), step-by-step replanning, a larger model, or code-level validation of plans are not tested here.
- W assumes correct routing, so it is a ceiling for a fixed workflow, not a deployable design.
- Q7's error message doesn't say whether the outage is temporary, so retrying once is reasonable; retrying until the limit is not.
- Answers are labeled by hand, by one person, against the written rules in `data/evaluation.json`. The `main_answer_right` field was added after the run.
- Token counts are Ollama's `prompt_eval_count` and `eval_count` for this model's tokenizer; latencies are for this machine, on battery, with nothing else running.
