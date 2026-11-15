# 01 · Tool design: does the shape of a tool change what an agent does next?

Companion demo for **Article #23, ["Tools, Function Calling, and the Foundation of AI Agents"](https://snehalagrawal.com/articles/tools-function-calling-and-the-foundation-of-ai-agents/)** on snehalagrawal.com.

The same support assistant, model, prompt, data, and agent loop answer the same ten questions with three different toolsets. Only the tools change. The demo records what each model did with each toolset: which tools it called, whether the facts its answer needed ever reached it, whether the answer was right, and how many tokens it took.

It builds on [`tool-calling`](https://github.com/SnehalAgrawal/ai-engineering-examples/tree/main/tool-calling) from Article #8: same loop, same local models, same Ollama version, same system prompt.

## The three toolsets

| Toolset | Tools | What changes |
|---|---|---|
| **A** endpoint-shaped | `search_help_center`, `get_invoices` | Article #8's read tools. `get_invoices` returns the invoice rows as stored; errors are short. |
| **B** richer responses | the same two tools, names, and descriptions | `get_invoices` adds each invoice's age and refund deadline; an empty search says what to try; errors say how to call the tool. Two changes at once (results and errors), so B doesn't isolate either one. |
| **C** plus a task-shaped tool | A's two tools + `get_refund_status` | One added tool that works out, from the account data at call time, whether each invoice can be refunded today, why, and its deadline. Nothing is precomputed per question. |

All tools only read. The code in `toolsets.py` parses and validates every requested call before anything runs.

## What was frozen first

The test material in `data/` was written and frozen before any toolset code existed: its hashes were recorded at 2026-11-15 08:49:32 UTC, and `toolsets.py` was created after that. It was committed on its own (ca4aa54, 08:56 UTC), after the code was written and before any model was run:

- `account.json`: synthetic billing data (Article #8's account plus two invoices and a description on each).
- `help_center.json`: Article #8's eight help articles, without its three planted-instruction tickets.
- `prompt.json`: Article #8's basic system prompt, unchanged. It names no tools, so it fits every toolset.
- `questions.json`: ten questions (eight that need more than one fact, two single-fact controls), each with its expected answer and the facts it depends on, and the grading rules.
- `FROZEN.json`: SHA-256 of each file. `run.py` refuses to run if any file has changed.

### Grading rules (from `questions.json`)

- **Facts retrieved:** every fact the question depends on appears in the text of a tool result that executed in that conversation. It shows what the model was given, not what it understood.
- **Answer label**, by hand, for every conversation: `correct`, `incorrect`, or `no_answer`, stored in `python/labels.json` with the SHA-256 of the exact reply (a label whose hash doesn't match is reported, not used).
- **Successful tool interaction:** labeled correct **and** facts retrieved. A correct answer without the facts is reported separately, not counted as a success.
- **Invalid call:** unknown tool, arguments that aren't JSON, or arguments that fail the schema. A tool call written into the reply as text is counted separately.

## Run it

Prerequisites: Python 3.10+ and [Ollama](https://ollama.com/download) with `llama3.2:3b` and `qwen2.5:3b` (about 2 GB each).

```bash
ollama pull llama3.2:3b
ollama pull qwen2.5:3b
cd 01-tool-design/python
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run.py --model llama3.2:3b      # 3 toolsets x 10 questions x 3 seeds = 90 conversations
.venv/bin/python run.py --model qwen2.5:3b
.venv/bin/python report.py results-llama3.2_3b.jsonl results-qwen2.5_3b.jsonl
```

`run.py` unloads the model before and after each run and runs the conversations in a fixed order (toolset, question, seed). Each run also writes `results-<model>-setup.json` (Ollama version, model digest, settings, frozen hashes) and the model's chat template, `template-<model>.txt`.

`report.py` calls no model: it grades the saved conversations against the frozen rules and the hand labels. `report.py <results> --unlabeled` prints every reply that still needs a label.

| File | What it does |
|---|---|
| `python/toolsets.py` | The three toolsets: definitions the model sees, and the code that validates and runs a call |
| `python/agent.py` | The agent loop (Article #8's): at most 5 model calls, 1,000 tokens each |
| `python/llm.py` | Ollama `/api/chat` over plain HTTP; records input, cached, and output tokens per model call |
| `python/run.py` | Checks the frozen hashes, runs every conversation, saves them |
| `python/report.py` | Grades saved conversations and prints the comparison |

## Important: the chat template is part of the tool interface

A validation run (`python/results-check.jsonl`, 6 conversations, kept as evidence and not part of the results) found that the two models don't see the same thing after a tool result:

- **`llama3.2:3b`** (Ollama 0.40.16): the template adds the tool definitions only to a user message that is the last message, and its system text says "When you receive a tool call response, use the output to format an answer to the orginal user question." Every request after a tool result reaches the model **without tools**. The per-call input tokens are consistent with it: in the full run, whenever the tool result was short (under 400 characters), the second request was smaller than the first (76 of 76), although it carried more conversation; long invoice lists outweighed the missing definitions. For `qwen2.5:3b`, the second request was always larger (80 of 80).
- **`qwen2.5:3b`**: the template lists the tools in the system message on every request.

So in this setup, `llama3.2:3b` can request tools only in its first response. Its results compare the toolsets on the first decision and on how it uses one result; only `qwen2.5:3b` tests tool design across a multi-step trajectory. This is a property of the configuration, not of the model: Meta's [Llama 3.2 prompt format](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/text_prompt_format.md) describes tool definitions in the system message as the default. The template was deliberately not changed, to keep the setup identical to Article #8's; a system-message template would be a separate experiment.

Ollama doesn't return the rendered prompt, so the saved template and the per-call token counts are the record of what the model saw.

## Results

Setup: Ollama 0.40.16 on an Apple M4 Pro; `llama3.2:3b` (`a80c4f17acd5`) and `qwen2.5:3b` (`357c53fb659c`), both Q4_K_M; default sampling, seed = run number (1–3), at most 5 model calls and 1,000 tokens per call; each model unloaded before and after its run; 2026-11-15. The full run took 99 s (`llama3.2:3b`) and 224 s (`qwen2.5:3b`) for 90 conversations each. Exact settings, frozen hashes, and template hashes are in `python/results-<model>-setup.json`.

`python report.py results-llama3.2_3b.jsonl results-qwen2.5_3b.jsonl` prints:

```text
model         toolset  n  correct  incorrect  no answer  facts  success  unsupp.  invalid  as text  model calls  tool calls  input tok
llama3.2:3b   A       30     7/30      18/30       5/30   6/30     6/30     1/30        0        0         2.00        1.00        436
llama3.2:3b   B       30     5/30      15/30      10/30   7/30     5/30     0/30        0        1         2.00        1.00        444
llama3.2:3b   C       30    10/30      10/30      10/30  10/30     9/30     1/30        2        1         1.97        0.97        517
qwen2.5:3b    A       30     7/30      22/30       1/30   7/30     6/30     1/30        0        0         1.93        0.93        724
qwen2.5:3b    B       30     6/30      21/30       3/30  24/30     5/30     1/30        0        0         1.93        0.93        861
qwen2.5:3b    C       30    14/30       6/30      10/30  15/30    13/30     1/30        0        0         2.00        1.07        840
```

Columns (see "Grading rules" above): `correct`, `incorrect`, and `no answer` are the hand labels, and add up to n. `facts`: every fact the question depends on appeared in a tool result that executed. `success`: labeled correct **and** facts retrieved. `unsupp.`: labeled correct without the facts; in every row, `success` + `unsupp.` = `correct`. `invalid`: tool calls that failed the name, JSON, or schema check. `as text`: final replies containing a tool call written as text. `model calls` and `tool calls`: means per conversation. `input tok`: median input tokens per conversation, summed over its model calls (Ollama's `prompt_eval_count`, cached tokens included).

On the eight multi-fact questions (24 conversations per model and toolset), successes were 0 / 0 / 6 for `llama3.2:3b` and 3 / 2 / 10 for `qwen2.5:3b` (A / B / C). With only three seeds per question, a difference of one or two conversations (such as A's 6 and B's 5 successes) should be treated as noise rather than evidence of a real improvement.

What the recorded conversations show (every reply was read and labeled by hand; see `python/labels.json`):

- **Richer results got the facts to the model, not into the answer.** With B, `qwen2.5:3b` retrieved the facts in 24 of 30 conversations (7 with A), but succeeded in 5. In 18 of B's 24 multi-fact conversations it had every fact and still didn't answer correctly (17 wrong, 1 no answer), mostly by comparing a deadline with today's date wrongly.
- **Deciding in code helped most.** With C, the required facts reached `qwen2.5:3b` in 15 of 30 conversations, and 13 of those were successful (with B: 24 of 30, and 5 successful). For `llama3.2:3b` with C: 10 with the facts retrieved, 9 successes. The gains are on questions the tool's output answers directly (Q01–Q03). The question that needs one more step on top of it (Q08, the oldest refundable invoice) was never answered correctly, with any toolset.
- **The task-shaped tool cost a simple question.** On the policy control (Q09), A and B answered correctly in 10 of 12 conversations; C in 1 of 6, and that one without retrieving the policy. `llama3.2:3b` reached for `get_refund_status`; `qwen2.5:3b` asked the customer for an invoice id.
- **Invented arguments.** `get_refund_status` works without an `invoice_id` (it returns every invoice), but the models filled one in for 39 of 44 calls (`llama3.2:3b` 24 of 26, `qwen2.5:3b` 15 of 18), and 23 calls didn't execute: `"get_invoices"`, empty strings, `null`, invented ids, and, five times, `INV-0000`, the example in the parameter's own description.
- **B's error messages had nothing to act on.** Neither model made an invalid call or got an error result with A or B, so no error message was ever shown. The closest cases were not errors: 6 searches (all `llama3.2:3b`) that ran successfully and found nothing. Under B, that empty result also carried a hint to try other words, but the next request had no tools (see the template note above), so the hint couldn't change a next call. The experiment didn't exercise error recovery and doesn't measure the value of better errors.
- **A and B start identically.** They present the same tools and descriptions, so each model's first tool decision was the same under A and B in all 60 pairs. With both, `llama3.2:3b` sent 23 of 24 multi-fact questions to the help-center search and never saw the invoices.
- **Announced, not taken.** Six `qwen2.5:3b` replies ended by describing a next step ("Let's check the next oldest invoice.") without requesting it; the loop treats that as done.

Five conversations were labeled correct without retrieving the facts (counted as `unsupp.`, not as successes). The validation run (`python/results-check.jsonl`, 6 conversations, before the full run) is kept as evidence of the template finding and is not part of these numbers.

## Limitations

- An educational experiment, not a benchmark: two small local models, ten questions, three seeds, one synthetic account.
- The facts check is a value-presence check on tool results: it shows what reached the model, not what it understood. Answers are graded by hand, by one person, against written rules.
- Toolset B changes results and errors together, so it can't say which of the two mattered.
- Token counts are Ollama's `prompt_eval_count` (cached tokens included) and `eval_count` for these models' tokenizers; other providers count differently.
