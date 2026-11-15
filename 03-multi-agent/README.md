# 03 · Multi-agent: what does a second agent add?

Companion demo for **Article #25, "Single-Agent vs Multi-Agent Systems"** on [snehalagrawal.com](https://snehalagrawal.com/articles/) (not published yet; this README will link to it when it is).

The same support assistant from [`02-agentic-workflows`](../02-agentic-workflows/) (Article #24) answers the same twelve questions three ways: as one agent, as a manager that delegates to two specialist agents, and as one agent whose draft is checked by a reviewer agent. The question isn't whether more agents are better. It's what capability each extra agent adds, and whether that is worth another context, another decision-maker, another handoff, and more model calls.

> **Status: design frozen, not yet implemented or run.** This README describes the experiment as frozen in `data/`. The system code, the run, and the results come after the frozen design is committed.

## The three systems

| System | Shape | What the extra agent is for |
|---|---|---|
| **S** single agent | Article #24's agent A, unchanged: one agent loop with the five data tools | (baseline) |
| **M** manager + specialists | A **manager** agent whose only tools are two agents: a **billing specialist** (invoices, refund status, payments, plan history) and a **help-center specialist** (help-center search). Each specialist runs its own tool loop on the manager's request and returns its reply as the tool result. | Specialized contexts: each agent has fewer tools and a narrower job |
| **R** agent + reviewer | S writes a draft. A **reviewer** agent with all five data tools checks it against the records, then replies `APPROVE` (the draft is sent) or with a replacement reply. One pass; no return to the drafter. | Independent verification |

Design choices, frozen in `data/agents.json`:

- **The manager uses the customer-facing prompt from Article #24, unchanged.** That prompt says only "Use the available tools to help the customer", so S and M differ only in what the tools are.
- **Specialists see only the manager's request,** not the customer's message or the manager's conversation. This is the "agents as tools" (manager) pattern as documented: the manager writes the task. Sharing the full context with every agent is a different design, discussed in the article but not tested here.
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

## Setup (planned)

- Ollama with `qwen2.5:3b` only. `llama3.2:3b` isn't used: with the Ollama template recorded in `01-tool-design`, it gets no tools after the first tool result.
- 3 systems × 12 questions × 3 seeds = **108 conversations**. Seed = run number (1–3) for every model call in a conversation; at most 5 model calls per agent loop and 1,000 tokens per call.
- Fixed run order (system, question, seed); the model is unloaded before and after the run and loaded with one throwaway request first; the run waits until no other process has used Ollama for 60 s, and records for each conversation whether another model was loaded.
- The data tools are the same five functions as in `02-agentic-workflows`; all only read.

## Limitations

- An educational experiment, not a benchmark: one small local model, twelve questions, three seeds, one synthetic account.
- One design of each pattern, written before the run. Other prompts, other models, or full-context delegation might behave differently; that would be a separate experiment.
- Published multi-agent results (for example Anthropic's research system) come from frontier models on broad research tasks. This experiment doesn't test that setting and isn't meant to contradict it.
- Answers, handoff messages, and failure stages are labeled by hand, by one person, against the written rules in `data/evaluation.json`.
