# ai-agent-patterns

Runnable examples for the AI Agents articles on [snehalagrawal.com](https://snehalagrawal.com/articles/): tools, agent loops, state, memory, planning, human-in-the-loop, and failure modes.

| Demo | Article | What it shows |
|---|---|---|
| [`01-tool-design`](01-tool-design/) | [#23 Tools, Function Calling, and the Foundation of AI Agents](https://snehalagrawal.com/articles/tools-function-calling-and-the-foundation-of-ai-agents/) | The same agent with three toolsets: how the shape of a tool changes what the model does next |
| [`02-agentic-workflows`](02-agentic-workflows/) | [#24 Agentic Workflows: When Should an LLM Control the Flow?](https://snehalagrawal.com/articles/agentic-workflows-when-should-an-llm-control-the-flow/) | The same assistant as a fixed workflow, an agent, and a hybrid: who should decide the steps |
| [`03-multi-agent`](03-multi-agent/) | [#25 Single-Agent vs Multi-Agent Systems: What Does Another Agent Add?](https://snehalagrawal.com/articles/single-agent-vs-multi-agent-systems/) | The same assistant as one agent, a manager with two specialist agents, and an agent with a reviewer: what a second agent adds and what it costs |
| [`04-agent-memory`](04-agent-memory/) | [#26 Memory in AI Agents: What Does an Agent Actually Need to Remember?](https://snehalagrawal.com/articles/memory-in-ai-agents-what-does-an-agent-actually-need-to-remember/) | The same assistant with no memory, transcripts, summaries, or structured notes of earlier conversations: what to carry forward, what to look up, and what happens when memory is stale |
| [`05-planning`](05-planning/) | [#27 Planning, Reasoning, and Execution in AI Agents: Who Decides the Steps, and When?](https://snehalagrawal.com/articles/planning-reasoning-and-execution-in-ai-agents/) | The same assistant with a fixed workflow, a step-by-step loop, a plan written up front, and a plan revised after every result: where planning fails, where reasoning over the right results fails, and what replanning costs |
| [`06-state-and-workflows`](06-state-and-workflows/) | [#28 Building AI Agents With State and Workflows](https://snehalagrawal.com/articles/building-ai-agents-with-state-and-workflows/) | The same refund assistant as an agent loop whose task progress lives only in its message list and as a state machine whose case record is owned by code: wrong refunds, retries, and recovery after a crash |
| [`07-human-in-the-loop`](07-human-in-the-loop/) | [#29 Human-in-the-Loop AI Agents: When Should an Agent Stop and Ask?](https://snehalagrawal.com/articles/human-in-the-loop-ai-agents-when-should-an-agent-stop-and-ask/) | The same assistant with a refund tool that pauses for a reviewer: hard checks in code, approval tiers and roles, expiry, a re-check before execution, execution once, an audit trail, and what the agent tells the customer while it waits and after each decision. Also used by [#39 Security Considerations for Production AI](https://snehalagrawal.com/articles/security-considerations-for-production-ai/): ten probes with a scripted, fully fooled model, showing which boundaries the code enforces and which paths it leaves open |
| [`08-failure-modes`](08-failure-modes/) | [#30 Why AI Agents Fail: Common Failure Modes and Design Patterns](https://snehalagrawal.com/articles/why-ai-agents-fail-common-failure-modes-and-design-patterns/) | No model calls: the recorded decisions of 01–07 replayed through guards at each failure boundary (argument provenance, repeats, budgets, a refund policy in code, typed handoffs, reply checks): what each guard would have stopped, what correct work it would have stopped too, and which failures pass every guard |

Each demo has its own README with setup, how to run it, expected output, and limitations. All data is synthetic.

## Linting

Python is checked with [Ruff](https://docs.astral.sh/ruff/) from the repository root:

```bash
pip install -r requirements-dev.txt
ruff check .
```
