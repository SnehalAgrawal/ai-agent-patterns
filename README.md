# ai-agent-patterns

Runnable examples for the AI Agents articles on [snehalagrawal.com](https://snehalagrawal.com/articles/): tools, agent loops, state, memory, planning, and human-in-the-loop.

| Demo | Article | What it shows |
|---|---|---|
| [`01-tool-design`](01-tool-design/) | [#23 Tools, Function Calling, and the Foundation of AI Agents](https://snehalagrawal.com/articles/tools-function-calling-and-the-foundation-of-ai-agents/) | The same agent with three toolsets: how the shape of a tool changes what the model does next |
| [`02-agentic-workflows`](02-agentic-workflows/) | [#24 Agentic Workflows: When Should an LLM Control the Flow?](https://snehalagrawal.com/articles/agentic-workflows-when-should-an-llm-control-the-flow/) | The same assistant as a fixed workflow, an agent, and a hybrid: who should decide the steps |
| [`03-multi-agent`](03-multi-agent/) | [#25 Single-Agent vs Multi-Agent Systems: What Does Another Agent Add?](https://snehalagrawal.com/articles/single-agent-vs-multi-agent-systems/) | The same assistant as one agent, a manager with two specialist agents, and an agent with a reviewer: what a second agent adds and what it costs |
| [`04-agent-memory`](04-agent-memory/) | #26 Memory in AI Agents: What Does an Agent Actually Need to Remember? (not published yet) | The same assistant with no memory, transcripts, summaries, or structured notes of earlier conversations: what to carry forward, what to look up, and what happens when memory is stale |

Each demo has its own README with setup, how to run it, expected output, and limitations. All data is synthetic.

## Linting

Python is checked with [Ruff](https://docs.astral.sh/ruff/) from the repository root:

```bash
pip install -r requirements-dev.txt
ruff check .
```
