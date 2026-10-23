# What changed with A2A (vs. one process with in-process agents)

Before: one process held the supervisor and both agents as LangGraph subgraphs (`graph.py` added
them as nodes). After: the supervisor is a client; each agent is its own A2A server. The
out-of-scope decline stays a local node: it only returns a constant, so a service would add a
process, a port and a failure mode for nothing. The routing
LLM call, the agents' internals, MCP and the database are unchanged.

```
BEFORE   cli process:  supervisor -> [order subgraph | support subgraph | out_of_scope node]

AFTER    cli process:  supervisor --A2A/HTTP--> :8001 order_management   --MCP stdio--> order SQL server
                                   --A2A/HTTP--> :8002 customer_support  --> Qdrant, local embeddings
                                   -> out_of_scope (local node, fixed reply: no LLM, no DB, no A2A)
```

| Topic | Single process | With A2A |
|---|---|---|
| Call from supervisor to agent | Python call inside LangGraph | HTTP `SendMessage` (JSON-RPC) after fetching the Agent Card |
| Agent discovery | Hard-coded node names | Agent Card at `/.well-known/agent-card.json` (name, URL, skills, version) |
| Who owns conversation state | One graph state (messages) | Supervisor only (Neon checkpointer). Agents are stateless per request: they get the latest text plus the last 6 messages |
| Context passed | Shared `messages` and run config (`customer_id`, callbacks) | Explicit: text part, a data part with history, and request metadata (`customer_id`, `thread_id`); `thread_id` is also the A2A `contextId` |
| Result | Messages appended to state | A `Task` with state and an artifact; the supervisor turns the artifact into an `AIMessage(name=<agent>)` |
| Failure | An exception breaks the turn | Agent down or failed task: supervisor logs the error and answers with an apology; other agents keep working |
| Deploy and scale | One unit | Each agent can run, scale and be released alone, and could be written in another language or framework |
| Startup and memory | Everything loads in the CLI | Embedding model and Qdrant load only in `customer_support`; MCP/DB only in `order_management` |
| Latency | No hop | Extra card fetch + HTTP per turn: 30-190 ms on localhost (`a2a-wire.jsonl`, lines 1-3 and 5-7). Dominated by the agents' own LLM and tool time |
| Tracing | One Langfuse trace per turn | Two traces per turn (supervisor, agent), same `session_id`, **not linked** (see `a2a-explained.md`, section 8) |
| Ops | `uv run python -m hm_agent` | Start the agents first (`scripts/a2a/run_agents.py`), then the CLI |

What did not change: supervisor routing prompt and model, agent prompts, the MCP server and its
`customer_id` injection, RAG pipeline, Neon checkpointing, log format.

What got worse: more moving parts (3 processes, ports, health), a network failure mode, tracing split
across processes, and a few extra HTTP round trips. The gain is isolation and independent
deployment, which only matters if agents are owned by different teams, scale differently, or must
run in different places. For this app's size, in-process was simpler; A2A is a deliberate architecture choice.
