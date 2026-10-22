# How to read the code (guide for new developers)

The code lives in `src/hm_agent/` as a package of small files instead of one big script.
This guide explains how to find your way around it.

## The core idea

In a single file you scroll top to bottom. In a package each file has **one job**, and every
file starts with a docstring that says what that job is and where it fits. You do not read
every file in order. Pick a question, follow the imports to the file that answers it, and
stop when you have what you need.

Python mechanics you need:

- `from hm_agent.core.llm import load_model` means "`load_model` lives in `src/hm_agent/core/llm.py`".
  Dots in the import path are folders.
- A folder with an `__init__.py` is a package.
- `python -m hm_agent` runs `hm_agent/__main__.py`; `python -m hm_agent.customer_support.rag` runs
  `hm_agent/customer_support/rag/__main__.py`.

## The map

```
config/                  base.yaml (shared defaults) + dev.yaml (overrides)
src/hm_agent/
  core/                  shared by every agent
    config.py              loads base.yaml and merges <APP_ENV>.yaml over it
    settings.py            exposes config values as constants (RAG_TOP_K, LOG_LEVEL, ...)
    llm.py                 builds the Groq chat model ("small" / "big") from config
    db.py                  the only place that talks to Neon (queries + checkpointer)
    state.py               agent names and the data passed between graph nodes
    logging_setup.py       log format and correlation fields
  supervisor/            one package per agent
    agent.py               picks the agent; also the out_of_scope reply
  order_management/
    agent.py               ReAct agent whose tools come from the MCP server
    mcp_server.py          stdio server with the order SQL tools (separate process)
    mcp_client.py          loads those tools for LangChain; injects customer_id from the run config
    a2a_service.py         runs this agent as its own A2A service
  customer_support/
    agent.py               condense question -> RAG -> respond
    a2a_service.py         runs this agent as its own A2A service
    rag/                   the knowledge-base pipeline (only this agent uses it)
      retriever.py           the only place that talks to Qdrant
      pipeline.py            retrieve -> generate (cited) -> retry -> finalize
  a2a/                   generic A2A plumbing: server, client, cards, remote node
  graph.py               wires the agents into one graph
  cli.py                 the command you run
```

Dependencies only point upward. Everything in `core/` imports nothing from the agents, and
agents never import each other. The agents use them, `graph.py` uses the agents, and `cli.py` uses the graph.
That is why you can read bottom-up or top-down without getting lost.

## The big picture

```
START -> supervisor -> order_management -> END      (SQL tools on Neon)
                    -> customer_support -> END      (RAG over Qdrant help articles)
                    -> out_of_scope     -> END      (local node, fixed polite reply)
```

- The **supervisor** reads the recent conversation and picks one of the three. A short
  follow-up ("and how long does it take?") stays with the agent that answered the previous
  turn. The supervisor is the only place that decides scope, so the sub-agents assume the
  message is theirs.
- **order_management** answers only through two read-only tools (last purchase, purchase
  count), served by an MCP server. The customer id comes from the run config, never from
  the LLM: the MCP client hides the argument from the model and overwrites it on every call.
- **customer_support** answers damaged / defective item questions from the help articles,
  with citations.
- **out_of_scope** replies with a fixed message when no agent fits.

## Reading order: supervisor and order flow

Read it in the order a request travels.

1. **`cli.py`**: what happens when you run the command. It loads `.env`, parses arguments,
   sets up tracing and the thread id, opens the Neon checkpointer, builds the graph and
   invokes it. Skip the Langfuse details on the first pass.
2. **`graph.py`**: the most useful file. It shows the whole flow above in about 20 lines.
   Come back here whenever you are lost.
3. **`core/state.py`**: the three agent names (`ORDER_MANAGEMENT`, `CUSTOMER_SUPPORT`,
   `OUT_OF_SCOPE`) are defined once here and used as node names, message names and routing
   values. `AgentState` holds the message history and `next_agent`, the supervisor's choice.
4. **`supervisor/agent.py`**: the prompt, then `supervisor_node` (asks the LLM to choose),
   then `route_to_agent` (reads the choice), then `out_of_scope_node`. If the model returns
   no decision, the message falls back to customer support, whose relevance gate handles
   off-topic text cheaply.
5. **`order_management/agent.py`**, **`order_management/mcp_client.py`** and
   **`order_management/mcp_server.py`**: the agent is a prompt plus tools served over MCP.
   The server (stdio, spawned per tool call) runs the SQL through `fetch_one` in `core/db.py`; the
   client hides `customer_id` from the model and injects it from the run config. MCP tools are
   async-only, so `mcp_client.py` wraps each one in a sync call (`asyncio.run`) to keep the
   graph synchronous with the sync Postgres checkpointer.
6. Only if needed: everything in `core/` (`llm.py`, `db.py`, `config.py`, `settings.py`). Treat them as plumbing.

## Reading order: customer support

Customer support is a small pipeline made of two graphs, one inside the other.

1. **`customer_support/agent.py`**: start at the bottom with `build_customer_support_agent()`.
   It wires `condense_query -> rag -> respond`:
   - `condense_query_node`: rewrites a follow-up ("what about the refund?") into a
     standalone question using the last few messages. On the first turn there is nothing to
     resolve, so it passes the message through and makes **no LLM call**.
   - `rag`: the whole RAG graph from `customer_support/rag/pipeline.py`, used as one node.
   - `respond_node`: wraps the cited answer in an AI message named `customer_support`, so
     the supervisor can see which agent wrote it on the next turn.
   - `CustomerSupportState` has `query` and `answer` next to `messages`. These are the only
     fields shared with the RAG graph; its documents, claims and retry counter stay private
     inside it.
2. **`customer_support/rag/pipeline.py`**: start at the bottom with `build_rag_graph()`, then read the nodes
   in flow order:
   - `retrieve_node`: fetches the top documents, then applies a **relevance gate**: chunks
     scoring below `min_score` are dropped.
   - `generate_node`: if nothing survived the gate it skips the LLM. Otherwise the model
     returns a list of claims, each citing document numbers. Citations that do not point at
     a retrieved document are removed, and claims left with no citation are discarded.
     Documents are wrapped in `<document>` tags and treated as untrusted data (prompt
     injection protection).
   - `route_after_generate`: if there are cited claims, go to `finalize`. If not, and a
     retry is left, go to `rewrite_query`.
   - `rewrite_query_node`: rewrites the question into a keyword-rich search query, then
     loops back to `retrieve` (at most `max_retries` times).
   - `finalize_node`: builds the answer with `[n]` markers and a `Sources:` list, or the
     fixed `NO_ANSWER` message if nothing is supported.
3. **`customer_support/rag/retriever.py`**: `retrieve()` embeds the query locally with fastembed and queries
   Qdrant. It is the only module that talks to Qdrant.
4. **`customer_support/rag/__main__.py`**: runs the RAG graph alone, without the supervisor:
   `uv run python -m hm_agent.customer_support.rag "What happens if my package can't be delivered?"`

## Where settings live

Change values in `config/base.yaml`, or override them per environment in `config/dev.yaml`.
`APP_ENV` (default `dev`) picks the override file. Code reads them through `core/settings.py`.
Secrets (API keys, DSNs) stay in `.env`, never in the YAML files.

| Section | Keys |
|---|---|
| `llm` | `small`, `big` (Groq model names) |
| `embedding` | fastembed model; must match the one used to ingest the Qdrant collection |
| `rag` | `collection`, `top_k`, `max_retries`, `min_score`, `condense_history_messages` |
| `supervisor` | `history_messages` |
| `database` | `checkpoint_schema` |
| `defaults` | `customer_id` |
| `logging` | `level` |
| `tracing` | `tags` |

## Running it

```
uv run python -m hm_agent "What was the last item I bought?"
uv run python -m hm_agent --customer-id cust002 "How many purchases have I made?"
uv run python -m hm_agent --thread-id <id> "And the one before?"    # continue a conversation
uv run python -m hm_agent.customer_support.rag "What happens if my package can't be delivered?"   # RAG only
```

Each run prints a `thread_id`. Pass it back with `--thread-id` to continue the conversation.

## Habits that help

- **Read the docstring first**, then decide whether you need the body. Every function lists
  its arguments and return value. Graph nodes also say "Reads: X. Writes: Y", which tells
  you the data flow without reading the code.
- **Use Go to Definition (F12) and Find All References (Shift+F12)** in VS Code instead of
  scrolling. They do what scrolling did in a single file.
- **Read the INFO log as a trace.** Lines such as `node=supervisor agent=...`,
  `node=retrieve gate=min_score ...` and `tool=get_purchase_count` (logged by the MCP server process) show the path a request
  took.
- **Add new things in the matching place.** A tool goes in the owning agent's package (an MCP server such as `order_management/mcp_server.py`), a new agent gets its own package,
  a setting in `config/base.yaml` (plus a constant in `core/settings.py`). Then wire it in
  `graph.py`.
- **Tests use mocks and the test database.** Never call the real LLM in a loop, and never
  point tests at the main database (see `CLAUDE.md`).
