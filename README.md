# H&M Fashion Assistant

A multi-agent assistant for H&M, built with LangGraph. A supervisor reads each
message and routes it to one of four sub-agents. The agents
run as separate A2A services, and an AgentCore-style entrypoint exposes the assistant over HTTP with streaming.

```
START -> supervisor -> order_management  -> END   (own orders and purchases, via MCP SQL tools)
                    -> customer_support  -> END   (agentic RAG over the support knowledge base)
                    -> shipment_tracking -> END   (carrier lookup; slow, streams progress, push fallback)
                    -> fashion           -> END   (outfit and similar-product advice from catalog search)
                    -> out_of_scope      -> END
```

| Component | What it does |
|---|---|
| `supervisor` | Classifies the latest message (with a short history window) and picks the agent. A message that mixes topics (e.g. order status plus outfit advice) fans out to up to two agents in parallel, each with its own sub-question; short follow-ups ("how long does it take?") stay with the previous agent without an LLM call (`supervisor.max_agents`, `supervisor.sticky_max_words`). |
| `order_management` | ReAct agent over read-only tools served by an MCP server on the Neon `order_management` schema. The `customer_id` is injected from the run config, so the model never chooses whose data is read. |
| `customer_support` | Condense query -> retrieve from Qdrant -> answer with citations (retries once if no citation). |
| `shipment_tracking` | Finds the order id (`ORD-1007`) in the conversation and runs a deliberately slow dummy carrier lookup. It streams progress lines, and its Agent Card advertises push notifications. It makes no LLM calls. |
| `fashion` | ReAct agent that recommends products and outfits. Its tools come from a fashion MCP server: semantic product search over a Qdrant collection built from `product_catalog`, purchase history, a short styling guide and the customer's style profile. It recommends only products the tools return. |
| Checkpointer | `PostgresSaver` on Neon (schema `checkpoints`) stores conversation state per `thread_id`. |
| Tracing | Langfuse; every turn is a trace, all turns of a thread form one Langfuse session. |

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env        # then fill in the values
```

| Variable | Used for |
|---|---|
| `GROQ_API_KEY` | LLM calls (Groq free tier, rate limited) |
| `NEON_POSTGRES_CONNECTION_STRING` | Order data and the conversation checkpointer |
| `NEON_POSTGRES_TEST_CONNECTION_STRING` | Separate test database (`neondb_test`); tests never use the main DSN |
| `QDRANT_ENDPOINT`, `QDRANT_API_KEY` | Vector store for the support knowledge base |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` | Tracing |

Seed the database (one schema per agent; `--reset` rebuilds, `--test` targets the test database):

```bash
uv run python scripts/neon-db-creation/init_neon.py
```

The support agent expects a Qdrant collection named `troubleshooting_kb` (see `config/base.yaml`),
embedded with `BAAI/bge-small-en-v1.5`. This repo does not yet include an ingestion script for it.

The fashion agent searches a second collection, `product_catalog`, built from the Neon `product_catalog`
table (recreated on every run):

```bash
uv run python scripts/fashion/ingest_catalog.py
```

## Usage

The supervisor reaches each agent over A2A, so start the agent services first (one process each,
ports in `config/base.yaml`) in a separate terminal and leave them running:

```bash
uv run python scripts/a2a/run_agents.py
```

This starts the fashion MCP server (port 8005, so its embedding model loads once) and the order management,
customer support, shipment tracking and fashion services (ports 8001 to 8004). The push-notification receiver
on 8010 is run by the supervisor side. How it works and real session
walkthroughs: `DOCS/working-info/a2a/` (`basics/`, `streaming/`, `push-notifications/`).

Agent replies stream end to end: runners yield chunks, the A2A server sends them as progress and
artifact updates, and the supervisor forwards them to the caller. For slow tasks, if the stream breaks,
the agent pushes the result to the webhook and it is saved into the same thread.

### Interactive chat (continuous conversation)

Run without a query to chat over many turns in one session. Every message uses the same
`thread_id`, so the assistant remembers earlier turns ("what about the one before that?").
Type `exit`, `quit` or press Ctrl+C to leave.

```bash
uv run python -m hm_agent
uv run python -m hm_agent --customer-id cust002
```

### Single query

```bash
uv run python -m hm_agent "What was the last item I bought?"
```

### Resume a conversation later

The `thread_id` is printed at the start of a chat and when the process ends. Pass it back to
continue where you left off (history is loaded from the Neon checkpointer):

```bash
uv run python -m hm_agent --thread-id <thread_id>
uv run python -m hm_agent --thread-id <thread_id> "And how many items have I bought in total?"
```

| Flag | Default | Meaning |
|---|---|---|
| `query` (positional, optional) | none | One message. Omit it to start the chat loop. |
| `--customer-id` | `cust001` | Customer the conversation belongs to (`cust001`..`cust005` in the seed data). |
| `--thread-id` | new UUID | Existing thread to resume. |

If you build an API or UI on top, keep one stable `thread_id` per conversation and call
`graph.invoke` once per message; the chat loop is only needed for the CLI.

### RAG pipeline on its own

```bash
uv run python -m hm_agent.customer_support.rag "What happens if my package can't be delivered?"
```

## Runtime entrypoint

`runtime/entrypoint.py` is the supervisor's front door, built with the AgentCore SDK (it replaced the
earlier FastAPI layer). Start the agent services first, then:

```bash
RUNTIME_PORT=8090 uv run python -m hm_agent.runtime.entrypoint
curl -N -X POST localhost:8090/invocations -H 'Content-Type: application/json'   -H 'X-Amzn-Bedrock-AgentCore-Runtime-Session-Id: local-session-000000000000000001'   -d '{"message": "Where is my order ORD-1001?", "customer_id": "cust001"}'
```

| Endpoint | Purpose |
|---|---|
| `POST /invocations` | `{"message": "..."}` runs one chat turn; `{"action": "history"}` reads the thread back. Streams JSON events and always ends with `done`. |
| `GET /ping` | Health check. |

The runtime session id header is the conversation `thread_id`. With `AUTH_ENABLED=true` the customer id comes
only from the verified Auth0 bearer token (the body's `customer_id` is ignored) and a thread belongs to the
customer who created it. Port, pool size and message limit are under `runtime:` in `config/base.yaml`.

`docker compose up --build` runs the supervisor (host port 8090) and the four agents as arm64 containers;
secrets come from `.env` at run time.

## Tests

Tests use fakes for the graph, the LLM and Neon, so they make no Groq calls:

```bash
uv run pytest
```

## Configuration

Layered YAML in `config/`: `base.yaml` holds the defaults and `dev.yaml` is merged over it
(`APP_ENV=dev` is the default). Model names per role, RAG settings, history window sizes, log level
and tracing tags live there, not in code.

| Role | Model |
|---|---|
| Small | `openai/gpt-oss-20b` |
| Big | `openai/gpt-oss-120b` |

Secrets never go in the YAML files; they live in `.env`.

## Project layout

```
src/hm_agent/
  cli.py            CLI entry point (single query and chat loop)
  graph.py          Supervisor graph assembly
  turns.py          One-message turn logic shared by the CLI and the runtime entrypoint
  runtime/          AgentCore entrypoint (/invocations, /ping) and thread ownership/history helpers
  auth/             Auth0 JWT verification and a standalone login app
  core/             Shared config, settings, LLM factory, logging, db and graph state
  supervisor/       Plans each request: one or more specialist agents, with sub-queries and sticky follow-ups
  order_management/ Agent, MCP server and client (injects customer_id), A2A service
  customer_support/ Agent, RAG pipeline (rag/), A2A service
  shipment_tracking/ Dummy carrier lookup agent (streams progress, push notifications), A2A service
  fashion/          Agent, MCP server and client (product search, styling guides), A2A service
  a2a/              Generic A2A server, client, cards, stream chunks, push receiver and supervisor remote node
tests/              Unit and entrypoint tests (fakes, no live services)
config/             base.yaml, dev.yaml
scripts/            Database creation and query scripts, fashion/ingest_catalog.py, and a2a/run_agents.py (starts the agent services)
DOCS/               Data documentation and code-reading guide
```

## Notes

- Groq is rate limited: keep LLM calls few and mock the LLM in tests.
- Run and test locally before touching AWS, and keep any AWS resources minimal and pay-per-use.
- Logs use `key=value` fields and never include secrets or customer data.
