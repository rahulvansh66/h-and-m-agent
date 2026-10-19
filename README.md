# H&M Fashion Assistant

A supervisor multi-agent assistant for H&M customers, built with LangGraph. A supervisor reads each
message and routes it to one of two sub-agents, or politely declines if it is out of scope.

```
START -> supervisor -> order_management  -> END   (own orders and purchases, via MCP SQL tools)
                    -> customer_support  -> END   (agentic RAG over the support knowledge base)
                    -> out_of_scope      -> END
```

| Component | What it does |
|---|---|
| `supervisor` | Classifies the latest message (with a short history window) and picks the agent. |
| `order_management` | ReAct agent over read-only tools served by an MCP server on the Neon `order_management` schema. The `customer_id` is injected from the run config, so the model never chooses whose data is read. |
| `customer_support` | Condense query -> retrieve from Qdrant -> answer with citations (retries once if no citation). |
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

## Usage

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
uv run python -m hm_agent.rag "What happens if my package can't be delivered?"
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
  agents/           supervisor, order_management, customer_support
  rag/              Retriever and agentic RAG pipeline
  mcp_servers/      MCP server with read-only order SQL tools
  mcp_client.py     Loads MCP tools and injects customer_id
  db.py             Neon helpers and the Postgres checkpointer
config/             base.yaml, dev.yaml
scripts/            Database creation and query scripts
DOCS/               Data documentation and code-reading guide
```

## Notes

- Groq is rate limited: keep LLM calls few and mock the LLM in tests.
- Run and test locally before touching AWS, and keep any AWS resources minimal and pay-per-use.
- Logs use `key=value` fields and never include secrets or customer data.
