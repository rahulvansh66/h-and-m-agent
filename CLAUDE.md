# H&M Assistant

## Local-only folders

`.tmp/` and `ref/` are local scratch and reference folders, not part of the project. Read from them only
when explicitly asked or referenced, and never mention, cite, or document them as a source anywhere in the codebase
(code, docstrings, comments, docs, configs, or notebooks).

## Database

One Neon Postgres database (`neondb`, connection string in `NEON_POSTGRES_CONNECTION_STRING`
in `.env`), one schema per agent. Set up and seeded by `scripts/neon-db-creation/init_neon.py` (`--reset` rebuilds).

| Schema | Tables |
|---|---|
| `order_management` | `customers`, `orders`, `order_items` |
| `product_recommendation` | `product_catalog`, `purchase_history` |
| `personalization` | `personalization` |

- Seeded from the H&M mini dataset (`data/H_and_M_mini_dataset/`); see `DOCS/data/user-item/schema-overview.md`
  for the mapping (order status, shipping and stock are derived or absent, not invented).
- Shared IDs across schemas: customers `cust001`..`005`, products `prod001`..`010`.
- Foreign keys only within a schema, never across schemas, so agents stay independent.

## Testing and the test database

- **Separate test database.** Tests never touch `neondb`. A second database, `neondb_test`
  (same Neon endpoint, identical schemas and seed data), is addressed by
  `NEON_POSTGRES_TEST_CONNECTION_STRING` in `.env`. Create or rebuild it with
  `uv run python scripts/neon-db-creation/init_neon.py --test` (add `--reset` to restore
  the seed data after tests that write).
- **Integration tests use the `db` fixture**, which points at the test database and refuses
  to run if the test DSN equals the main DSN. Never point tests at
  `NEON_POSTGRES_CONNECTION_STRING`.

## LLM model selection (Groq)

Pick the cheapest model that is reliable for each role; configure model names per
role in `config/` (not hardcoded in graph code) so they can be swapped and compared.

| Role | Model |
|---|---|
| Small model | `openai/gpt-oss-20b` |
| Big model | `openai/gpt-oss-120b` |

## Cost and usage limits

- **Groq is the free tier and rate-limited.** Do not use it heavily: keep LLM calls few, mock the
  LLM in tests, avoid large eval loops, and default to the cheapest model that works.
- **AWS: minimal configuration.** Choose the smallest/cheapest sizes and settings (memory,
  instance class, retention, capacity) for every resource. The goal is to keep AWS cost from
  shooting up, so avoid always-on or provisioned resources where a pay-per-use option exists.
- **Flag costly code.** If any code or infrastructure I'm adding or touching is likely to be
  expensive (large model calls, loops of LLM calls, always-on AWS resources, high retention),
  alert me explicitly before proceeding, with the reason and a cheaper alternative.
- **Local first.** Use AWS only after local testing is done and passing; never deploy or
  create AWS resources to iterate or debug.

## Code style

Code should be clean, readable, structured and modular. Comments follow the rules in "Comments" below.

### Comments (non-docstring)

Docstrings say what a unit does; comments say why the code is the way it is.

1. **Explain why, not what.** Never restate the code. If a comment is needed to
   follow *what* happens, rename or extract a function first.
2. **Comment these, always:** concurrency and locking, security and ownership
   checks, ordering that matters, workarounds, deliberate broad `except`, and
   non-obvious data shapes (reducers, protocol events, SQL tricks).
3. **Do not comment** obvious code, simple assignments, getters, or plain loops.
4. **Placement.** Put the comment on its own line(s) directly above the code it
   describes, with no blank line between them. Use a trailing `# ...` only for a
   very short fact (about 6 words or fewer): a unit, a sentinel, or a tag like `# noqa`.
5. **Length.** At most 3 lines per comment. Anything longer goes in the function or
   class docstring, or into a well-named helper.
6. **Spacing.** Leave a blank line before every commented block so each block reads
   as a paragraph; do not stack commented blocks back to back.
7. **Constants.** Say the unit or meaning of a magic value: `TIMEOUT_S = 600  # seconds`.
8. **Markers.** Use `# NOTE:` for a trap the next editor must know, `# WARNING:` for
   something that breaks if changed, and `# TODO(name): ...` with an owner. No bare TODOs.
9. **Keep them in sync.** Update or delete a comment in the same edit that changes its
   code. A stale comment is worse than none.
10. **Never** leave commented-out code, change-history notes ("added for X", "old
    logic"), or secrets, PII and local-only paths in comments.
11. **Other file types.** SQL: a one-line header per non-trivial query (purpose, tables).
    YAML/Terraform/Dockerfile: comment each non-default setting with its reason
    (especially cost- or security-driven ones). Tests: name the behaviour in the test name
    and comment only the non-obvious setup.

### Docstrings

1. **Style.** Use Google-style docstrings consistently across the repo.
2. **Content.** A one-line summary of what it does (and *why*, when not obvious).
   For anything non-trivial, add a short description paragraph.
3. **Every argument is documented.** List all parameters under `Args:` with name,
   type, and meaning. Do not leave a docstring that skips or only partially covers
   the arguments; if you touch a function with a partial docstring, complete it.
4. **Also document** `Returns:` (type + meaning), `Raises:` (exceptions the caller
   should expect), and `Yields:` for generators. Omit a section only when it truly
   does not apply (e.g. no return value).
5. **Keep it in sync.** When you change a signature, update the docstring in the
   same edit.
6. **Type hints.** Annotate every function and method: all parameters and an explicit
   return type (`-> None` when nothing is returned). Graph builders return
   `CompiledStateGraph`. Never leave a return annotation off.
7. **Module docstring** states the file's purpose and its role in the system
   (e.g. which agent, graph, tool, or pipeline stage it belongs to)


### Logging conventions (follow always)

Application code (agents, tools, ingestion, API services) may run in containers
or serverless environments where logs are shipped to a central platform such as
CloudWatch. Write logs so they are usable once they land there.

1. **One logger per module**, never configure the root logger from library code:
   `logger = logging.getLogger(__name__)`. Only the entry point (app startup,
   script `main`, worker handler) calls `logging.basicConfig(...)` or sets up
   handlers.
2. **Log levels matter.** `DEBUG` for verbose diagnostics, `INFO` for normal
   progress (node start/end, tool calls, record counts), `WARNING` for
   recoverable issues (retries, fallbacks, empty retrieval results), `ERROR` for
   failures. Use `logger.exception(...)` inside `except` blocks so the traceback
   is captured. Never use `ERROR` for expected control flow.
3. **Structure messages for querying.** Use consistent key=value fields, e.g.
   `logger.info("node=retrieve status=complete docs=%d duration_s=%.2f", n, dt)`
   instead of free-form prose. Use `%s`-style lazy formatting, not f-strings, in
   log calls.
4. **Include correlation identifiers** (thread/session ID, run ID, trace ID,
   agent or node name, git commit hash) so a single conversation or run can be
   traced across agents and tool calls.
5. **Log GenAI-specific signals as aggregates**: model name, prompt/response
   token counts, latency, retry count, tool name, tool success/failure, number of
   retrieved chunks and top similarity score. Do not log full prompts,
   completions, or retrieved documents at `INFO`; if needed for debugging, log
   them at `DEBUG` only, truncated and with sensitive data masked.
6. **Never log secrets or PII.** No API keys, tokens, database credentials, full
   card numbers, or customer PII (names, emails, addresses, order details tied to
   a person) in log messages; mask or omit them. This includes user messages and
   tool arguments that may contain such data.
7. **Respect log volume and retention.** Don't log full DataFrames, embeddings,
   large arrays, or per-item output in tight loops; log counts and samples.

### Jupyter notebooks

- Structure the notebook into clear sections and subsections using markdown
  headings (`#`, `##`, `###`), so it reads top-to-bottom as a document, not a
  scratchpad.
- Every section/subsection should be well documented: a short markdown
  description of what it covers and why it's there.
- Precede every code cell with a markdown cell explaining the purpose of that
  specific cell: what it does and why it's needed.
- Code inside cells follows the same rules as `.py` files above (docstrings for
  any function/class defined in the notebook, type hints, `logging` over
  `print`, sparse in-code comments for block separation).
