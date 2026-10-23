# FastAPI layer + A2A streaming (with push) + documented live run

## Context
The assistant is reachable only through the CLI. Goals: (1) a FastAPI layer over the supervisor graph,
(2) real streaming end to end (agent -> A2A -> supervisor -> API SSE), (3) a real recorded session that
documents how streaming works, including what changes when an agent has streaming AND push on.

Current state (uncommitted working tree on `feature/fastapi`): three A2A agents exist
(`order_management` :8001, `customer_support` :8002, `shipment_tracking` :8003 with push on), a push
receiver (`a2a/push_receiver.py`, :8010) and CLI delivery of pushed results. Everything is `streaming=False`:
- `a2a/cards.py:93` `AgentCapabilities(streaming=False, ...)`
- `a2a/client.py:105,190` `ClientConfig(streaming=False)`
- `a2a/server.py` `RunnerExecutor.execute` runs `graph.invoke` in `asyncio.to_thread` and publishes ONE artifact at the end
- `a2a/remote_node.py` sync node returns one `AIMessage`
Decisions: API key + `X-Customer-Id` auth; chat + SSE streaming; shipment_tracking = **stream live, push as durable fallback**.

## Part A - Streaming through A2A
1. **Runner contract** (`a2a/server.py`): `Runner` becomes a generator yielding `Chunk(kind: "progress"|"answer", text)`. `graph_runner` uses `graph.stream(stream_mode="messages")` and yields token text of the final answering node (filter by node / last AI message, not tool or routing calls). Non-LLM runners yield one chunk.
2. **Executor**: run the generator in a thread, bridge to the loop (`asyncio.Queue` + `call_soon_threadsafe`). `progress` -> `updater.update_status(WORKING, message)`; `answer` -> `updater.add_artifact([...], artifact_id="reply", append=True, last_chunk=False)`; end -> final `last_chunk=True` then `complete()`. Failure -> `failed()` as today. Verify exact a2a-sdk 1.2.2 signatures (`TaskUpdater.add_artifact` append/last_chunk) before coding.
3. **Cards**: `streaming=True` for all three agents; shipment keeps `push_notifications=True`; bump `CARD_VERSION`.
4. **Client** (`a2a/client.py`): `ClientConfig(streaming=card.capabilities.streaming)`; new `stream_remote_agent(...)` generator yielding events (progress, answer delta, terminal state) from `send_message` responses; for push-capable agents also send `task_push_notification_config` (webhook) in the same request, but NOT `return_immediately`, so the stream stays open. Keep `call_remote_agent` as the non-streaming path; `fetch_task` unchanged.
5. **Remote node** (`a2a/remote_node.py`): consume the stream; forward each chunk with `get_stream_writer()` (custom stream mode); return the full reply as one `AIMessage` (checkpointing unchanged). If the stream breaks before a terminal state and the task id is known and the agent supports push: register `PendingTask` and return `PENDING_REPLY` (existing push path). On a clean stream, the later webhook POSTs for that task are ignored; fix `push_receiver._handle` so a terminal notification for a task that finished via the stream does not sit in `_finished_early` forever (add a `_streamed_done` set / mark claimed).
6. **Shipment progress**: `check_carrier_status` yields progress steps ("Contacting carrier", "Waiting for carrier response", ...) at intervals during the configured delay, so the stream shows real `working` updates. Remains LLM-free.
7. **Customer support**: make `respond` node stream from the LLM (`llm.stream` / default messages mode picks it up). Order agent (ReAct) streams its final answer tokens. Keep LLM calls few (Groq free tier).
8. **CLI**: print streamed tokens as they arrive (custom + messages stream) while keeping the push delivery path.

## Part B - FastAPI layer (`src/hm_agent/api/`)
Files: `app.py` (`create_app`, lifespan builds a pooled `PostgresSaver` + `build_graph` + starts `PushReceiver` and `set_receiver`), `__main__.py` (uvicorn), `schemas.py`, `deps.py`, `service.py` (reuse CLI's `_build_config` logic; Langfuse handler, flush), `routes/chat.py`, `routes/health.py`.
- Auth: `X-API-Key` vs `API_KEY` env (`hmac.compare_digest`); `X-Customer-Id` validated against a pattern; 401/403/422 handlers.
- `POST /v1/chat` -> `{thread_id, reply, agent}`; new uuid4 thread if omitted.
- `POST /v1/chat/stream` -> SSE (`sse-starlette` already installed): events `agent_selected`, `progress`, `token`, `reply`, `done`, `error`. Source: `graph.stream(stream_mode=["updates","custom","messages"])` in a thread bridged via asyncio queue.
- `GET /v1/threads/{thread_id}/messages` -> history via `graph.get_state`; 404 when no state or thread belongs to another customer (record `customer_id` in thread metadata on first turn).
- Pushed results (shipment fallback): receiver's `deliver` callback saves into the thread (as CLI does, under a per-thread lock) and publishes to an in-memory per-thread queue; `GET /v1/threads/{id}/events` (SSE) lets a client receive it.
- `GET /healthz`, `GET /readyz` (agent cards reachable).
- Config: `api:` section in `config/base.yaml` + constants in `core/settings.py`; `API_KEY` in `.env.example`. Middleware: request id, key=value request logs (no message bodies/PII). Pool max 4 for Neon. Per-customer in-memory rate limit (~10/min) to protect Groq quota. Pydantic message length cap.
- Deps: add `fastapi`, `psycopg-pool`, `boto3` (already imported undeclared); dev: `pytest`, `pytest-asyncio`.

## Part C - Live run + documentation (`DOCS/working-info/a2a/streaming/`, folder exists, empty)
Run only after Parts A/B pass tests locally. No AWS involved. Real Groq/Neon use limited to ~3 turns (supervisor small-model call per turn + order agent calls): within the free-tier guidance.
Session (one thread, customer `cust001`, `A2A_WIRE_LOG` set, agents started with `--log-dir`):
1. "What was the last item I bought?" (order_management, no history -> history data part empty)
2. "And how many purchases have I made?" (order_management, history now carries turn 1, shows the `history` data part, same `contextId`)
3. "Where is my order ORD-1007?" (shipment_tracking, streaming + push both on)
Saved files: `commands.md` (exact commands), `launcher.log`, `agent-*.log`, `session-console.log`, `a2a-wire.jsonl` (incl. SSE frames and `PUSH A->S` posts), `agent-cards.txt`, `thread-state.txt`.
`streaming-explained.md` (written for the repo owner, same format as `push-notifications-explained.md`): step-by-step walk of the real session; SSE wire anatomy (status-update vs artifact-update, `append`, `lastChunk`, terminal event); turn 2 vs turn 1 showing history in the request; query 3 side by side: stream events vs webhook POSTs for the same task, why the client keeps the stream and treats push as fallback, dedup, what happens if the stream drops; comparison table (blocking / streaming / push / both); practical notes (proxies buffering SSE, timeouts, thread->loop bridging, idempotent webhook handling, checkpoint races, token auth, Groq reasoning tokens, what not to stream).
Existing logs in sibling folders are not overwritten.

## Verification
- Unit tests (new `tests/`, `db` fixture per CLAUDE.md, LLM mocked): runner chunking -> artifact append/last_chunk order; client stream parsing; remote node stream break -> pending fallback; push_receiver dedup; API auth, chat, SSE event order, history ownership 404, readyz with agent down.
- Manual local: start agents (`scripts/a2a/run_agents.py`), `uv run python -m hm_agent.api`, curl `/v1/chat` and `/v1/chat/stream` with `-N`; confirm tokens arrive incrementally, thread reuse keeps context, Langfuse session = thread_id.
- Part C run itself is the end-to-end check: wire log must show SSE frames for all three turns and both stream + webhook events for turn 3.
