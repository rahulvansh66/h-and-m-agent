The FastAPI layer, end-to-end streaming through A2A (with push as the fallback for the shipment agent) and the recorded session are done. All 26 tests pass and nothing is committed. They use fakes for the graph, the LLM and Neon, so they don't exercise the real Neon, Groq or agents. I checked that path with the live runs below.

## Why streaming was blocked, and the fix
The code chose blocking in three places:
- **Client and cards:** both had `streaming=False`.
- **Executor:** it ran `graph.invoke` and published one artifact only at the end.
- **Remote node:** it returned a single message, so LangGraph never saw partial output.

Runners now yield chunks, which the executor sends as progress status updates and appended artifact chunks. The client streams them, and `remote_node` forwards them to LangGraph's `custom` stream. The graph still stores one `AIMessage`, so history and checkpoints are unchanged.

## What was built
- **Streaming:** a `Chunk` contract in [chunks.py](src/hm_agent/a2a/chunks.py), plus changes to [server.py](src/hm_agent/a2a/server.py), [client.py](src/hm_agent/a2a/client.py), [remote_node.py](src/hm_agent/a2a/remote_node.py) and [wire_log.py](src/hm_agent/a2a/wire_log.py).
  - **Shipment agent:** it now sends progress lines during the 20 s carrier wait.
  - **Order agent:** it streams its tokens.
  - **Customer support:** it sends its validated answer in one piece, because that answer is checked for citations and can be regenerated.
- **API** in [api/](src/hm_agent/api/), started with `uv run python -m hm_agent.api`. Auth is `X-API-Key` plus `X-Customer-Id`.
  - **Endpoints:** `/v1/chat`, `/v1/chat/stream` (SSE), `/v1/threads/{id}/messages`, `/v1/threads/{id}/events`, `/healthz` and `/readyz`.
  - **Safeguards:** a connection pool, per-thread locks, thread ownership (another customer gets 404) and a rate limit.
- **Shared turn logic** in [turns.py](src/hm_agent/turns.py), used by both the CLI and the API.

## Live checks
- **Recorded session:** one thread with the two order questions, then ORD-1007. It is saved in `DOCS/working-info/a2a/streaming/`, with `streaming-explained.md` and `commands.md` alongside the logs.
- **API:** I ran it against Neon and the real agents (one chat and one streamed follow-up). The follow-up kept the thread's context, the history read back correctly, and the second customer's request got 404.

## Findings
- **Dropped push notification:** one push was lost with `httpx.ReadError`. The receiver's uvicorn closes idle connections after 5 s, which matches the 5 s spacing of the progress events, and the SDK doesn't retry. I raised `timeout_keep_alive` to 75 s and a rerun delivered 8 pushes for 8 stream events. I kept the failed run in the docs as an example that push is best-effort.
- **Appended chunks:** they arrive as separate artifact parts, and `task_reply` was joining them with newlines. It now joins them with `""`.
- **Short answers:** token streaming barely helps. In turn 1 the answer arrived in two pieces 0.3 s apart, after 6.5 s of LLM and tool time. The progress lines and the 20 s shipment lookup are where streaming helps.
- **Stream-first on shipment:** the turn now blocks for about 20 s, with live progress, instead of the old acknowledgement in under 1 s. The push fallback only takes over if the stream breaks.

## Not covered
- **Push fallback:** the live push fallback (a stream breaking mid-task) is only covered by a unit test.
- **Client disconnect:** I didn't test what the SDK does to a running task when the client disconnects.
- **Pushed results:** `/v1/threads/{id}/events` for pushed results isn't covered by a live check either.
- **Live run:** it used about six Groq calls.
- **Added files:** `uv add` added fastapi, psycopg-pool, sse-starlette, pytest and pytest-asyncio.
- **Docs folder:** `DOCS/working-info/a2a/streaming/` also holds a `plan.md` I didn't create, which I left in place.