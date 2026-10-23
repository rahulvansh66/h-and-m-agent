# Backend concepts used in this API

Written for the owner of this repo, who knows the agents and A2A well but is new to backend engineering.
Each section says what the concept is, why it matters, and where it lives in this code (file and line numbers are for the
current state of the repo and will drift; search for the names). Real log output comes from
`DOCS/working-info/a2a/streaming/`. Section 22 lists concepts worth learning that the API does **not** have yet.

Contents: 1 request flow, 2 API and REST, 3 versioning, 4 validation and OpenAPI, 5 dependency injection, 6 authentication,
7 authorization and thread ownership, 8 rate limiting, 9 sync vs async, 10 threads and the event loop bridge, 11 SSE,
12 connection pool, 13 per-thread locks, 14 lifespan and shared state, 15 middleware, 16 logging and correlation ids,
17 CORS, 18 health probes, 19 webhooks, 20 timeouts and failures, 21 error handling, 22 what is missing.

---

## 1. The path of one request

```
client --HTTP--> uvicorn (server) --> FastAPI app
                                        |- middleware: request id, access log        app.py:118
                                        |- router dependencies: API key              chat.py:27
                                        |- route dependencies: customer id, rate     chat.py:50
                                        |- request body validated (Pydantic)         schemas.py
                                        |- route function: check thread ownership    deps.py:84
                                        '- run the LangGraph turn (worker thread)    service.py:185
                                              |- supervisor LLM -> routes to an agent
                                              '- A2A call (streaming) -> agent service
```

The order matters: cheap checks (key, header shape, rate limit) run **before** anything touches the database or an LLM, so
a bad or abusive request costs almost nothing. That is a general backend rule: *fail early, fail cheap*.

## 2. API, REST and HTTP basics

An **API** is a contract: "send this request, get this response". Here the client is a frontend (or curl) and the server is
`hm_agent.api`. We follow REST conventions loosely:

| Idea | In this API |
|---|---|
| **Resource** = a noun with a URL | `/v1/threads/{thread_id}/messages` is "the messages of a thread" |
| **Verb** = HTTP method says the action | `GET` reads (safe, repeatable), `POST` creates or triggers work |
| **Status code** = the outcome class | table below |
| **Headers** = metadata about the call | `X-API-Key`, `X-Customer-Id`, `X-Request-Id`, `Retry-After` |
| **Body** = JSON payload | `{"message": "...", "thread_id": "..."}` |

Status codes the API actually returns:

| Code | Meaning here | Raised in |
|---|---|---|
| 200 | OK | normal replies |
| 401 | missing or wrong API key (*who are you?*) | `deps.py:32` |
| 404 | thread does not exist **or is not yours** | `deps.py:84` |
| 422 | the request is well-formed HTTP but its content is invalid (bad `X-Customer-Id`, empty or over-long message) | `deps.py:48`, `schemas.py` |
| 429 | too many requests, with `Retry-After` | `deps.py:65` |
| 502 | the upstream (agents / LLM) failed, not the API itself | `chat.py:77` |
| 503 | `/readyz` when an agent is down | `health.py` |

Rule of thumb: **4xx = the caller can fix it, 5xx = the server (or something it depends on) is at fault.**

`GET` is *safe* and *idempotent* (calling it twice changes nothing). `POST /v1/chat` is **not** idempotent: sending the same
message twice adds two turns to the conversation. Retrying it blindly after a timeout can duplicate a message (see 22).

## 3. API versioning

The `/v1` prefix (`routes/chat.py:27`, `APIRouter(prefix="/v1", ...)`) is **URL-path versioning**. Why bother: once a
frontend depends on a response shape, changing it breaks that frontend. With a version in the URL you can ship `/v2` with a new
shape and keep `/v1` alive until clients move.

- *Non-breaking* changes can stay in v1: adding a new endpoint, adding an optional request field, adding a field to a response.
- *Breaking* changes need v2: removing or renaming a field, changing a type, changing the meaning of a status code.
- Alternatives you will see elsewhere: a header (`Accept: application/vnd.x.v2+json`) or a query param. Path versioning is the simplest and the easiest to debug (the version is visible in logs).
- `/healthz` and `/readyz` are **unversioned** on purpose: load balancers and orchestrators call them, and they must not change.
- The app also has `version="0.1.0"` (`app.py:109`); that is the *software* version shown in the docs page, a different thing from the *API* version.

Note the SSE **event names** (`token`, `progress`, `reply`, ...) are part of the contract too: renaming one is a breaking change.

## 4. Validation and the OpenAPI docs

`schemas.py` defines the request and response shapes as **Pydantic** models:

```python
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=API_MAX_MESSAGE_CHARS)
    thread_id: str | None = Field(default=None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
```

FastAPI parses the JSON, checks it against this model and answers **422** automatically if it does not fit. Benefits:
the route function only ever sees valid data; limits (2000 characters) protect the LLM quota and memory; the `pattern`
on `thread_id` blocks odd characters before they reach the database. **Never trust input, even from your own frontend.**

The same models produce the **OpenAPI** document. With the API running, open `http://127.0.0.1:8000/docs` for an interactive page
that lists every endpoint and lets you try it. `response_model=ChatResponse` also *filters* the output: only declared fields are returned.

## 5. Dependency injection (`Depends`)

A **dependency** is a small function FastAPI runs before your route and whose result it passes in:

```python
@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(enforce_rate_limit)])
async def chat(body: ChatRequest,
               customer_id: str = Depends(get_customer_id),
               services: Services = Depends(get_services)) -> ChatResponse:
```

Why it is useful: auth, validation and shared objects are written once (`deps.py`) and reused by every route; a dependency
that raises `HTTPException` stops the request before the route runs; and tests can swap dependencies. Dependencies can depend
on each other (`enforce_rate_limit` needs `get_customer_id`, which is validated first). Putting `dependencies=[Depends(require_api_key)]`
on the **router** (`chat.py:27`) protects every route in it, so a new endpoint cannot forget the key check.

## 6. Authentication: the API key

**Authentication** = proving who the caller is. Here a shared secret: the client sends `X-API-Key`, the server compares it
with `API_KEY` from the environment (`deps.py:32-45`):

```python
expected = os.environ.get(API_KEY_ENV, "")
if not expected or not hmac.compare_digest(x_api_key.encode(), expected.encode()):
    raise HTTPException(status_code=401, detail="invalid or missing API key")
```

Concepts packed in these lines:
- **Fail closed**: if the server has no key configured, *everything* is rejected, never everything allowed. `__main__.py` also refuses to start without `API_KEY`.
- **Constant-time comparison** (`hmac.compare_digest`): a normal `==` stops at the first differing character, so response time can leak how many leading characters were right. Use constant-time compare for any secret.
- **Secrets live in the environment** (`.env`, never in code or git), and are never logged.
- **Be honest about the limits**: one shared key proves "a trusted client", not "this person". `X-Customer-Id` is then *trusted* from that client. That is fine when the caller is your own backend or a demo, but if a browser calls the API directly, anyone who has the key can claim to be any customer. The production answer is a per-user token (JWT / OAuth) from which the server *derives* the customer id. See 22.

## 7. Authorization and thread ownership

**Authentication** asks "who are you?"; **authorization** asks "are you allowed to touch *this*?". A valid key does not mean
customer `cust002` may read customer `cust001`'s conversation. The check is `require_thread_access` (`deps.py:84`):

```python
owner = await run_in_threadpool(thread_owner, services.graph, thread_id)
if (owner is None and must_exist) or (owner is not None and owner != customer_id):
    raise HTTPException(status_code=404, detail="thread not found")
```

How ownership is stored: every turn's run config carries `customer_id` in `configurable`, and LangGraph copies it into the
**checkpoint metadata** (`turns.py::build_config`). `thread_owner` (`service.py:150`) reads it back from the latest checkpoint. That is why the
push-delivery code also passes `customer_id` when it writes into a thread (`app.py` `deliver`): otherwise the newest checkpoint would have no owner and
the legitimate customer would lose access.

Design choices worth knowing:
- **404, not 403.** 403 would confirm "that thread exists but is not yours", which leaks information. Answering 404 for both "missing" and "not yours" reveals nothing (this is *IDOR* prevention: Insecure Direct Object Reference, the classic bug of trusting an id from the URL).
- **Fail closed on legacy data**: a stored thread with no recorded owner returns `""`, which equals no customer, so it is denied (`service.py:150-166`).
- Unguessable ids (uuid4) are a second layer, **not** a substitute for the check.

## 8. Rate limiting

Protects the Groq free tier (and the server) from a client that loops. `RateLimiter` (`service.py:51`) is a **sliding window**:
it remembers the timestamps of the last calls per customer, drops those older than 60 s, and refuses when the count reaches the limit.
The refusal is `429 Too Many Requests` with a `Retry-After` header telling the client how long to wait (a well-behaved client sleeps that long).

Limits of this version: it lives **in the process's memory**, so a restart resets it, and with several worker processes each has its own counter
(the real limit becomes limit × workers). Production setups use Redis or the gateway (API Gateway, nginx) for this. It is configured in
`config/base.yaml` (`api.rate_limit_per_min`).

## 9. Sync vs async, and why it matters here

A web server handles many connections with very few threads. **`async def`** routes run on one **event loop**; when a route says `await`
(waiting for the network), the loop serves other requests meanwhile. The rule that follows: **never run slow *blocking* code on the event
loop.** One blocking call (a database query, an LLM call, `time.sleep`) freezes *every* request on that loop.

Our situation: LangGraph graphs, `psycopg` and the Groq client used here are **blocking (sync)**. So:

| Code | Style | How it avoids blocking the loop |
|---|---|---|
| Routes (`chat.py`) | `async def` | they only `await` things |
| `thread_owner`, `history_messages` (DB reads) | sync | called through `run_in_threadpool(...)` (`deps.py`, `chat.py`) which runs them on a worker thread |
| The whole graph turn | sync | runs in `asyncio.to_thread(...)` (`service.py:185`) |
| A2A executor on the agent | async | its runner is blocking, so also `asyncio.to_thread` (`a2a/server.py:192`) |

If you ever see the API "hang for everyone while one request is slow", the first suspect is a blocking call on the event loop.
Note the mixture is deliberate: rewriting the agents to be fully async would be a much bigger change.

## 10. Bridging a worker thread to the event loop

SSE needs the turn's events *while it is still running*, but the turn runs on a worker thread and the response is written by the loop.
The bridge is a queue (`service.py:185-215`, same pattern in `a2a/server.py:180-215`):

```python
outbox = asyncio.Queue()
def produce():                                   # runs on a worker thread
    for event in run_turn_events(...):
        loop.call_soon_threadsafe(outbox.put_nowait, event)   # thread-safe handoff
    loop.call_soon_threadsafe(outbox.put_nowait, None)        # end marker
worker = asyncio.create_task(asyncio.to_thread(produce))
while (event := await outbox.get()) is not None:  # runs on the loop
    yield event
```

Why `call_soon_threadsafe`: asyncio objects are **not thread-safe**; a worker thread must not touch the queue directly, it asks the loop
to do it. The `None` is a **sentinel** (end marker). An exception inside `produce` is caught and sent as an `error` event, so the stream ends cleanly
instead of hanging forever. This producer/consumer pattern is worth recognising; it shows up everywhere.

## 11. Server-Sent Events (SSE)

How can a server send data *as it happens*? Three options:

| | How | Good for |
|---|---|---|
| Polling | client asks every N seconds | simple, wasteful, laggy |
| **SSE** | one long HTTP response, server writes events, one-way server→client | token streams, progress, notifications |
| WebSocket | upgraded two-way connection | chat apps needing client→server messages mid-stream, games |

SSE was chosen because the data only flows server→client, it is plain HTTP (works with auth headers, proxies, `curl`), and browsers reconnect automatically.
On the wire it is text: lines `event: <name>` and `data: <json>`, a blank line ends the event. A real response from `POST /v1/chat/stream`:

```
event: thread           data: {"type": "thread", "thread_id": "a7182d15-..."}
event: agent_selected   data: {"type": "agent_selected", "agent": "order_management"}
event: progress         data: {"type": "progress", "agent": "order_management", "text": "Looked up your order data"}
event: token            data: {"type": "token", "agent": "order_management", "text": "It was purchased online."}
event: reply            data: {"type": "reply", ...}
event: done             data: {"type": "done"}
```

`sse_frame` (`service.py:218`) builds each frame; `EventSourceResponse` (library `sse-starlette`) sends the right headers
(`Content-Type: text/event-stream`, `Cache-Control: no-cache`, `X-Accel-Buffering: no`) and a periodic ping.

Practical points: (a) **status codes are decided before the stream starts** (the 200 header is sent first), so a failure mid-stream must be reported as an *event*
(`error` then `done`), not an HTTP status; that is why auth and ownership are checked *before* returning `EventSourceResponse`. (b) **Proxies may buffer** the stream and defeat the purpose; check every hop (nginx, ALB, CDN). (c) Idle connections get closed by timeouts, hence heartbeats/pings. (d) `EventSource` in browsers can only do GET and cannot set headers, so for POST + `X-API-Key` the frontend uses `fetch` and reads the stream manually. (e) The same technique, one line per event, is how the A2A agents talk to the supervisor; see `DOCS/working-info/a2a/streaming/a2a-wire.jsonl` lines tagged `S->C SSE`:

```
{"result": {"artifactUpdate": {... "artifact": {"artifactId": "reply", "parts": [{"text": " was a **Berry Tee**."}]},
            "append": true, "lastChunk": true}}, "id": "5d736249-...", "jsonrpc": "2.0"}
```

## 12. Connection pool

Opening a database connection is slow (network handshake, TLS, auth: hundreds of ms to Neon) and the server limits how many can be open at once.
A **connection pool** opens a few connections once and **lends** them out: borrow, use, give back. `neon_checkpointer_pool` (`core/db.py:75`):

```python
ConnectionPool(dsn, min_size=1, max_size=max_size,
               kwargs={"autocommit": True, "row_factory": dict_row, "options": f"-c search_path={CHECKPOINT_SCHEMA}"})
```

Why it is here: the CLI used **one** shared connection (`neon_checkpointer`). That is unsafe with concurrent requests, since two threads would interleave on the same connection.
The pool gives each checkpoint operation its own. `max_size` (config `api.pool_max_size`, default 4) is a **budget**: the Neon free tier allows few connections, so the
pool caps usage; a 5th simultaneous operation waits for a free connection instead of opening a 5th. Remember the total across *all* processes counts (API + any other service).

## 13. Per-thread locks (race conditions)

A **race condition** is a bug that depends on timing. Here, two writers on the same conversation:

```
turn running:   reads state (4 msgs) ........... writes state (4 msgs + reply)
push arrives:           reads state (4 msgs) -> writes 4 msgs + update          <- one of the two writes is lost
```

Each checkpoint write assumes it started from the latest state. If two overlap, the later one overwrites the earlier one's message.
`ThreadLocks` (`service.py:26`) hands out one `threading.Lock` per `thread_id`; the turn (`run_turn_events`, `service.py:169`) and the push delivery (`app.py` `deliver`)
both do `with services.locks.get(thread_id): ...`. Different threads (conversations) still run in parallel; only the *same* thread is serialised.
This is **fine-grained locking**: lock the smallest thing that must not be shared, not the whole app.

Consequences to know: a second message on a thread while the first is still running **waits** (up to the 20 s of a shipment lookup); a lock is held by a worker thread, so it must never be held across a slow `await` on the loop; and
it only protects **one process**. With two API instances behind a load balancer the locks do not see each other; the fix would be a database lock (Postgres advisory lock) or optimistic concurrency (checkpoint version check).

## 14. Lifespan and shared state

Some objects must be created **once** at startup and closed at shutdown: the graph, the pool, the push receiver. FastAPI's **lifespan** context manager does this (`app.py:77`):

```python
with neon_checkpointer_pool(API_POOL_MAX_SIZE) as checkpointer:
    app.state.services = Services(graph=build_graph(checkpointer), ...)
    receiver.start(); set_receiver(receiver)
    try: yield                       # <- the app serves requests here
    finally: receiver.stop(); get_client().flush()
```

Building the graph per request would be slow and wasteful; the compiled graph is read-only and safe to share. `app.state` is where the shared `Services` bundle lives and `get_services` (a dependency) hands it to routes.
Because `create_app(services=...)` accepts ready-made services, tests inject a fake graph and need no database (`tests/test_api.py`). **Designing for testability** (injecting dependencies instead of creating them inside) is a habit worth building.

## 15. Middleware

Middleware wraps *every* request, before and after the route. Ours (`app.py:118`) creates a request id, times the call, adds the `X-Request-Id` response header and writes one log line.
Use middleware for cross-cutting work (logging, CORS, compression, metrics). Not for business rules. Order matters: it runs outside the dependencies, so even a 401 or 429 gets logged and gets an id.

## 16. Logging and correlation ids

One log line per request, `key=value` style so a log platform (CloudWatch) can query it:

```
component=api request_id=3f9a1c2d4e5b method=POST path=/v1/chat status=200 duration_s=7.94
```

Rules applied (from `CLAUDE.md`): never log message bodies, API keys or tokens; log durations and counts.
A **correlation id** lets you follow one request across components: `request_id` for the API call, `thread_id` for the conversation, the A2A `taskId` for an agent run
(visible in `agent-order_management.log`: `task_id=6f48537e... context_id=0dc2fe6a...`) and the same `thread_id` as the Langfuse session. When something fails, you grep one id and see the whole story.

## 17. CORS

Browsers block a web page at `https://shop.example` from calling `http://api.example` unless the API sends **CORS** headers saying that origin is allowed.
It is a *browser* rule; curl ignores it. `app.py` adds `CORSMiddleware` only when `api.cors_origins` is non-empty, and allows only the methods and headers the API uses.
Never use `"*"` together with credentials. CORS is **not security for your API** (anyone can call it without a browser); it only protects users' browsers.

## 18. Health and readiness probes

`/healthz` ("the process is up") and `/readyz` ("I can actually serve": all three agent cards answer). Orchestrators and load balancers use them:
*liveness* failing means restart the container; *readiness* failing means stop sending traffic but do not restart. They are unauthenticated and return no customer data.
`readyz` checks dependencies **in parallel** with short timeouts (`asyncio.gather`, 2 s), so one dead agent cannot make the probe itself hang.

## 19. Webhooks and push notifications

A **webhook** is the server calling *you*: you give a URL, it POSTs events there. The push receiver (`a2a/push_receiver.py`) is a tiny HTTP server inside the API process.
Things that make webhooks tricky, all present here:
- **Authenticity**: anyone can POST to a public URL, so the call carries a secret token header checked with `compare_digest` (`push_receiver.py:205`).
- **At-most-once, no retry**: the agent's sender does not retry a failed POST. In the recorded session one notification vanished with `httpx.ReadError` (`agent-shipment_tracking.log`) because uvicorn's default 5 s **keep-alive** closed an idle pooled connection just as the agent reused it. Fix: `KEEP_ALIVE_S = 75` (`push_receiver.py:41`). Lesson: design so that losing one notification is harmless (we re-read the final state with `GetTask`).
- **Idempotency / duplicates**: the same event can arrive twice, or arrive on two channels (stream and webhook). `_claimed` (`push_receiver.py:113`) records tasks already handled so the answer is delivered once.
- **Ordering and races**: a terminal notification can arrive *before* the code that registers the task; hence the `_finished_early` parking set.
- **Respond fast**: a webhook handler should acknowledge quickly and do slow work elsewhere (`asyncio.to_thread(self._finish, ...)`).

## 20. Timeouts, cancellation and failure

Every network call needs a **timeout**; without one a stuck dependency hangs your request forever and eats a worker. Ours: A2A calls `a2a.request_timeout_s` (90 s), readiness probes 2 s.
For a stream the timeout means the longest *silence* between events, not the total time, so the shipment agent sends progress every 5 s (it doubles as a **heartbeat**).
Related ideas: **graceful degradation** (a dead agent returns an apology message instead of crashing the conversation, `remote_node.py`); **fallbacks** (stream broke, so push delivers); **backpressure** (the pool and the rate limiter say "wait" instead of letting load pile up).
Not yet handled: if the client disconnects mid-stream, the server-side turn keeps running to the end (see 22).

## 21. Error handling

- **Never leak internals.** An exception text can reveal table names, file paths or prompts. Routes return generic messages (`"the assistant could not produce a reply"`) and the full traceback goes only to the log (`logger.exception`). A test checks this (`test_graph_failure_is_a_502_without_details`).
- **Use the right class of status**: 4xx for the caller's mistakes, 502 for a failing upstream, 500 for our bug.
- **Errors inside a stream** are events (`error`, then `done`), since the HTTP status is already sent.
- **Catch narrowly where you can handle, broadly at the boundary.** The producer thread catches everything because an uncaught exception there would leave the consumer waiting forever.

## 22. Concepts worth learning that this API does not have yet

| Concept | Why you will want it |
|---|---|
| **Per-user auth (JWT / OAuth)** | derive `customer_id` from a signed token instead of trusting a header; short-lived tokens, revocation |
| **Idempotency keys** | let clients safely retry `POST /v1/chat` after a timeout without duplicating a turn (`Idempotency-Key` header) |
| **Pagination** | `GET .../messages` returns the whole history; long threads need `limit`/`cursor` |
| **Shared state across instances** | rate limiter, thread locks, SSE hub and task store are per-process; scaling to 2+ instances needs Redis / DB locks / sticky sessions |
| **Cancellation on disconnect** | stop the LLM/agent work when the client goes away (saves Groq quota); not tested here |
| **Graceful shutdown** | let in-flight streams finish on deploy (`SIGTERM` handling, drain period) |
| **Request size and concurrency limits** | cap body size at the server/proxy, limit simultaneous streams per customer |
| **Structured error model** | one consistent error JSON (`code`, `detail`, `request_id`) across all failures |
| **Metrics and tracing** | counters, latency histograms (Prometheus / CloudWatch metrics); distributed traces beyond Langfuse |
| **TLS / reverse proxy** | the API speaks plain HTTP; production terminates HTTPS in front (ALB, nginx) and must not buffer SSE |
| **Secrets management** | `.env` is fine locally; deployed, use AWS Secrets Manager / SSM, and rotate keys |
| **Containerisation and deployment** | Dockerfile, health-checked rollout, min/max instances; remember CLAUDE.md: local first, AWS minimal and pay-per-use |
| **Contract and load tests** | schema tests against the OpenAPI doc; a small load test to see where the pool and rate limit bite |
| **Webhook hardening** | validate callback URLs you accept (SSRF), HTTPS, retries with backoff, replay protection |
| **Database migrations** | schema changes tracked as versioned scripts (currently one init script) |

## Code map

| Concept | File |
|---|---|
| App, lifespan, middleware, CORS, push delivery | `src/hm_agent/api/app.py` |
| Entry point (loads env, refuses without key) | `src/hm_agent/api/__main__.py` |
| Routes: chat, stream, history, events | `src/hm_agent/api/routes/chat.py` |
| Health and readiness | `src/hm_agent/api/routes/health.py` |
| Auth, customer id, rate limit, thread access | `src/hm_agent/api/deps.py` |
| Locks, limiter, SSE hub, thread bridge, ownership | `src/hm_agent/api/service.py` |
| Request/response models | `src/hm_agent/api/schemas.py` |
| Connection pool | `src/hm_agent/core/db.py` (`neon_checkpointer_pool`) |
| Turn as events, run config | `src/hm_agent/turns.py` |
| Tests with a fake graph | `tests/test_api.py` |
