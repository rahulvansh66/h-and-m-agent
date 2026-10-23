# A2A explained from a real session

Written for the owner of this repo, who already knows MCP. One real chat (two questions, one thread)
is walked through step by step. Every claim points at a saved file in this folder.

| File | What it is |
|---|---|
| `commands.md` | Exact commands used to produce everything below. |
| `session-console.log` | The chat terminal: supervisor and A2A client logs plus the answers. |
| `a2a-wire.jsonl` | The HTTP bodies between supervisor and agents, one JSON object per line (8 lines). |
| `agent-order_management.log`, `agent-customer_support.log` | Each agent service's console output. |
| `agent-cards.txt` | The Agent Cards as served (captured before `out_of_scope` became a local node, so it still lists a third card). |
| `langfuse-observations.txt` | Langfuse traces and spans of this session. |
| `launcher.log` | Process start and ready lines. |

Session: customer `cust001`, thread (and A2A `contextId`, and Langfuse session) `d4874bd8-83d5-4a82-91e1-8353dbe167d2`.

| Turn | You asked | Routed to | Answer |
|---|---|---|---|
| 1 | What was the last item I bought? | `order_management` (:8001) | Your most recent purchase was a Berry Tee. |
| 2 | The zipper on my jacket arrived broken, what can I do? | `customer_support` (:8002) | Cited return/replace steps from the help articles. |

`out_of_scope` was a third A2A service (:8003) when this session was captured. It has since been removed: it only returned a
fixed reply, so it is now a local node in the supervisor graph (no A2A call).

---

## 1. What A2A is, and how it differs from MCP

**A2A (Agent2Agent)** is a standard for one *agent* to hand a task to another *agent* over HTTP, without knowing how
the other is built. **MCP** lets one agent call a *tool* (a function). The practical difference:

| | MCP | A2A |
|---|---|---|
| Caller talks to | A tool: deterministic function, you pass arguments | An agent: it reasons, may use LLMs and tools, and answers when done |
| Unit of work | `tools/call` returns a result | A **Task** with a lifecycle (submitted, working, completed, failed...) |
| Discovery | `tools/list` after `initialize` | **Agent Card**, a JSON file at a well-known URL |
| Transport here | stdio, one subprocess per tool call | HTTP + JSON-RPC 2.0 to a long-running server |
| In this repo | `order_management` agent -> SQL server | supervisor -> each agent |

Both are used in one request: A2A for supervisor -> agent, MCP for agent -> tool.

## 2. The cast and every kind of communication in this flow

```
you --(stdin/stdout)--> CLI/supervisor process
                          |-- LangGraph state, Neon checkpointer (Postgres)       [DB]
                          |-- routing LLM call to Groq                            [HTTPS to LLM]
                          |-- Langfuse callback                                   [HTTPS, async batch]
                          '-- A2A client --HTTP JSON-RPC--> agent service
                                                              |-- Groq LLM calls            [HTTPS]
                                                              |-- MCP client --stdio--> order MCP server --> Neon  [MCP + SQL]
                                                              |-- Qdrant + local embeddings [HTTPS / in-process]
                                                              '-- Langfuse callback         [HTTPS]
```

| # | Link | Protocol | Style | Where to see it |
|---|---|---|---|---|
| a | You -> CLI | stdin/stdout | text | `session-console.log` |
| b | Supervisor -> routing LLM | HTTPS (Groq) | structured output call | `session-console.log` line 4 (`node=supervisor agent=... reason=...`), Langfuse `ChatGroq` |
| c | Supervisor -> Agent Card | HTTP GET | discovery | `a2a-wire.jsonl` lines 1-2, 5-6 |
| d | Supervisor -> agent | HTTP POST, JSON-RPC 2.0 | request/response, blocking | `a2a-wire.jsonl` lines 3-4, 7-8 |
| e | Agent -> LLM | HTTPS (Groq) | chat completion | Langfuse `a2a-*` traces |
| f | Agent -> tool (order only) | MCP over stdio | JSON-RPC, new subprocess per call | `agent-order_management.log` lines 12-13 |
| g | Agent -> knowledge base (support only) | HTTPS (Qdrant) + local embedding | query | `agent-customer_support.log` line 13 (`step=retrieve docs=2 top_score=0.595`) |
| h | Both processes -> Langfuse | HTTPS | async batches, flushed per request | `langfuse-observations.txt` |
| i | Supervisor <-> Postgres | SQL | checkpoint per thread | not logged; this is how turn 2 knows turn 1 |

Only (c) and (d) are A2A. Notice the supervisor's decision (b) is **not** an A2A call: routing stays an in-process LLM
call, and A2A starts only after the target is chosen.

## 3. Step 0: startup and discovery

`scripts/a2a/run_agents.py` starts one process per agent (`launcher.log`; three when captured, two now). Each builds its agent once and serves two things:

- `GET /.well-known/agent-card.json`: the **Agent Card** (`agent-cards.txt`).
- `POST /`: the JSON-RPC endpoint.

The launcher polls the card URL until it answers; that is why each agent log begins with several `GET /.well-known/...` lines
(readiness polling, then the supervisor's fetches).

### Agent Card fields (generic -> this conversation)

| Field | Generic meaning | `order_management` card here |
|---|---|---|
| `name` | Human/agent name | `order_management`, the same string the supervisor routes on |
| `description` | What the agent is for | "Answers questions about the customer's own purchases..." |
| `supportedInterfaces[].url` | Where to send requests | `http://127.0.0.1:8001` |
| `supportedInterfaces[].protocolBinding` | Transport: `JSONRPC`, `GRPC`, `HTTP+JSON` | `JSONRPC`, so the client picks the JSON-RPC transport |
| `supportedInterfaces[].protocolVersion` | A2A spec version spoken | `1.0` |
| `version` | Version of this agent/card | `0.1.0` (our value; bump on behaviour change) |
| `capabilities.streaming` / `pushNotifications` | Whether SSE streaming / webhooks are supported | both `false`: blocking only, so the client uses plain request/response |
| `defaultInputModes` / `defaultOutputModes` | MIME types accepted/produced | `text/plain` |
| `skills[]` (`id`, `name`, `description`, `tags`, `examples`) | What it can do, for routers and humans | one skill, e.g. `order-history` |

Not set here (but available): `provider`, `securitySchemes`/`securityRequirements` (auth), `signatures` (signed cards),
`documentationUrl`, `iconUrl`, extended cards.

## 4. Turn 1: "What was the last item I bought?"

### Step 1: supervisor routes (in-process, not A2A)
`session-console.log` line 4: `node=supervisor agent=order_management reason=Customer is asking about their last purchase.`
One small-model call; the result sets `next_agent`. Langfuse: span `supervisor` 3.5 s, including a `ChatGroq` generation of ~1 s.

### Step 2: discover the agent
`a2a-wire.jsonl` line 1 (`GET .../8001/.well-known/agent-card.json`) and line 2 (the card, HTTP 200). The client reads
`supportedInterfaces` to learn the URL and binding. This happens on **every** call in the current code (cost: ~30-190 ms here).

### Step 3: the request (`a2a-wire.jsonl` line 3, client -> server)

```json
{
  "jsonrpc": "2.0",
  "id": "2d41ee1e-b9e5-433b-97e8-0a78711279a9",
  "method": "SendMessage",
  "params": {
    "message": {
      "messageId": "msg-1792390297296333700",
      "contextId": "d4874bd8-83d5-4a82-91e1-8353dbe167d2",
      "role": "ROLE_USER",
      "parts": [
        {"text": "What was the last item I bought?"},
        {"data": {"history": []}}
      ]
    },
    "configuration": {},
    "metadata": {"customer_id": "cust001", "thread_id": "d4874bd8-83d5-4a82-91e1-8353dbe167d2"}
  }
}
```

| Field | Generic meaning | In this conversation |
|---|---|---|
| `jsonrpc` | JSON-RPC version, always `"2.0"` | same |
| `id` | Caller-chosen id; the response echoes it so replies can be matched to requests | UUID made by the SDK. Same value in line 4 |
| `method` | The operation. `SendMessage` = give the agent a message (creates or continues a Task). Others: `GetTask`, `CancelTask`, `ListTasks`, subscribe, push-config calls | `SendMessage` |
| `params.message` | The message from the caller | the customer's turn |
| `message.messageId` | Unique id of this message; the basis for **idempotency/dedup** | `msg-<time_ns>` created in `client.py` |
| `message.contextId` | Groups related tasks into one conversation; client-chosen on the first message | we set it to the thread id, so one customer thread = one A2A context everywhere |
| `message.taskId` | Present only when continuing an existing Task | absent: new task. The server assigns one |
| `message.role` | `ROLE_USER` (the caller) or `ROLE_AGENT` | `ROLE_USER`: the supervisor speaks as the user side |
| `message.parts[]` | Content, each part one of `text`, `data` (JSON), `url`, `raw` (bytes), with optional `mediaType`, `filename`, `metadata` | Part 1 `text`: the latest customer message. Part 2 `data`: our own convention `{"history": [...]}`, the prior turns (empty on turn 1) |
| `message.metadata`, `extensions`, `referenceTaskIds` | Free metadata, protocol extensions, links to earlier tasks | unused |
| `params.configuration` | Per-request options: `acceptedOutputModes`, `historyLength`, `returnImmediately` (true = reply right away with a Task to poll) | `{}`: defaults, i.e. wait until finished |
| `params.metadata` | Request-level free-form JSON | our carrier for `customer_id` and `thread_id`. The agent never trusts a customer id typed by the user; it uses this one |

### Step 4: the agent works (server side, `agent-order_management.log`)
- Line 11 `status=start task_id=1f401f99-... context_id=d4874bd8-... customer_id=cust001 history=0`: the executor accepted the message.
  Behind it the SDK moved the Task through **submitted** then **working** (`server.py` publishes them; in blocking mode the
  client only sees the final state).
- Lines 12-13: the order agent's LLM asked for a tool; it called the MCP server (`CallToolRequest`, `tool=get_last_bought_product found=True`).
  This is link (f), a *separate* protocol nested inside the A2A task. `customer_id` was injected from the config the A2A layer
  rebuilt from `metadata`.
- Line 15 `status=complete ... duration_s=16.20 reply_chars=42`. Line 16: uvicorn's `POST / 200`.

### Step 5: the response (`a2a-wire.jsonl` line 4, server -> client)

```json
{
  "jsonrpc": "2.0",
  "id": "2d41ee1e-b9e5-433b-97e8-0a78711279a9",
  "result": {"task": {
    "id": "1f401f99-e229-450b-b65f-e6d3e93fa5f3",
    "contextId": "d4874bd8-83d5-4a82-91e1-8353dbe167d2",
    "status": {"state": "TASK_STATE_COMPLETED", "timestamp": "2026-10-19T06:11:53.515929Z"},
    "artifacts": [{"artifactId": "9a45841f-...", "name": "reply",
                   "parts": [{"text": "Your most recent purchase was a Berry Tee."}]}],
    "history": [ { "...": "the user message above, now with taskId filled in" } ]
  }}
}
```

| Field | Generic meaning | In this conversation |
|---|---|---|
| `id` | Echo of the request id | identical to line 3 |
| `result` | Success payload (an error would be a top-level `error` object with `code`, `message`, `data` instead) | present |
| `result.task` | The server answered with a Task (it could answer with a plain `message` for trivial replies) | yes |
| `task.id` | Server-assigned id; use it for `GetTask`/`CancelTask` and to correlate logs | `1f401f99-...`, same as in the agent log and the client log |
| `task.contextId` | Echo of the conversation id | the thread id |
| `task.status.state` | Lifecycle: `SUBMITTED`, `WORKING`, `COMPLETED`, `FAILED`, `CANCELED`, `REJECTED`, `INPUT_REQUIRED`, `AUTH_REQUIRED`. The last four non-success ones matter for control flow | `TASK_STATE_COMPLETED` |
| `status.timestamp` | When that state was reached (UTC) | 06:11:53.5, matching the wire log line time |
| `status.message` | Optional message explaining the state (used for `failed` / `input_required`) | none |
| `task.artifacts[]` | The *output* of the task (separate from chat messages). Each has `artifactId`, `name`, `parts` | one artifact `reply` with the answer text. The supervisor reads this |
| `task.history[]` | Messages exchanged in the task | only the user message was kept (the agent's answer lives in the artifact) |
| `task.metadata` | Free JSON | unused |

### Step 6: back in the supervisor
`remote_node.py` turns the artifact into `AIMessage(..., name="order_management")` and the checkpointer stores it.
`session-console.log` lines 6-7: `a2a_call status=complete ... state=TASK_STATE_COMPLETED` and
`node=order_management status=complete duration_s=17.70`. Line 9 is the answer.

Timing for turn 1 (Langfuse, `langfuse-observations.txt`): whole supervisor trace 22.6 s; supervisor routing 3.5 s;
`order_management` node 17.7 s (this is the A2A call as the supervisor sees it); agent-side trace 13.5 s, of which the
MCP tool call was **10.4 s** (a new Python subprocess per call on Windows plus the DB query). The difference between 17.7 s
(client), 16.2 s (agent log) and 13.5 s (agent trace) is request handling, trace flush and task bookkeeping.

## 5. Turn 2: "The zipper on my jacket arrived broken, what can I do?"

Same steps, with these differences:

1. **Different agent**: `session-console.log` line 11 routes to `customer_support`; lines 5-6 of the wire log become 5-6/7-8 for port 8002.
2. **History is now filled** (`a2a-wire.jsonl` line 7). The supervisor sent the previous two messages, including which agent wrote the
   answer:
   ```json
   {"data": {"history": [
     {"role": "user", "text": "What was the last item I bought?", "name": null},
     {"role": "assistant", "text": "Your most recent purchase was a Berry Tee.", "name": "order_management"}
   ]}}
   ```
   | Field | Meaning |
   |---|---|
   | `role` | `user` or `assistant`, our own simple convention (not an A2A field) |
   | `text` | message text |
   | `name` | which agent produced an assistant message; the supervisor uses it to keep follow-ups with the same agent. `null` for the customer |

   The history is capped at `a2a.history_messages: 6`. The agent is stateless: it knows only what is in this request.
3. **Same `contextId`**, new `taskId` (`060cb882-...`): one conversation, a separate Task per turn.
4. **More internal work** in `agent-customer_support.log`: line 11 `history=2`, line 12 `condense_query ... history_messages=3`
   (history + new message, so it did an LLM rewrite), line 13 retrieval `docs=2 top_score=0.595`, line 14 generation `claims=5 cited_claims=5`.
5. **Response** (line 8): same shape; the artifact text carries citations `[1] [2]` and a `Sources:` list. Agent trace 11.6 s:
   `condense_query` 4.5 s, `retrieve` 3.7 s, `generate` 3.4 s.

## 6. How the pieces map to code

| Concern | File |
|---|---|
| Agent Cards | `src/hm_agent/a2a/cards.py` |
| Server wrapper, task lifecycle, per-request Langfuse | `src/hm_agent/a2a/server.py` |
| Client (card fetch, `SendMessage`, error on non-completed task, optional wire log) | `src/hm_agent/a2a/client.py` |
| Graph node calling a remote agent, apology on failure | `src/hm_agent/a2a/remote_node.py` |
| One entry point per agent | `src/hm_agent/<agent>/a2a_service.py` |
| Ports and URLs (`ORDER_MANAGEMENT_AGENT_URL` etc. override) | `config/base.yaml`, `src/hm_agent/core/settings.py` |
| Start all agents | `scripts/a2a/run_agents.py` |

## 7. What to look for in the logs

- **Follow one id across files.** `context_id` (= `thread_id`) appears in the CLI log, the client log line, both ends of the wire log,
  and the agent log. `task_id` ties the client's `a2a_call` line to the agent's `status=start/complete` lines. Search for it first when debugging.
- **Every request has a start and a complete/failed line** on the agent side (`status=start` / `status=complete duration_s=...`).
  A `start` with no `complete` means a hang or crash. `status=failed` comes with a traceback and the task ends `TASK_STATE_FAILED`.
- **Client vs server duration**: client `duration_s` (`remote_node`) minus server `duration_s` is the network and handling overhead;
  here 17.7 vs 16.2 s and 15.4 vs 14.3 s.
- **`state=` in the client log must be `TASK_STATE_COMPLETED`.** Anything else makes the node log `status=remote_failed` and answer
  with the generic apology, so a user sees a polite reply while the real problem is only in the logs. Alert on `remote_failed`.
- **Routing quality**: the `reason=` on the supervisor line. Wrong agent = wrong task, not an A2A bug.
- **uvicorn access lines** (`POST / HTTP/1.1" 200 OK`) always say 200 even when the *task* failed; JSON-RPC errors and failed tasks
  travel inside a 200. Never alert on HTTP status alone.
- **The card is fetched every call** (several `GET /.well-known/...` lines per agent log).
- **Noise to know about**: the A2A library logs the *whole* Agent Card at INFO on every fetch (`session-console.log` lines 5 and 12). Consider
  raising `a2a.client.card_resolver` to WARNING.
- **No customer text in app logs**, by design (CLAUDE.md logging rules). The wire log does contain it; keep `A2A_WIRE_LOG` off in production.

## 8. What to look for in Langfuse

From `langfuse-observations.txt`: this session has **four traces**, two per turn.

| Trace | Process | Root name | Latency |
|---|---|---|---|
| 1 | CLI | `hm-supervisor` (turn 1) | 22.6 s |
| 2 | order agent | `a2a-order_management` | 13.5 s |
| 3 | CLI | `hm-supervisor` (turn 2) | 19.6 s |
| 4 | support agent | `a2a-customer_support` | 11.6 s |

- All four share `sessionId = <thread_id>` and `userId = cust001`, so opening the **session** shows the whole conversation. Filter by tag
  `a2a-order_management` / `a2a-customer_support` for the agent side.
- **The supervisor trace and the agent trace are not linked.** In the supervisor trace the agent node is one opaque span (17.7 s);
  the detail (LLM calls, `get_last_bought_product` 10.4 s, `retrieve`, `generate`) is in the other trace. This is the main observability
  gap of this setup; closing it means passing the W3C `traceparent` in `params.metadata` and starting the agent's trace as its child (not done yet).
- Check per turn: supervisor `ChatGroq` ~1 s; the agent-node span versus the agent trace latency (the gap is overhead); the `TOOL` span for MCP;
  `GENERATION` spans for model name and tokens (not logged elsewhere).
- Agent services create their `CallbackHandler` per request and flush after it; a missing agent trace usually means the flush failed or
  the process died mid-request.
- Latency budget: in turn 1 the largest single item was the MCP tool call (10.4 s), not A2A or the LLM.

## 9. Practical things to know beyond MCP (production)

1. **Task state is in memory.** `InMemoryTaskStore` loses all tasks on restart and is per-process, so multiple replicas behind a load balancer
   cannot serve `GetTask` for each other's tasks. Use the SDK's database-backed store before running more than one replica or if you use long tasks.
   Our conversation state is safe regardless: it lives in the supervisor's Postgres checkpointer.
2. **Authentication.** The cards here declare no `securitySchemes` and the endpoints are open. In production put auth on the agent endpoints
   (token/mTLS, declared in the card), do not expose them publicly, and treat `metadata.customer_id` as trusted only because the caller is trusted.
   A compromised caller can ask for any customer's orders.
3. **Timeouts and retries.** `a2a.request_timeout_s` is 90. A timeout on the supervisor leaves the agent still running. Retries must reuse
   the same `messageId` for the server to dedupe; the current client builds a new id per call and never retries.
4. **Blocking vs streaming vs polling.** We use blocking (`streaming: false`). Replies over a minute or human-in-the-loop (`INPUT_REQUIRED`)
   need streaming (SSE), push notifications or polling with `returnImmediately`.
5. **Failure is soft.** One agent down = apology message for that route only (see `agent-down-test.log`; alert on `status=remote_failed`).
   Add health checks on the card URL and a circuit breaker so a dead agent does not cost a full timeout every turn.
6. **Card caching.** Fetching the card per call doubles the round trips; cache it with its `ETag` (the server sends one) and refresh on version change.
7. **Cold start and process model.** Each agent has its own startup cost (support loads an embedding model). The launcher waits for the card
   before declaring ready; do the same in orchestrators (readiness probe = card URL).
8. **MCP spawn cost moved, not removed.** The order agent still starts a subprocess per tool call (10.4 s here). A2A did not cause it, but now it is
   hidden behind one opaque span in the supervisor. Keep the MCP session alive per agent process for production.
9. **Versioning.** The card `version` and `protocolVersion` are your compatibility contract; change skills or message conventions (like our
   `history` data part) with a version bump, since the supervisor and agents deploy independently now.
10. **Our own conventions on top of A2A**: the `history` data part and the `customer_id`/`thread_id` metadata are not part of the standard.
    Another A2A client calling these agents would need to know them (document it, or put it in a card extension).
11. **Groq free-tier limits are per API key, shared by all processes.** Parallel agents or load tests hit the same limit sooner.
12. **PII.** The wire log and Langfuse hold full customer text. Agent logs do not. Decide retention for both.
13. **Cost of the demo**: 2 turns = 6 small-model LLM calls (routing, 2 order-agent, condense, generate, plus routing for turn 2); no loops.

## 10. Known gaps in this implementation (found while running it)

- Supervisor and agent traces are separate (section 8).
- The Agent Card is fetched on every call; the library logs it in full at INFO.
- The wire log shows only the final Task, not the `submitted`/`working` states (blocking mode collapses them).
- In `history`, customer messages have `"name": null` (harmless).
- `out_of_scope` is now a local node, so a question such as "Tell me a joke" is answered without any A2A call.
- Agent-down behaviour is shown in `agent-down-test.log`: card fetch fails, traceback logged with `status=remote_failed`, the customer gets the apology reply.
