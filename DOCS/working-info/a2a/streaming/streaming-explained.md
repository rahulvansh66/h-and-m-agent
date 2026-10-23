# A2A streaming explained from a real session

Hope you already knows the blocking A2A flow (`../basics/a2a-explained.md`) and
push notifications (`../push-notifications/push-notifications-explained.md`).
One real chat (three questions, one thread) is walked through step by step. Every claim points at a saved file in this folder.

| File | What it is |
|---|---|
| `commands.md` | Exact commands used to produce everything below. |
| `session-console.log` | The chat terminal: supervisor, A2A client and push receiver logs, progress lines, answers. |
| `a2a-wire.jsonl` | The HTTP bodies, one JSON object per line (35 lines). `S->C SSE` lines are single Server-Sent-Event frames, `PUSH A->S` lines are webhook posts. |
| `agent-order_management.log`, `agent-shipment_tracking.log` | The agents' own view: start, complete, chunks sent, push sender warnings. |
| `agent-cards.txt` | Cards of `order_management` (streaming) and `shipment_tracking` (streaming + push). |
| `thread-state.txt` | The messages saved in the conversation thread afterwards. |

Session: customer `cust001`, thread (and A2A `contextId`) `0dc2fe6a-f4cc-4691-a9ba-6b22fdc64563`.

| Turn | You asked | Routed to | What was different |
|---|---|---|---|
| 1 | What was the last item I bought? | `order_management` (:8001) | Streaming only. History sent to the agent: **empty**. Wire lines 1-9. |
| 2 | Was that one bought online or in a store? | `order_management` (:8001) | Streaming only. History sent: **2 messages**, which is what lets the agent understand "that one". Lines 10-17. |
| 3 | Where is my order ORD-1007? | `shipment_tracking` (:8003) | **Streaming and push both on.** The same events travel twice. Lines 18-35. |

---

## 1. The idea in one paragraph

A blocking A2A call is a phone call where you hear nothing until the agent says the answer. Streaming keeps the line open and
the agent talks while it works: "I looked up your data", then the answer in pieces, then "done". Push notifications are a
callback number you leave in case the line drops or you hang up. In this repo the supervisor does both for the shipment agent:
it listens on the open line, and has also left the callback number.

| | Blocking `SendMessage` | **Streaming `SendStreamingMessage`** | Push notification | **Streaming + push** |
|---|---|---|---|---|
| Connection | held, silent until the end | held, events flow | none held | held, and a webhook is registered |
| Caller sees progress | no | **yes, live** | yes, at its webhook | yes, on the stream (and at the webhook) |
| Survives a dropped connection | no | no | **yes** | **yes**, via the webhook |
| Caller needs a public endpoint | no | no | yes | yes (only for the fallback) |
| Turn latency for a 20 s task | 20 s | 20 s, but with live feedback | < 1 s acknowledgement | 20 s with live feedback; ack only if the stream breaks |

## 2. The cast

```
you --(stdin/stdout)--> CLI / supervisor process
                          |-- routing LLM call (Groq)
                          |-- A2A client --POST SendStreamingMessage--> agent (:8001 / :8003)
                          |        '<---- SSE frames: task, status, artifact chunks, completed ----'
                          '-- push receiver (:8010)  <--POST webhook-- agent (:8003 only, same events)
```

What the code does with it (all new in this change):

| Step | Where | What happens |
|---|---|---|
| Runner yields `Chunk`s | `a2a/chunks.py`, `a2a/server.py::graph_runner` | `progress` = a status line, `answer` = a piece of the reply |
| Executor publishes events | `a2a/server.py::RunnerExecutor` | progress -> `working` status update with a message; answer -> artifact chunk; last one flagged `lastChunk` |
| Cards advertise it | `a2a/cards.py` | `streaming: true` on all agents, `pushNotifications: true` only on shipment |
| Client consumes the stream | `a2a/client.py::stream_remote_agent` | yields `started / progress / answer / state` events |
| Graph node forwards them | `a2a/remote_node.py` | writes each event to LangGraph's `custom` stream, returns the full reply as one `AIMessage` |
| CLI / API show them | `turns.py`, `cli.py`, `api/` | `graph.stream(["updates","custom"])` -> printed lines or SSE to the browser |

## 3. Turn 1, line by line (streaming, no history)

Wire lines 1-9, all against `order_management`:

| # | Time | Dir | What it is |
|---|---|---|---|
| 1-2 | 12:53:45.77 | GET card | Client reads the Agent Card. It says `"streaming": true`, so the client will stream. |
| 3 | 45.82 | `C->S` | **`SendStreamingMessage`** (not `SendMessage`). Text part plus data part `{"history": []}`; `"configuration": {}` (no webhook). |
| 4 | 45.82 | SSE | First frame is the whole **Task**: `state: TASK_STATE_SUBMITTED`, with the user message in `history`. |
| 5 | 45.83 | SSE | `statusUpdate` `TASK_STATE_WORKING`, no message: "I started". |
| 6 | **52.29** | SSE | `statusUpdate` `TASK_STATE_WORKING` **with a message**: `Looked up your order data`. This is our *progress* chunk; it fired when the ReAct agent's `tools` node finished. |
| 7 | 53.22 | SSE | `artifactUpdate`, `artifactId: "reply"`, text `Your most recent purchase`. No `append` flag: this **creates** the artifact. |
| 8 | 53.54 | SSE | `artifactUpdate`, text ` was a **Berry Tee**.`, **`"append": true, "lastChunk": true`**. |
| 9 | 53.54 | SSE | `statusUpdate` `TASK_STATE_COMPLETED`. Terminal: the stream ends. |

Things to notice:

- Every frame carries the same JSON-RPC `"id"` (`5d736249-...`): it is the id of your request, so you can match frames to the call that opened the stream.
- The agent's own log says `chunks=3 reply_chars=46`: one progress + two answer pieces (`agent-order_management.log`).
- Honest latency picture: 6.5 s went into the LLM and the tool call before *anything* could be streamed; the answer then arrived as
  two pieces 0.3 s apart. For short answers from a fast model, **token streaming barely changes perceived latency**. The progress
  line (6) is what gives feedback earlier. Streaming pays off for long answers and long-running work (turn 3).
- The reply pieces are about 24 characters each on purpose (`TOKEN_FLUSH_CHARS`). One SSE frame per LLM token would be hundreds of events for a paragraph; grouping keeps frames meaningful.

## 4. Turn 2: what changes when there is history

Compare the request bodies (wire lines 3, 12, 20), same `contextId` in all three:

| Turn | `parts[1].data.history` sent to the agent | `taskId` |
|---|---|---|
| 1 | `[]` | `6f48537e-...` |
| 2 | `[user "What was the last item I bought?", assistant(order_management) "Your most recent purchase was a **Berry Tee**."]` | `443f5eb2-...` (new) |
| 3 | the four messages of turns 1-2 | `a66addb8-...` (new) |

- **The agents are stateless; the supervisor holds the conversation** (Neon checkpointer, `thread-state.txt`). On every call it
  replays the last `a2a.history_messages` (6) messages as a data part, and sends the new message as the text part.
- **`contextId` = the conversation** (we set it to the thread id), **`taskId` = one unit of work** (new each turn, assigned by the agent).
  Turn 2 is a *new task in the same context*, not a continuation of turn 1's task: turn 1's task is already `completed`, and a completed task cannot take more messages.
- That is why "Was that one bought online or in a store?" works: the order agent sees "Berry Tee" in the history and calls its tools to find that purchase. Without the history part it would have no idea what "that one" is.
- The routing decision is also history-aware: the supervisor keeps follow-ups with the same agent via the `name` on the AI messages (`"name": "order_management"` in the history entries).
- The streaming part is unchanged: lines 13-17 have the same shape as 4-9, except that the reply is **one** chunk, so it has `lastChunk: true` and *no* `append` (line 16). Rule: `append` says "add to an existing artifact", `lastChunk` says "nothing follows".

## 5. Turn 3: streaming and push both on

### What is different in the request (wire line 20 vs line 12)

```
"configuration": {"taskPushNotificationConfig": {"url": "http://127.0.0.1:8010/a2a/notify", "token": "<redacted>"}}
```

Same `SendStreamingMessage`, but now with a webhook registered *for this task*. The client did **not** set `return_immediately`
(the push-only flow does, and gets a task back at once). So the connection stays open.

### The same events, twice (wire lines 21-35)

| Time (+s from request) | Event | On the stream (`S->C SSE`) | At the webhook (`PUSH A->S`) |
|---|---|---|---|
| +0.04 | task `SUBMITTED` | line 22 | line 21 |
| +0.05 | `WORKING` | 24 | 23 |
| +0.05 | progress: Contacting the carrier about ORD-1007 | 26 | 25 |
| +5.02 | progress: Waiting for the carrier system to respond | 28 | 27 |
| +10.07 | progress: Reading the latest scan events | 29 | **missing** (section 6) |
| +15.03 | progress: Preparing your delivery update | 31 | 30 |
| +20.03 | answer artifact, `lastChunk: true` | 33 | 32 |
| +20.04 | `COMPLETED` | 35 | 34 |

The agent's push sender and its SSE writer are two independent consumers of the same event queue. Nothing in A2A makes them
coordinate; the client has to.

### What the supervisor did with both (`session-console.log`)

1. The node consumed the stream until `TASK_STATE_COMPLETED`, forwarding progress lines to the terminal live (`  ... Contacting the carrier about ORD-1007`).
2. It returned the joined reply as the turn's answer after **20.27 s** (`node=shipment_tracking status=complete duration_s=20.27`).
3. Meanwhile the webhook receiver logged `status=received` for each notification. It did nothing with them, because the task is not registered as "pending".
4. After the turn, the node called `receiver.mark_streamed(task_id)` and the receiver logged `status=streamed_ignore`. Any notification arriving later (or one that arrived *before* this point and was parked as "finished early") is dropped, so the answer is never delivered twice.

### Push-only vs streaming + push for the same agent

| | Push only (`../push-notifications/`) | Streaming + push (this folder) |
|---|---|---|
| First reply to the user | acknowledgement after ~0.8 s | nothing until the stream ends, but progress lines every 5 s |
| Can the user type meanwhile | yes | no (the turn is running) |
| Real answer arrives | 20 s later, out of band, printed as `assistant (shipment_tracking, update)>` | as the turn's own answer, saved to the thread normally |
| If nobody is connected any more | still delivered | **still delivered, if the stream broke** (below) |
| Moving parts | webhook receiver is the only path | stream is the main path, webhook is insurance |

**The fallback (not exercised live, covered by `tests/test_remote_node.py::test_stream_break_falls_back_to_push`):** if the stream
breaks before a terminal state and the task was registered for push, the node registers the task with the receiver, answers
with the "I'll post the result here as soon as it arrives" acknowledgement, and the existing push path delivers the answer later.

Which to choose per agent is a product decision: stream-first is right when the user waits anyway and wants feedback; push-first is right when the task can take minutes and the user should be free to leave.

## 6. The lost notification (wire line 29 has no `PUSH` twin)

`agent-shipment_tracking.log` shows what happened on the agent side:

```
WARNING ... base_push_notification_sender  Some push notifications failed to send for task_id=a66addb8-...
httpx.ReadError
```

Cause: the supervisor's webhook receiver is a uvicorn server, whose default keep-alive timeout is **5 s**. Our progress events are exactly
5 s apart. The agent's HTTP client reuses a pooled connection, and the server closes it at the very moment the next POST goes out, so the request
fails with `ReadError`. The A2A SDK sender does **not** retry. The event was simply lost.

Fix (`push_receiver.py`, `KEEP_ALIVE_S = 75`): keep idle connections open longer than the gap between updates. Verified with a second run of the
same question: 8 `PUSH` posts for 8 `SSE` frames, no warning.

The lesson is bigger than the bug:

- **Push notifications are at-most-once, best effort.** Never build logic that needs every notification. This repo reads the final result with `GetTask` after a terminal notification, which is why losing a *progress* notification is harmless, but losing the *terminal* one would leave the task unreported.
- **A stream is ordered and tied to one connection**; a webhook is a series of independent HTTP calls with no ordering guarantee.
- This is exactly why "stream live, push as fallback" is a good pairing: the stream is complete and ordered while it lasts, and the webhook covers the case when it does not.

## 7. Conceptual points to remember

1. **Streaming is negotiated.** The card says `capabilities.streaming`; the client chooses `SendStreamingMessage` or `SendMessage`. A blocking client still works against a streaming agent: the SDK assembles the events into the final Task.
2. **Two kinds of event, two jobs.** `statusUpdate` carries *state and progress messages* ("working", "Looked up your order data"). `artifactUpdate` carries *the result*. Do not put the answer in status messages or progress in artifacts: clients treat them differently (the artifact is the deliverable).
3. **Chunks are parts, not string concatenation.** An appended artifact chunk becomes another *part* of the same artifact. Joining parts with the SDK default (`"\n"`) turned `"...is with DHL"` + `"Express"` into two lines. The blocking path in our own tests hit this (`task_reply` now joins parts of one artifact with `""`).
4. **`append` / `lastChunk` semantics.** First chunk: no `append`. Later chunks: `append: true`. Final chunk: `lastChunk: true`. The executor holds back one chunk so it can flag the real last one instead of sending an empty closing frame.
5. **Only stream what is final.** `order_management`'s last model call *is* the customer-facing text, so its tokens stream. `customer_support` does not: its answer is structured output that is checked for citations and may be regenerated, so streaming it would show text that is later retracted. It streams step progress and sends the validated answer whole.
6. **Runners are blocking; the server is async.** The runner runs in a worker thread and hands chunks to the event loop through a queue (`call_soon_threadsafe`). A generator that raises is turned into a `failed` task, not a crashed server.
7. **Logging can silently break streaming.** The original wire-log hook called `response.aread()`, which waits for the *end* of the response, so it would have turned every SSE stream into one buffered blob. For event streams the hook now wraps the body and logs each `data:` line as it passes.
8. **Timeouts mean something different.** For a stream, the client timeout is the longest *silence* between events, not the total duration. A progress heartbeat (every 5 s here) keeps long tasks alive.
9. **Proxies buffer SSE unless told not to.** Behind nginx / ALB / API Gateway you need `X-Accel-Buffering: no` (sse-starlette sets it), `Cache-Control: no-cache`, and an idle timeout longer than your heartbeat (ALB default is 60 s). Check that every hop in front of the API supports streamed responses before relying on it.
10. **Know where results live.** The agent keeps task state in memory (`InMemoryTaskStore`), so it is lost on restart. What happens to a running task when the stream client disconnects depends on the SDK and was not tested here; the push fallback is covered by a unit test only, not a live disconnect.
11. **Webhooks are an inbound attack surface.** The receiver checks a per-process random token (`X-A2A-Notification-Token`) with `hmac.compare_digest`. In production also validate the callback URL you accept (otherwise an agent can be made to POST anywhere: SSRF) and use HTTPS.
12. **The graph still sees one message.** Progress and tokens go to LangGraph's `custom` stream (`get_stream_writer`); the node still returns one `AIMessage`, so checkpoints, history and routing are identical with or without streaming.
13. **Free-tier reality.** Streaming does not reduce LLM calls; it only changes when text appears. Do not add LLM calls to "make progress messages nicer": the progress lines here are fixed strings.

## 8. How an API client sees it

`POST /v1/chat/stream` (FastAPI, `src/hm_agent/api/`) relays the same events as Server-Sent Events. A real response from a follow-up on an existing thread:

```
event: thread           data: {"type": "thread", "thread_id": "a7182d15-..."}
event: agent_selected   data: {"type": "agent_selected", "agent": "order_management"}
event: started          data: {"type": "started", "agent": "order_management", "task_id": "87288c8c-..."}
event: progress         data: {"type": "progress", "agent": "order_management", "text": "Looked up your order data"}
event: token            data: {"type": "token", "agent": "order_management", "text": "It was purchased online."}
event: reply            data: {"type": "reply", "agent": "order_management", "text": "It was purchased online."}
event: done             data: {"type": "done"}
```

`token` events are the A2A artifact chunks, `progress` the A2A status messages, `reply` the full text (what is saved in the thread). Clients should render `token`s as they arrive and use `reply` as the source of truth.

## 9. Where to look in the code

| Concept | File |
|---|---|
| Chunk contract | `src/hm_agent/a2a/chunks.py` |
| Runner -> A2A events, thread/queue bridge, `lastChunk` hold-back | `src/hm_agent/a2a/server.py` |
| Streaming client, `RemoteEvent`, joined artifact parts | `src/hm_agent/a2a/client.py` |
| Stream-first with push fallback | `src/hm_agent/a2a/remote_node.py`, `src/hm_agent/a2a/push_receiver.py` (`mark_streamed`, `KEEP_ALIVE_S`) |
| SSE-safe wire logging | `src/hm_agent/a2a/wire_log.py` |
| Progress from the slow lookup | `src/hm_agent/shipment_tracking/tools.py::stream_carrier_status` |
| One turn as events (CLI and API) | `src/hm_agent/turns.py` |
| HTTP API | `src/hm_agent/api/` (`routes/chat.py` for the SSE endpoints) |
| Tests | `tests/test_a2a_streaming.py`, `tests/test_remote_node.py`, `tests/test_api.py` |
