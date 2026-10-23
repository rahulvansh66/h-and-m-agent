# A2A push notifications explained from a real session

Hope you already knows the blocking A2A flow (see `../basics/a2a-explained.md`).
One real chat (two questions, one thread) is walked through step by step. Every claim points at a saved file in this folder.

| File | What it is |
|---|---|
| `commands.md` | Exact commands used to produce everything below. |
| `session-console.log` | The chat terminal: supervisor, A2A client and push receiver logs plus the answers. |
| `a2a-wire.jsonl` | The HTTP bodies, one JSON object per line (16 lines), including the notifications the agent pushes (`PUSH A->S`). |
| `agent-shipment_tracking.log` | The push-enabled agent: task start, tool call, each "Push-notification sent" line. |
| `agent-cards.txt` | Cards of `order_management` (push off) and `shipment_tracking` (push on). |
| `thread-state.txt` | The messages saved in the conversation thread afterwards. |

Session: customer `cust001`, thread (and A2A `contextId`) `9d9db403-8f2d-432d-a18b-824e35a8678b`.

| Turn | You asked | Routed to | How it was answered |
|---|---|---|---|
| 1 | Where is my order ORD-1007? | `shipment_tracking` (:8003) | **Push**: acknowledgement after 0.8 s, real answer 20 s later |
| 2 | How many purchases have I made? | `order_management` (:8001) | **Blocking**: answered in 10.7 s, while turn 1's task was still running |

---

## 1. The idea in one paragraph

A normal A2A call is a phone call: the supervisor dials, stays on the line, and hangs up when the agent has the
answer. Push notifications turn it into a voicemail: the supervisor says "here is my request, and here is my
callback number (a webhook URL)", the agent says "got it, task id X" and hangs up at once, and when the work
changes state the agent calls the number back. Nobody holds a connection open for the 20 seconds (in real life:
minutes or hours) the carrier lookup takes.

| | Blocking (`SendMessage`) | Streaming (SSE) | **Push notification** | Polling (`GetTask`) |
|---|---|---|---|---|
| Who keeps the connection | caller, until done | caller, open for the whole task | nobody | nobody |
| How the caller learns of progress | gets only the end | agent pushes on the open connection | agent calls the caller's webhook | caller asks repeatedly |
| Caller needs a public endpoint | no | no | **yes** (the webhook) | no |
| Good for | seconds | live progress / tokens | long or offline tasks | fallback for any of the others |

In this repo only `shipment_tracking` uses it; the other two agents stay blocking (section 7, item 10).

## 2. The cast

```
you --(stdin/stdout)--> CLI / supervisor process
                          |-- routing LLM call (Groq)                                   [turn 1 and 2]
                          |-- A2A client --HTTP JSON-RPC--> shipment_tracking (:8003)
                          '-- push receiver (Starlette on :8010)  <--HTTP POST-- shipment_tracking
                                  token check -> GetTask -> print + save into the thread (Neon checkpointer)
```

| # | Link | Protocol | Where to see it |
|---|---|---|---|
| a | Supervisor -> Agent Card | HTTP GET | wire lines 1-2 |
| b | Supervisor -> agent: start the task and register the webhook | JSON-RPC `SendMessage` | wire lines 3, 5 |
| c | **Agent -> supervisor webhook**: task updates | HTTP POST, one per update | wire lines 4, 6, 11, 12 (`PUSH A->S`) |
| d | Supervisor -> agent: read the final result | JSON-RPC `GetTask` | wire lines 13-16 |

The new link is (c): it goes in the **opposite direction** to every other A2A call, which is why the supervisor needs
an HTTP server of its own (`src/hm_agent/a2a/push_receiver.py`).

## 3. Step 0: the agent advertises it (Agent Card)

`agent-cards.txt`, the two cards side by side:

```json
"name": "order_management",    ... "capabilities": { "streaming": false }
"name": "shipment_tracking",   ... "capabilities": { "streaming": false, "pushNotifications": true }
```

(`order_management` shows no `pushNotifications` key because `false` is the default and is omitted in the JSON.)

`capabilities.pushNotifications: true` means "I accept a webhook per task and will call it". It is only an
offer. Nothing is pushed unless the client asks, per task, in step 2. The client reads this flag
(`src/hm_agent/a2a/client.py`): if the flag is false it never sends a webhook, even though the supervisor has one.

Code: `PUSH_AGENTS` in `src/hm_agent/a2a/cards.py` sets the flag, and `build_a2a_app` in `src/hm_agent/a2a/server.py`
attaches the SDK's push config store and sender only for such cards.

`src/hm_agent/a2a/cards.py`: the set of push-enabled agents, and the flag on the card:

```python
# Agents whose tasks outlive a normal request, so their card advertises push notifications.
PUSH_AGENTS = frozenset({SHIPMENT_TRACKING})

# inside build_agent_card(agent)
capabilities=AgentCapabilities(streaming=True, push_notifications=agent in PUSH_AGENTS),
```

`src/hm_agent/a2a/server.py`, in `build_a2a_app`: the store and sender are created only when the card says so:

```python
push_kwargs = {}
if card.capabilities.push_notifications:
    # The caller registers a webhook per task (SendMessage configuration); the sender POSTs
    # every status/artifact update of that task to it.
    push_store = InMemoryPushNotificationConfigStore()
    push_kwargs = {
        "push_config_store": push_store,
        "push_sender": BasePushNotificationSender(httpx.AsyncClient(timeout=10), push_store),
    }
handler = DefaultRequestHandlerV2(
    agent_executor=RunnerExecutor(agent, runner),
    task_store=InMemoryTaskStore(),
    agent_card=card,
    **push_kwargs,   # empty for agents that stay blocking
)
```

## 4. Turn 1, step by step (what really happened)

### Step 1: supervisor routes, then reads the card (`session-console.log`, wire 1-2)
`node=supervisor agent=shipment_tracking reason=Customer is asking for the location/status of a specific order.`
Then `GET :8003/.well-known/agent-card.json` at 12:01:41.230. The client sees `pushNotifications: true`.

### Step 2: the request carries the callback (wire line 3, client -> server)

```json
{
  "method": "SendMessage",
  "params": {
    "message": { "messageId": "msg-1792670501353523400",
                 "contextId": "9d9db403-8f2d-432d-a18b-824e35a8678b",
                 "role": "ROLE_USER",
                 "parts": [ {"text": "Where is my order ORD-1007?"}, {"data": {"history": []}} ] },
    "configuration": {
      "taskPushNotificationConfig": { "url": "http://127.0.0.1:8010/a2a/notify",
                                      "token": "oE2Op...Vx" },
      "returnImmediately": true
    },
    "metadata": { "customer_id": "cust001", "thread_id": "9d9db403-..." }
  },
  "id": "59147e04-...", "jsonrpc": "2.0"
}
```

Only `configuration` is new compared to the blocking call (`v0.1.0/a2a-explained.md` section 4):

| Field | Meaning | Here |
|---|---|---|
| `configuration.returnImmediately` | `true`: reply right away with the Task and keep working in the background. Without it the server holds the request until the task finishes (the blocking mode) | `true` |
| `taskPushNotificationConfig.url` | The webhook the agent will POST to. Must be reachable **from the agent** | the supervisor's receiver, `http://127.0.0.1:8010/a2a/notify` |
| `taskPushNotificationConfig.token` | A secret chosen by the client; the agent echoes it in the `X-A2A-Notification-Token` header of every push so the receiver can tell real calls from forged ones | random per supervisor process |
| `taskPushNotificationConfig.authentication` | Optional: scheme and credentials the agent should send as an `Authorization` header. Not used here | absent |
| `taskPushNotificationConfig.id`, `tenant`, `taskId` | Optional ids; the server fills in the task id | absent |

### Step 3: immediate response (wire line 5, server -> client, 12:01:41.462, 107 ms after the request)

```json
{
  "result": {
    "task": {
      "id": "7f14a7dc-7df3-4aa9-91dd-6580128366c5",
      "contextId": "9d9db403-8f2d-432d-a18b-824e35a8678b",
      "status": { "state": "TASK_STATE_SUBMITTED" },
      "history": [
        {
          "messageId": "msg-1792670501353523400",
          "contextId": "9d9db403-8f2d-432d-a18b-824e35a8678b",
          "taskId": "7f14a7dc-7df3-4aa9-91dd-6580128366c5",
          "role": "ROLE_USER",
          "parts": [ { "text": "Where is my order ORD-1007?" }, { "data": { "history": [] } } ]
        }
      ]
    }
  },
  "id": "59147e04-43e6-414b-bd17-6afa6c970089",
  "jsonrpc": "2.0"
}
```

No artifact, state `SUBMITTED`: the task exists but has no answer. Note that the sentence "I'm checking with the
carrier now..." is **not in this response**. The server never sends it; the response only echoes the user's own
message. The supervisor writes the placeholder itself: it sees a `Task` in state `SUBMITTED` instead of a finished
answer, so `remote_node.py` registers `task_id` with the push receiver (`status=registered`) and substitutes its
constant `PENDING_REPLY`:

```python
PENDING_REPLY = "I'm checking with the carrier now. This can take a little while; I'll post the result here as soon as it arrives."
```

`node=shipment_tracking status=pending duration_s=0.81`, then the customer sees:
"I'm checking with the carrier now. This can take a little while; I'll post the result here as soon as it arrives."
The graph run for this turn is finished; the CLI is free for the next question.

### Step 4: the agent pushes its first updates (wire lines 4 and 6)
While the response travels back, the agent already POSTs to the webhook (note the timestamps):

| Wire line | Time | Body | Meaning |
|---|---|---|---|
| 4 | 12:01:41.449 | `{"task": {... "status": {"state": "TASK_STATE_SUBMITTED"} ...}}` | the whole Task snapshot |
| 6 | 12:01:41.461 | `{"statusUpdate": {"taskId": ..., "status": {"state": "TASK_STATE_WORKING", "timestamp": "...41.365Z"}}}` | the executor started |
| 5 | 12:01:41.462 | (the response to the `SendMessage`, shown above) | |

Both pushes **arrived before the response to the original request**. See section 7, item 3.
The receiver logged them as `state=TASK_STATE_SUBMITTED` and `state=TASK_STATE_WORKING` and, because they are not
terminal, did nothing else. The agent side (`agent-shipment_tracking.log`) shows the matching
`Push-notification sent for task_id=7f14a7dc-...` lines.

### Step 5: the agent works for 20 seconds
`agent-shipment_tracking.log`: `tool=check_carrier_status status=start order_id=ORD-1007 delay_s=20.0`.
The agent's executor runs the tool in a worker thread (`asyncio.to_thread`), so the agent's event loop is not blocked.

`RunnerExecutor.execute` in `src/hm_agent/a2a/server.py` wraps the blocking runner (which ends in the tool's `time.sleep`) in `produce()`, runs it on a worker thread, and reads its chunks from a queue:

```python
loop = asyncio.get_running_loop()
outbox: asyncio.Queue[Chunk | Exception | None] = asyncio.Queue()

def produce() -> None:
    """Worker thread: iterate the runner and hand each chunk to the event loop."""
    try:
        for chunk in self._runner(text, history, customer_id, thread_id):
            loop.call_soon_threadsafe(outbox.put_nowait, chunk)
    except Exception as exc:  # reported to the task below
        loop.call_soon_threadsafe(outbox.put_nowait, exc)
    finally:
        loop.call_soon_threadsafe(outbox.put_nowait, None)  # end marker

worker = asyncio.create_task(asyncio.to_thread(produce))   # blocking work leaves the event loop
...
while (item := await outbox.get()) is not None:            # the loop stays free while waiting
    ...
```

While the thread sleeps, the event loop keeps serving other requests (for example `GetTask`) and sending pushes.

Nothing is on the wire for 20 s, which is the point.

### Step 6: meanwhile turn 2 runs the *blocking* path (wire lines 7-10)
At 12:01:49.6 (8 s into the task) the customer asked "How many purchases have I made?". The supervisor routed it to
`order_management`, whose card has no `pushNotifications`. The client therefore sent a normal `SendMessage` without
`configuration`, and held the connection until the answer arrived 10.1 s later (wire line 10, `TASK_STATE_COMPLETED`
with the answer in `artifacts`). One supervisor, two modes, chosen per agent by its card.

Look at the `history` data part of turn 2's request: it contains the placeholder
"I'm checking with the carrier now..." as an assistant message from `shipment_tracking`. The conversation
history sees what was *said*, not the answer that is still pending.

The `SendMessage` request to `order_management` for turn 2 (`a2a-wire.jsonl`, `configuration` is empty because this agent is blocking):

```json
{
  "method": "SendMessage",
  "params": {
    "message": {
      "messageId": "msg-1792670509695144900",
      "contextId": "9d9db403-8f2d-432d-a18b-824e35a8678b",
      "role": "ROLE_USER",
      "parts": [
        { "text": "How many purchases have I made?" },
        { "data": { "history": [
            { "name": null, "role": "user", "text": "Where is my order ORD-1007?" },
            { "name": "shipment_tracking", "role": "assistant",
              "text": "I'm checking with the carrier now. This can take a little while; I'll post the result here as soon as it arrives." }
        ] } }
      ]
    },
    "configuration": {},
    "metadata": { "customer_id": "cust001", "thread_id": "9d9db403-8f2d-432d-a18b-824e35a8678b" }
  },
  "id": "b91c4f4e-f012-4764-92f3-d4072c9e5ec3",
  "jsonrpc": "2.0"
}
```

The second history entry is the placeholder, not the carrier's answer: that answer does not exist yet (it arrives 12 s later by push).

### Step 7: the final pushes (wire lines 11 and 12, 12:02:01)

```json
// line 11, 12:02:01.383
{ "artifactUpdate": { "taskId": "7f14a7dc-...", "artifact": { "name": "reply",
    "parts": [ { "text": "Order ORD-1007 is with DHL Express (tracking DHL4966249063). Last scan: Hamburg sorting hub. In transit, expected delivery in 2 days." } ] } } }

// line 12, 12:02:01.399
{ "statusUpdate": { "taskId": "7f14a7dc-...", "status": { "state": "TASK_STATE_COMPLETED", "timestamp": "2026-10-22T12:02:01.377334Z" } } }
```

From `WORKING` (41.365) to `COMPLETED` (01.377) is 20.01 s, the dummy carrier delay. Four pushes in total for one
task: `SUBMITTED`, `WORKING`, artifact, `COMPLETED` (4 "Push-notification sent" lines in the agent log).

### Step 8: the receiver reads the result and delivers it (wire lines 13-16, `session-console.log`)
The receiver acts only on a **terminal** state (`COMPLETED`, `FAILED`, `CANCELED`, `REJECTED`). On `COMPLETED` it
does not trust the notification body; it asks the agent: `GetTask {"id": "7f14a7dc-..."}` (line 15) and reads
the artifact from the response (line 16). Then (`cli.py`, `_make_deliver`):

1. Prints `assistant (shipment_tracking, update)> Order ORD-1007 is with DHL Express ...` at once.
2. Saves the same text as an `AIMessage(name="shipment_tracking")` into the thread, so the conversation continues as if the agent had answered normally.
3. Log: `component=push_receiver status=delivered task_id=7f14a7dc-... state=TASK_STATE_COMPLETED`.

`thread-state.txt` shows the result: user, placeholder, user, order answer, **shipment answer**.

### Timeline

```
12:01:41.23  GET card                           12:01:49.6  turn 2 starts (blocking path)
12:01:41.36  SendMessage + webhook config       12:01:59.8  turn 2 answered (10.1 s)
12:01:41.45  push: SUBMITTED   <- before the response
12:01:41.46  push: WORKING     <- before the response
12:01:41.46  response: Task SUBMITTED  -> "checking with the carrier..."
             ... 20 s: tool check_carrier_status ...
12:02:01.38  push: artifact (the answer)
12:02:01.40  push: COMPLETED
12:02:02.01  GetTask -> 12:02:02.02 result -> printed + saved
```

## 5. What each side must do (map to code)

| Concern | Agent side | Supervisor side |
|---|---|---|
| Advertise | `capabilities.push_notifications=True` (`cards.py`) | read the flag before sending a webhook (`client.py`) |
| Accept a webhook | `InMemoryPushNotificationConfigStore` (stores the config per task id) | send `taskPushNotificationConfig` + `returnImmediately` (`client.py`) |
| Send updates | `BasePushNotificationSender` POSTs every task event with the token header (`server.py`; the executor needed **no change**, the SDK pushes on every state change) | run an HTTP endpoint that accepts them (`push_receiver.py`) |
| Verify | include the token the client gave | compare the `X-A2A-Notification-Token` header with `hmac.compare_digest`; 401 if wrong |
| Finish | keep the task and artifact available for `GetTask` | on a terminal state `GetTask`, deliver, save to the thread, then forget the task |
| Slow work | tool runs in a worker thread (`asyncio.to_thread`) | the graph turn ends at once with a placeholder (`remote_node.py`) |

## 6. What to look for in the logs

- **Follow `task_id`** across files: client `a2a_call status=pending task_id=...`, receiver `status=received/registered/delivered task_id=...`, agent `status=start/complete task_id=...`, and the `Push-notification sent for task_id=...` lines.
- **A task with `registered` but no `delivered`** means the answer never came back (webhook unreachable, agent crashed, or the CLI exited). One-shot runs wait up to 120 s (`PUSH_WAIT_S`) and then log `status=push_timeout`.
- **`status=rejected reason=bad_token`** on the receiver is a forged or misconfigured push (or an agent talking to a receiver from an older process: the token changes on every supervisor start).
- **Agent log `Error sending push-notification`** with a traceback: the webhook was unreachable. The SDK sender logs it and does not retry (item 5 below).
- **State `None` in `status=received`** is an artifact update, not an error.
- **Duration fields:** `node=shipment_tracking status=pending duration_s=0.81` is how long the *customer waited for the acknowledgement*, not the task time. The task time is in the agent log (`status=complete duration_s=`) and in the `WORKING` -> `COMPLETED` timestamps.

## 7. Important aspects to know

1. **The flag is an offer, the config is the request.** `pushNotifications: true` on the card does nothing alone. Each task needs its own `taskPushNotificationConfig`, and the webhook belongs to that task only.
2. **The notification is a doorbell, not the package.** The body can carry the artifact (it did, wire line 11), but the receiver reads the result with `GetTask` anyway: the task store is the source of truth, a notification can be duplicated, reordered or lost, and `GetTask` gives the same answer however often you call it.
3. **You get several notifications per task, and not necessarily in order.** Here 4 pushes for one task, two of them arrived before the response to the start call. The receiver therefore (a) ignores non-terminal updates, (b) buffers a terminal state that arrives before `register()` ran, and (c) keeps a set of claimed task ids so a duplicate terminal notification cannot deliver twice. Design every webhook handler to be idempotent.
4. **The webhook must be reachable from the agent, in the opposite direction.** On one machine `127.0.0.1:8010` works. Across hosts or AWS the supervisor needs a stable HTTP endpoint (load balancer, API Gateway, Lambda URL) reachable from the agents. A CLI process cannot receive pushes once it exits. That is why this receiver lives inside a long-running process and why one-shot runs wait. A real deployment would put the receiver in a service and push results to the user's session (websocket, email, etc.).
5. **Delivery is best effort, with no retry in this setup.** The SDK sender logs a failure and moves on (`BasePushNotificationSender` returns False; there is no retry or dead-letter). A lost webhook means a lost answer unless the client also polls. A robust client keeps a periodic `GetTask` for tasks that stay `pending` too long. This repo has none yet.
6. **Security is your job.** The agent POSTs to whatever URL the client supplies, so an untrusted client can aim the agent at internal addresses (SSRF). The SDK accepts a `push_url_validator`; none is configured here. The `token` check on the receiver is the minimum, and the token is visible in `a2a-wire.jsonl` (wire line 3), so **keep `A2A_WIRE_LOG` off in production**. For real use also consider `authentication` (a bearer token the agent sends), HTTPS, and not exposing the receiver publicly beyond the agents.
7. **Concurrent writes to the same thread can lose the pushed answer. This bit during development.** In my first run, the push arrived while turn 2 was still executing; `graph.update_state` wrote the shipment answer, then turn 2 finished and wrote its checkpoint on top, built from the state *before* the update. Result: the answer was printed but missing from the saved thread. A stub graph reproduces it exactly: without a lock the saved messages are `['turn 2', 'turn-2 reply']`; with the lock they are `['turn 2', 'turn-2 reply', 'PUSHED ANSWER']`. The fix is `_THREAD_LOCK` in `cli.py`, held during a turn and during the save. In the final run recorded here turn 2 happened to finish 1.6 s before the push, so the lock was not exercised in this particular log; the stub test is what proves it.
8. **State is in memory on both sides.** Tasks, artifacts and push configs live in the agent's memory (`InMemoryTaskStore`, `InMemoryPushNotificationConfigStore`), and pending tasks live in the supervisor's. A restart of either loses in-flight work. Use the SDK's database-backed stores and persist the pending list before running replicas or long tasks.
9. **The customer sees two messages.** The placeholder is stored in the thread and sent to other agents as history (turn 2's request contained it). Phrase placeholders so they make sense later ("I'll post the result here" is true; "your parcel is on its way" would not be).
10. **Only the agent whose tasks are slow should advertise it.** If `order_management` advertised push, "what did I last buy?" would also answer with a placeholder and an update 10 s later. Make the decision per agent, in the card.
11. **Timeouts change meaning.** `a2a.request_timeout_s` (90 s) now covers only the short start call. How long the task may take is a separate, explicit policy (here: one-shot runs wait 120 s; chat mode waits forever).
12. **Cost and tracing.** The push path adds no LLM calls: the agent is a regex plus the dummy tool, so the only Groq call per turn is the supervisor's routing. The agent's work is not in Langfuse (it does not use `graph_runner`), so latency is visible only in the logs above.
13. **Compatibility.** The `history` data part and `customer_id`/`thread_id` metadata are still our own conventions (see `v0.1.0/a2a-explained.md` section 9). A generic A2A client could start a push task on this agent but would not know them. The agent card version is now `0.3.0` for all three agents because the card constant is shared.

## 8. Try variations yourself

| Change | Expected effect |
|---|---|
| Ask "where is my parcel?" without an order id | `ASK_FOR_ORDER_REPLY` ("Which order should I track?"): the task still completes via push, within milliseconds |
| Set `shipment.carrier_delay_s` to 1 in `config/base.yaml` | Pushes may arrive before `register()`; check that the answer is still delivered (the early-terminal buffer) |
| Stop the CLI right after the placeholder | The agent logs `Error sending push-notification`; the answer is lost (item 5) |
| Send a wrong token with `curl -X POST :8010/a2a/notify -H 'X-A2A-Notification-Token: x' -d '{}'` | 401 and `status=rejected reason=bad_token` |
| Make `order_management` advertise push (`PUSH_AGENTS` in `cards.py`) | Every order question becomes placeholder + update (item 10) |
