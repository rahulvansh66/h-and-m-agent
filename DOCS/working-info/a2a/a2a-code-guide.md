# A2A code guide: one picture, one request, and async from zero

Read this top to bottom once,
then keep it open beside the code. For a walkthrough of real logs and wire messages see
[a2a-explained.md](a2a-explained.md); this guide is about **the code**.

Contents:

1. [The big picture](#1-the-big-picture)
2. [What each file does](#2-what-each-file-does)
3. [Follow one message through the code](#3-follow-one-message-through-the-code)
4. [Async from zero](#4-async-from-zero)
5. [Where async appears in this repo](#5-where-async-appears-in-this-repo)
6. [Walkthrough of each file](#6-walkthrough-of-each-file)
7. [Glossary](#7-glossary)
8. [Questions beginners ask](#8-questions-beginners-ask)
9. [Suggested reading order](#9-suggested-reading-order)

---

## 1. The big picture

There are **two kinds of process**. Everything in the A2A folder exists to let them talk.

```
 PROCESS 1: the chat (supervisor)                     PROCESS 2 and 3: agent services (one each)
 ------------------------------                       ----------------------------------------
 python -m hm_agent                                   python -m hm_agent.order_management.a2a_service
                                                      python -m hm_agent.customer_support.a2a_service

 LangGraph "supervisor graph"                         uvicorn web server (listens on :8001 / :8002)
   supervisor_node  (picks an agent)                    |
        |                                                |-- GET  /.well-known/agent-card.json  -> the Agent Card
        v                                                '-- POST /  (JSON-RPC "SendMessage")
   remote_node.node  (a graph node)                            |
        |                                                      v
        v                                                RunnerExecutor.execute()   (server.py)
   client.call_remote_agent()  --- HTTP request --->           |
   client._send()              <--- HTTP response ---          v
                                                          graph_runner -> the real LangGraph agent
                                                          (LLM calls, MCP tools, RAG ...)
```

In plain words:

- The **supervisor** decides *which* agent should handle the customer's message (an LLM call, local, not A2A).
- The **client side** (`remote_node.py` + `client.py`) packs the message into an HTTP request and sends it.
- The **server side** (`server.py` + each agent's `a2a_service.py`) receives it, runs the real agent, and sends the reply back.
- **A2A is only the hop between the two.** Inside an agent, nothing about A2A is visible: it is just a LangGraph graph.
- The **Agent Card** (`cards.py`) is a small JSON "business card" each agent serves so the client knows where and how to talk to it.

A useful mental model: **A2A is like a restaurant order.** The supervisor is the waiter, the agent is the kitchen,
the HTTP request is the order slip, the Task is the ticket that moves from "submitted" to "completed",
and the Artifact is the plate that comes back.

---

## 2. What each file does

| File | Side | One-line job |
|---|---|---|
| [graph.py](../../../src/hm_agent/graph.py) | client | Builds the supervisor graph; adds one `remote_node` per agent. |
| [a2a/remote_node.py](../../../src/hm_agent/a2a/remote_node.py) | client | Graph node: collects state, calls the remote agent, turns the reply into an `AIMessage`. Catches failures. |
| [a2a/client.py](../../../src/hm_agent/a2a/client.py) | client | Does the actual A2A call: fetch card, send message, read the reply. **Contains the async code.** |
| [a2a/wire_log.py](../../../src/hm_agent/a2a/wire_log.py) | client | Optional debug log of raw HTTP bodies (only when `A2A_WIRE_LOG` is set). |
| [a2a/cards.py](../../../src/hm_agent/a2a/cards.py) | server | Describes each agent (name, skills, URL) as an Agent Card. |
| [a2a/server.py](../../../src/hm_agent/a2a/server.py) | server | Generic wrapper: turns any "text in, text out" function into an A2A server. **Contains the async code.** |
| [order_management/a2a_service.py](../../../src/hm_agent/order_management/a2a_service.py) | server | Entry point (`main()`) for the order agent process. |
| [customer_support/a2a_service.py](../../../src/hm_agent/customer_support/a2a_service.py) | server | Entry point (`main()`) for the support agent process. |
| [scripts/a2a/run_agents.py](../../../scripts/a2a/run_agents.py) | tooling | Starts both agent processes and waits until their cards answer. |
| [core/settings.py](../../../src/hm_agent/core/settings.py) + [config/base.yaml](../../../config/base.yaml) | both | Ports, URLs, timeout, history size. `agent_url(agent)` and `agent_port(agent)` live here. |

**The one idea that removes most confusion:** `server.py` is written *once* and reused by every agent.
The two `a2a_service.py` files are tiny; they only choose *which* agent graph and card to plug into it:

```
order_management/a2a_service.py main():   build_a2a_app( card , name , graph_runner( build_order_agent() ) )  -> uvicorn.run
customer_support/a2a_service.py main():   build_a2a_app( card , name , graph_runner( build_customer_support_agent() ) )  -> uvicorn.run
                                                |
                                                '-- both call the SAME function in server.py
```

---

## 3. Follow one message through the code

The customer types: *"What was the last item I bought?"* Here is every function that runs, in order.
Keep this list next to the code; each line says which file to open.

### Client side (the chat process)

| # | Where | What happens |
|---|---|---|
| 1 | `supervisor/agent.py` `supervisor_node` | LLM picks `order_management`, stored in `state["next_agent"]`. |
| 2 | `supervisor/agent.py` `route_to_agent` | Conditional edge sends the graph to the `order_management` node. |
| 3 | `a2a/remote_node.py` `node(state, config)` | Reads `customer_id`/`thread_id` from `config`, takes the last message as `latest`, builds `history` from the previous (up to `A2A_HISTORY`) messages. |
| 4 | `a2a/remote_node.py` | Calls `call_remote_agent(agent_url(agent), latest, history, customer_id, thread_id, A2A_TIMEOUT_S)`. |
| 5 | `a2a/client.py` `call_remote_agent` | A normal (sync) function. Its only line is `asyncio.run(_send(...))`; this is the **sync-to-async bridge** (section 5). |
| 6 | `a2a/client.py` `_send` | Opens an `httpx.AsyncClient`, asks the SDK to build an A2A client from the URL. The SDK **fetches the Agent Card** (`GET /.well-known/agent-card.json`). |
| 7 | `a2a/client.py` `_send` | Builds a `SendMessageRequest` with 2 parts (text + `{"history": ...}`) and metadata (`customer_id`, `thread_id`); `contextId` = thread id. |
| 8 | `a2a/client.py` `_send` | `async for response in client.send_message(request)`: sends `POST /` (JSON-RPC `SendMessage`) and **waits** for the answer. |

### Server side (the agent process), while the client waits

| # | Where | What happens |
|---|---|---|
| 9 | `uvicorn` + `DefaultRequestHandlerV2` (library code) | Receives the POST, creates a task id, calls our executor. |
| 10 | `a2a/server.py` `RunnerExecutor.execute` | Publishes the Task (`submitted`), then `start_work()` (`working`). |
| 11 | `a2a/server.py` `execute` | Pulls `text`, `history`, `customer_id`, `thread_id` back out of the request. |
| 12 | `a2a/server.py` `execute` | `await asyncio.to_thread(self._runner, ...)`: runs the **blocking** agent in a worker thread (section 5). |
| 13 | `a2a/server.py` `graph_runner` -> `run(...)` | Rebuilds LangChain messages, calls `graph.invoke(...)` with Langfuse tracing, returns the last message's text. |
| 14 | `a2a/server.py` `execute` | `add_artifact([text part], name="reply")` then `complete()`. The library sends the finished Task as the HTTP response. |

### Back on the client side

| # | Where | What happens |
|---|---|---|
| 15 | `a2a/client.py` `_send` | The `async for` receives the Task. Checks `state == TASK_STATE_COMPLETED`, else raises `RuntimeError`. |
| 16 | `a2a/client.py` `_send` | Joins the text of all artifacts and returns it. `asyncio.run` returns it to `call_remote_agent`, which returns it to the node. |
| 17 | `a2a/remote_node.py` `node` | Wraps it as `AIMessage(reply, name="order_management")` and returns `{"messages": [...]}`. The graph ends and the answer is shown. |

If anything in steps 4-16 raises (agent down, timeout, failed task), `remote_node.node` catches it,
logs `status=remote_failed`, and returns the apology text, so the chat never crashes.

### The same flow as a timeline

```
chat process                         agent process
------------                         -------------
node() -------------------------+
call_remote_agent()             |
  asyncio.run( _send() ) ...    |   (everything below happens while this line waits)
     GET card  ----------------------->  serve card
     POST SendMessage  --------------->  execute():
                                            publish "submitted", "working"
                                            to_thread( graph.invoke ... )  <- LLM + tools, takes seconds
                                            add_artifact(reply), complete()
     <-------------- Task(completed) --
  returns reply text            |
node() returns AIMessage <------+
```

---

## 4. Async from zero

You only need five ideas. Each has a tiny example, then where it appears here.

### 4.1 The problem async solves: waiting

Most of what this app does is **waiting**: waiting for Groq to answer, for Postgres, for another
server's HTTP response. While a normal function waits, the whole program sits idle.

Async lets one program say: *"I am waiting on this; go and do something else meanwhile."*
It is not parallel computing (nothing runs at the same instant on one event loop); it is
**taking turns efficiently**.

### 4.2 `async def` and `await`

```python
def normal():                # normal function
    return 1                 # calling normal() runs it right away and gives 1

async def coroutine():       # "coroutine function"
    return 1                 # calling coroutine() does NOT run it; it gives a coroutine object
```

A coroutine is a **paused recipe**. Calling `coroutine()` only creates the recipe. Something must run it.

Inside an `async def`, `await something()` means:

> "Start this, and **pause me here** until it finishes. While I'm paused, the event loop may run other work."

```python
async def fetch():
    reply = await http.get(url)      # pause here until the network answers
    return reply.text                # then continue
```

Rules to remember:

1. `await` is only allowed inside an `async def`.
2. You can only `await` something awaitable (a coroutine, task, or future). Forgetting `await` is the classic bug:
   `x = fetch()` gives you a coroutine object, not the result (Python warns "coroutine was never awaited").
3. An `async def` is "contagious": to `await` it, the caller must itself be `async def`... until you reach the top.

### 4.3 The event loop

The **event loop** is the manager that runs coroutines. It keeps a list of paused coroutines, and when one
can continue (its network reply arrived), it resumes it. Without a running event loop, no `await` can work.

```
event loop:  run A until A hits an await  ->  run B until B hits an await  ->  A's data arrived, resume A  -> ...
```

The key consequence, and the most important thing in this guide:

> **Everything on the loop takes turns. If one piece of code blocks (does slow, non-async work),
> the whole loop freezes and nothing else can run.**

### 4.4 How you start async code from normal code: `asyncio.run`

The top of any program is normal (sync) code. To enter the async world you call:

```python
result = asyncio.run(fetch())    # create a loop, run fetch() to completion, close the loop, return result
```

`asyncio.run` **blocks** the caller until the coroutine is done, so from the outside it looks like
an ordinary function call. This is exactly how `client.py` uses it.

### 4.5 `async with` and `async for`

Same as `with` / `for`, but the setup, teardown, or "get next item" steps can themselves wait:

```python
async with httpx.AsyncClient() as http:    # opening/closing connections can wait (network)
    ...

async for item in source():                # getting each next item can wait (e.g. next chunk from a server)
    ...
```

`async for` iterates over an **async generator**: something that produces values over time, pausing between them.

### 4.6 Running blocking code safely: `asyncio.to_thread`

Some libraries (LangChain `graph.invoke`, the MCP wrapper here) are **not async**. They block the thread
while they work. If you called them directly inside `async def`, you would freeze the event loop (4.3).

```python
reply = await asyncio.to_thread(blocking_function, arg1, arg2)
```

This runs `blocking_function` in a **separate worker thread** and `await`s it. The loop stays free
while the thread works. It is the standard way to use sync code from async code.

### Cheat sheet

| You see | It means |
|---|---|
| `async def f()` | `f()` returns a paused recipe; must be awaited or run |
| `await f()` | Pause here until `f()` finishes; loop may do other work meanwhile |
| `asyncio.run(f())` | From sync code: run `f()` start to finish and give me the result |
| `async with x as y` | Open/close `x` where opening/closing can wait |
| `async for a in x` | Loop over values that arrive over time |
| `await asyncio.to_thread(g, ...)` | Run blocking `g` in a thread without freezing the loop |

---

## 5. Where async appears in this repo

Only **three places** use async. Everything else is plain synchronous code.

### 5.1 Client: `call_remote_agent` -> `asyncio.run(_send(...))` ([client.py:87](../../../src/hm_agent/a2a/client.py#L87))

```python
def call_remote_agent(...) -> str:          # normal function, so the sync graph node can call it
    return asyncio.run(_send(...))          # start a loop, run _send to the end, return its string
```

**Why this shape?** The supervisor graph is synchronous (`graph.invoke(...)` in `cli.py`), and so is the node.
But the A2A SDK's client is async-only. `asyncio.run` is the bridge: the sync world calls a sync function,
which privately enters the async world and comes back with a plain string.

For the caller, nothing is "async" at all, it just waits for the string. A new event loop is created and
destroyed on every call, which is fine for this low-volume, one-call-at-a-time use.

`asyncio.run` must **not** be called from inside an already-running loop (it raises an error). Here that
cannot happen because the supervisor node runs in plain sync code.

### 5.2 Inside `_send` ([client.py:23-64](../../../src/hm_agent/a2a/client.py#L23-L64))

Read it line by line with these labels:

```python
async def _send(...):
    async with httpx.AsyncClient(timeout=..., event_hooks=wire_log_hooks()) as http:   # (a)
        factory = ClientFactory(ClientConfig(streaming=False, httpx_client=http))
        client = await factory.create_from_url(url)                                    # (b)
        request = SendMessageRequest(...)                                              # plain, no waiting
        task = None
        async for response in client.send_message(request):                            # (c)
            if response.HasField("task"):
                task = response.task
        await client.close()                                                           # (d)
    ...                                                                                # plain code again
```

| Label | What it does | Why `await`/`async` |
|---|---|---|
| (a) `async with` | Opens an HTTP connection pool, guarantees it is closed at the end. | Closing sockets can wait. |
| (b) `await create_from_url` | Fetches the Agent Card over HTTP, builds the client. | Network call. |
| (c) `async for` | Sends the request and receives the response(s). | With `streaming=False` there is typically **one** item (the final Task); the loop form exists because the SDK uses the same API for streaming replies that arrive in pieces. We keep the last `task` we see. |
| (d) `await client.close()` | Cleanly shuts down the client. | Network cleanup. |

Everything after the `async with` block (state check, building the return string) is ordinary code.

### 5.3 The wire-log hooks ([wire_log.py:44-71](../../../src/hm_agent/a2a/wire_log.py#L44-L71))

httpx lets you register functions that run on every request and response ("event hooks").
With `httpx.AsyncClient` those hooks must be `async def`, which is why `_log_request` and `_log_response` are:

```python
async def _log_response(response):
    await response.aread()        # download the body now; before this, .text is not available
    _write_wire(..., response.text, ...)
```

The `await response.aread()` is **necessary**: httpx streams bodies lazily, so the body has not been
read yet when the hook runs. Reading it inside the hook also means the later code can still use it.

`_write_wire` itself is a normal function that appends to a file. Blocking file writes inside async code are
usually discouraged, but this is a debug-only log, so the tiny cost is acceptable.

`wire_log_hooks()` returns `{}` when `A2A_WIRE_LOG` is unset, so with logging off nothing is registered at all.

### 5.4 Server: `async def execute` and `asyncio.to_thread` ([server.py:109-148](../../../src/hm_agent/a2a/server.py#L109-L148))

The server (uvicorn + Starlette + the A2A SDK) is **async from top to bottom**: one event loop serves every
incoming request. So `execute` is `async def`, and every interaction with the task queue is awaited:

```python
await event_queue.enqueue_event(...)     # publish the Task
await updater.start_work()               # publish state=working
...
reply = await asyncio.to_thread(self._runner, text, history, customer_id, thread_id)   # <- the key line
...
await updater.add_artifact(...)          # publish the reply
await updater.complete()                 # publish state=completed
```

**The key line, explained.** `self._runner` ends up calling `graph.invoke(...)`, which blocks for many seconds
(LLM calls, SQL, embeddings). If we wrote `reply = self._runner(...)` directly:

```
event loop thread:  [ execute() calls graph.invoke ......... blocked 15 s ......... ]
                    -> the card endpoint can't answer, the launcher's readiness check times out,
                       a second request waits in line, even cancel() can't run.
```

With `asyncio.to_thread`:

```
event loop thread:  [ execute() hands work to a thread ]  [ serves other requests / card ]  [ resumes execute() ]
worker thread:                                            [ graph.invoke ......... 15 s ......... ]
```

The loop stays responsive while the worker thread does the slow part.

**Why the `try/except` is around the `await`:** if the thread's function raises, the exception is re-raised
at the `await` line, so a normal `except Exception` catches it. We then publish `failed` instead of crashing.

### 5.5 Everything else is sync

| Code | Sync or async | Note |
|---|---|---|
| Supervisor graph, `remote_node.node`, `cli.py` | sync | Calls `call_remote_agent` and just waits. |
| `graph_runner.run` | sync | Runs in a worker thread via `to_thread`. |
| Agent LangGraph graphs, RAG pipeline | sync | Same. |
| `order_management/mcp_client.py` | sync wrapper | Has its own `asyncio.run` for the MCP SDK, the same bridge pattern as 5.1. |
| `run_agents.py` launcher | sync | Uses `subprocess` and `time.sleep`, no asyncio at all. |

**Pattern to remember:** the repo keeps its business logic synchronous and confines async to thin
"adapters" at the edges where a library demands it (`asyncio.run` going in, `to_thread` going out).

---

## 6. Walkthrough of each file

### 6.1 `cards.py`: the Agent Card

- `_AGENT_SPECS` is a dict `{agent name: (description, [skills])}`; each skill is a tuple of
  `(id, name, description, examples)`.
- `build_agent_card(agent)` turns one entry into an `AgentCard` protobuf object:
  `supported_interfaces` holds the URL (from `agent_url`) and says "I speak JSON-RPC"; `capabilities`
  says no streaming and no push notifications; `skills` lists what it can do.
- The server publishes it at `/.well-known/agent-card.json`; the client reads it first.
- To add or change a skill: edit `_AGENT_SPECS` and bump `CARD_VERSION`.

### 6.2 `server.py`: the generic server

It has three layers. Read them bottom-up if easier:

1. **`graph_runner(graph, agent)` -> `Runner`**: a function-factory (a function that returns a function).
   It captures `graph` and `agent`, and returns `run(text, history, customer_id, thread_id) -> str`.
   `run` rebuilds the chat history as LangChain messages, invokes the graph, flushes Langfuse, returns
   the final text. `Runner` is just the type name for such a function.
2. **`RunnerExecutor`**: a class implementing the SDK's `AgentExecutor` interface. The SDK calls
   `execute()` for each incoming message and `cancel()` if asked. `execute` is the task lifecycle
   (section 3 steps 10-14).
3. **`build_a2a_app(card, agent, runner)`**: wires the executor into the SDK's request handler, adds two
   routes (the card route and the JSON-RPC route at `/`), and returns a Starlette app that `uvicorn.run` serves.

Why a `Runner` abstraction? So `server.py` knows nothing about orders or support: any "text in, text out"
function becomes an A2A agent.

Things worth knowing:

- `InMemoryTaskStore()` keeps tasks in RAM: lost on restart (fine for this project).
- `context.task_id` / `context.context_id` are filled in by the SDK. `context_id` is what the client sent as
  `contextId` (the thread id).
- `new_task_from_user_message(...)` is only enqueued when `context.current_task is None`, because the first
  event of any task must be the Task itself.

### 6.3 `client.py`: see section 5.1-5.2

Also note: `message_id=f"msg-{time.time_ns()}"` makes a unique id per call; `context_id=thread_id` groups
all turns of one conversation; the history rides along as a **data part** (our own convention, not part
of the A2A standard).

### 6.4 `remote_node.py`: the bridge into LangGraph

`make_remote_agent_node(agent)` is another function-factory: it returns `node`, a closure that remembers
which `agent` it is for. `graph.py` calls it once per agent, so you get two nodes from one function.

Why the history window `messages[:-1][-A2A_HISTORY:]`? `[:-1]` drops the latest message (sent separately as
text), `[-A2A_HISTORY:]` keeps only the last N of the rest to bound prompt size. The `name=m.name` is
how the supervisor remembers which agent answered, so follow-ups can go to the same agent.

### 6.5 `<agent>/a2a_service.py`: the entry points

Run with `python -m hm_agent.order_management.a2a_service`. Each `main()`:
loads `.env`, configures logging, builds the agent graph, builds the card, builds the app, runs uvicorn on
that agent's port. The launcher script starts these as subprocesses.

### 6.6 `run_agents.py`: the launcher

Starts each service with `subprocess.Popen`, then polls each card URL until it answers (an agent is "ready"
when it can serve its card), then idles until Ctrl+C and terminates the children. This is where you get the
"All agents ready" message. No async here: plain loops and `time.sleep`.

---

## 7. Glossary

| Term | Meaning here |
|---|---|
| **A2A** | Agent2Agent protocol: a standard way for one agent to hand a task to another over HTTP. |
| **Agent Card** | JSON "business card" at `/.well-known/agent-card.json`: name, URL, skills, capabilities. |
| **Task** | The unit of work on the server. Moves `submitted -> working -> completed` (or `failed`/`canceled`). |
| **Artifact** | The *output* of a task. Our reply text is one artifact named `reply`. |
| **Message / Part** | What the client sends. A message has parts: a text part (the question) and a data part (the history). |
| **contextId** | Groups all tasks of one conversation. We set it to the thread id. |
| **JSON-RPC** | The request format: `{"method": "SendMessage", "params": ..., "id": ...}` over HTTP POST. |
| **Runner** | Our name for "a plain function: text, history, ids in -> reply text out". |
| **Executor** | The SDK interface our `RunnerExecutor` implements; it runs one task. |
| **Event queue / TaskUpdater** | How the executor reports progress (`start_work`, `add_artifact`, `complete`) to the SDK. |
| **uvicorn / Starlette** | The web server / web framework serving the agent. uvicorn owns the event loop. |
| **Coroutine** | The paused recipe an `async def` call returns. |
| **Event loop** | The manager that runs coroutines and resumes them when what they await is ready. |
| **Blocking** | Doing slow work without ever yielding control; on the event loop this freezes everything. |
| **Closure / factory** | A function that returns a function remembering the arguments it was created with (`graph_runner`, `make_remote_agent_node`). |

---

## 8. Questions beginners ask

**Why is the client `_send` async but `call_remote_agent` not?**
The A2A SDK client is async-only, but the rest of the supervisor is sync. `call_remote_agent` hides the async
part behind `asyncio.run` so the sync graph node can use it as a normal function.

**Why not make everything async?**
LangGraph invoke, the RAG pipeline and the MCP wrapper are sync. Converting them would be a large change
for no benefit at this scale; the two adapter points (`asyncio.run`, `to_thread`) are enough.

**If I call `asyncio.run` for every request, is that slow?**
Creating a loop costs far less than the multi-second LLM calls. It would matter at high request rates.

**What happens if two chat requests hit an agent at once?**
The loop accepts both, and each `execute` hands its work to a worker thread, so they run concurrently
(limited by the default thread pool and by Groq's rate limit).

**Why does the server wrap the agent in `to_thread` instead of just defining `execute` as a normal function?**
The SDK calls `execute` on its event loop and expects a coroutine. `to_thread` is how a coroutine safely
delegates to blocking code.

**Why is there an `async for` if there's only one reply?**
`client.send_message` is the SDK's one API for both blocking and streaming; it always returns an async
iterator. We iterate it and keep the Task.

**The HTTP status was 200 but the task failed. Is that a bug?**
No. JSON-RPC errors and failed tasks travel inside a 200 response. That is why `_send` checks
`task.status.state` instead of the HTTP status.

**Where do `customer_id` and `thread_id` get used on the server?**
The client puts them in `metadata`; `execute` reads them from `context.metadata`; `graph_runner.run` puts
them into the graph `config["configurable"]` (the MCP tool interceptor reads `customer_id` there) and into
Langfuse session/user fields.

**How do I debug what is really sent?**
`export A2A_WIRE_LOG=some/file.jsonl` before running the chat. See [commands.md](commands.md). It records
every HTTP body; do not leave it on in production (it contains customer text).

---

## 9. Suggested reading order

Read in the order data flows, not alphabetically. Allow about 10 minutes per step, with the section 3 table open.

1. [graph.py](../../../src/hm_agent/graph.py): how a remote agent becomes just another node (about 25 lines of real code).
2. [a2a/remote_node.py](../../../src/hm_agent/a2a/remote_node.py): the node that starts the call.
3. [a2a/client.py](../../../src/hm_agent/a2a/client.py): sync wrapper first (bottom), then `_send` (top); use section 5.2 as a map.
4. [a2a/cards.py](../../../src/hm_agent/a2a/cards.py): what the client discovers.
5. [order_management/a2a_service.py](../../../src/hm_agent/order_management/a2a_service.py): how a server process is assembled (short).
6. [a2a/server.py](../../../src/hm_agent/a2a/server.py): `build_a2a_app` first, then `RunnerExecutor.execute`, then `graph_runner`.
7. [a2a/wire_log.py](../../../src/hm_agent/a2a/wire_log.py) and [scripts/a2a/run_agents.py](../../../scripts/a2a/run_agents.py): supporting pieces, skip until the rest makes sense.

A good exercise once you finish: set `A2A_WIRE_LOG`, send one message, and match each line of the resulting
`.jsonl` to a step in the section 3 tables.
