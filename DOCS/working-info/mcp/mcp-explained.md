# MCP explained from a real session log

This guide uses one real session of this project (two questions, one chat thread) to explain
how MCP and JSON-RPC work, who plays which role, what every field means, and what to watch
in production.

Files in this folder:

| File | What it is |
|---|---|
| `session-console.log` | What the terminal printed: our app logs plus the MCP server's stderr logs. |
| `mcp-wire.log` | The raw JSON-RPC lines exchanged between client and server, one per line, with time, server process id (`pid`) and direction (`C->S` client to server, `S->C` server to client). |

The session was run with:

```
uv run python -m hm_agent        # then typed the two questions below, then `exit`
```

| Turn | You asked | Assistant answered |
|---|---|---|
| 1 | What was the last item I bought? | Your most recent purchase was a Berry Tee. |
| 2 | How many purchases have I made? | You've bought a total of 9 items. |

thread_id (also the Langfuse session id): `f21654c3-862b-4720-8112-cee39dae0239`, customer `cust001`.

> How the wire log was captured: the code does not log protocol traffic, so a throwaway
> pass-through wrapper (not part of the repo) was placed between client and server. It copies
> each line to `mcp-wire.log` and forwards it unchanged. It adds a little startup time, so treat
> the timings below as slightly pessimistic.

---

## 1. How MCP works

**MCP (Model Context Protocol)** is a standard way for an AI application to discover and call
capabilities that live in a *separate* program. Without it, every app needs custom glue for
every tool provider (N apps x M tools). With it, a tool provider writes one MCP server and any
MCP-aware app can use it.

A server can offer three kinds of things (called primitives):

| Primitive | Meaning | Used here? |
|---|---|---|
| **Tools** | Functions the model may ask to run (our two SQL lookups). | Yes |
| **Resources** | Read-only data the app can load (files, rows). | No (declared empty) |
| **Prompts** | Reusable prompt templates. | No (declared empty) |

### The lifecycle (what you see in `mcp-wire.log`)

Every connection goes through the same steps:

```
Client                                         Server
  | -- initialize (my version, my abilities) -->|   1. handshake
  |<-- result (agreed version, your abilities) -|
  | -- notifications/initialized -------------->|   2. "ready" (no reply expected)
  | -- tools/list ----------------------------->|   3. discovery: what can you do?
  |<-- result: tool names, descriptions, schemas|
  | -- tools/call {name, arguments} ----------->|   4. use: run a tool
  |<-- result: content / isError ---------------|
  | (close stdin)                               |   5. shutdown (stdio: process exits)
```

### Transports: how the bytes travel

MCP defines the message format; a *transport* carries it.

* **stdio** (what we use): the client starts the server as a child process, writes JSON lines
  to the server's stdin and reads JSON lines from its stdout. No network, no port, no auth.
  That is why **stdout is reserved for the protocol** and logs must go to stderr
  (see `main()` in `mcp_servers/order_management.py`).
* **Streamable HTTP**: the server is a long-running web service; messages are HTTP POSTs. This
  is the transport for shared, remote, multi-user servers.

---

## 2. What is RPC in JSON-RPC?

**RPC = Remote Procedure Call**: calling a function that runs in another process (or machine)
as if it were a local function. You name the function and give arguments; you get a return
value back. The caller does not care how the bytes travel.

**JSON-RPC 2.0** is a tiny, transport-independent RPC convention that says: *encode the call
and the result as JSON, in exactly this shape.* That is the whole spec (about 3 message types).
MCP is built on top of it.

Our `tools/call` is literally "call the remote function `get_last_bought_product` with
`customer_id="cust001"`":

```json
{"jsonrpc":"2.0","id":1,"method":"tools/call",
 "params":{"name":"get_last_bought_product","arguments":{"customer_id":"cust001"}}}
```

## 3. Why not just JSON?

JSON is only a *data format*; it says nothing about how two programs should converse. With
"just JSON" you would have to invent, document and maintain answers to:

| Question | JSON-RPC answer |
|---|---|
| Which operation is this? | `method` field |
| How do I match a reply to my request when several are in flight? | `id` is echoed in the response |
| How do I say "fire and forget"? | A message with no `id` (a notification) |
| How do I report failure in a machine-readable way? | Standard `error` object with `code` and `message` |
| How do I know which protocol/version this is? | `"jsonrpc": "2.0"` |

Every team would invent a slightly different shape, so a generic client could not talk to a
generic server. A shared envelope is what lets any MCP client use any MCP server. JSON-RPC
also needs no extra framework, is human-readable (you can read the log above), and works over
any byte stream, which is why it fits stdio as well as HTTP.

---

## 4. Host, client and server in our case

MCP names three roles. Note that the **LLM is not one of them**: it never speaks MCP.

```
+----------------------------- HOST: the hm_agent process -----------------------------+
|  cli.py -> LangGraph supervisor -> order_management agent -> LLM (Groq)               |
|                                         |                                             |
|                                tool call chosen by the LLM                            |
|                                         v                                             |
|                      CLIENT: MultiServerMCPClient (mcp_client.py)                     |
|                      + inject_customer_id interceptor                                 |
+-----------------------------------------|---------------------------------------------+
                                          |  JSON-RPC lines over stdio pipes
                                          v
                  SERVER: python -m hm_agent.mcp_servers.order_management
                          (FastMCP, separate process) -> Neon Postgres
```

| Role | In this project | Responsibility |
|---|---|---|
| **Host** | The `hm_agent` application (CLI + LangGraph + the LLM) | Owns the user conversation and the model; decides *whether* a tool call is allowed and with *which* arguments. |
| **Client** | `MultiServerMCPClient` from `langchain-mcp-adapters`, configured in `mcp_client.py` | One client per server connection. Speaks the protocol, turns MCP tools into LangChain tools, runs our interceptor. |
| **Server** | `mcp_servers/order_management.py` (`FastMCP("order-management")`) | Declares tools, runs them, returns results. Knows nothing about the LLM or the conversation. |

How the pieces cooperate for one question:

1. The LLM sees the tool list (names, descriptions, JSON schemas from `tools/list`) and answers
   with a tool call such as `get_last_bought_product` with no arguments.
2. LangChain hands that to the client. The interceptor `inject_customer_id` **adds**
   `customer_id` from the run config (the model never supplies it).
3. The client sends `tools/call` to the server; the server queries Neon and replies.
4. The result goes back to the LLM as a tool message, which writes the final sentence.

---

## 5. What is passed in JSON-RPC, and what each field means

Three message shapes exist. All carry `"jsonrpc": "2.0"`.

### 5.1 Request: expects a reply

| Field | Meaning |
|---|---|
| `jsonrpc` | Always `"2.0"`; identifies the envelope format. |
| `id` | Number or string chosen by the **sender**. The reply repeats it so replies can be matched, even out of order. Per connection, ids start at 0 here. |
| `method` | The operation, e.g. `initialize`, `tools/list`, `tools/call`. |
| `params` | Arguments of the method (object); shape depends on the method. Optional. |

### 5.2 Response: the reply to a request

Has the same `id` and **exactly one** of:

| Field | Meaning |
|---|---|
| `result` | Success payload; shape depends on the method. |
| `error` | Protocol-level failure: `{ "code": int, "message": str, "data"?: any }`. |

Standard error codes: `-32700` parse error, `-32600` invalid request, `-32601` method not
found, `-32602` invalid params, `-32603` internal error.

### 5.3 Notification: fire and forget

Like a request but **no `id`**, so no reply is ever sent. Example:
`{"jsonrpc":"2.0","method":"notifications/initialized"}`.

### 5.4 MCP methods seen in our log

**`initialize` request** (client to server, first message):

```json
{"jsonrpc":"2.0","id":0,"method":"initialize",
 "params":{"protocolVersion":"2025-11-25","capabilities":{},
           "clientInfo":{"name":"mcp","version":"0.1.0"}}}
```

| Field | Meaning |
|---|---|
| `protocolVersion` | The newest MCP spec version the client supports. The server replies with the version it will actually use; if they cannot agree, the connection fails. |
| `capabilities` | Optional features the client offers the server (e.g. sampling, roots). `{}` = none. |
| `clientInfo` | Name and version of the client library (here the generic `mcp` Python SDK, not our app), useful for debugging. |

**`initialize` result** (server to client):

```json
{"jsonrpc":"2.0","id":0,"result":{
  "protocolVersion":"2025-11-25",
  "capabilities":{"experimental":{},"prompts":{"listChanged":false},
                  "resources":{"subscribe":false,"listChanged":false},
                  "tools":{"listChanged":false}},
  "serverInfo":{"name":"order-management","version":"1.30.0"}}}
```

| Field | Meaning |
|---|---|
| `capabilities.tools` | The server offers tools. `listChanged:false` = the tool list never changes at runtime, so the client will not get "list changed" notifications. |
| `capabilities.prompts` / `resources` | Declared but we define none. `subscribe:false` = clients cannot subscribe to resource updates. |
| `serverInfo.name` | `order-management`, the name we gave `FastMCP(...)`. |
| `serverInfo.version` | `1.30.0` is the **MCP SDK version**, not our app version, because we did not set one. |

**`notifications/initialized`**: the client's "handshake accepted, I'm ready". No reply.

**`tools/list` result**: one entry per tool:

| Field | Meaning |
|---|---|
| `name` | Unique tool id the model must use when calling. Taken from the Python function name. |
| `description` | Text the **LLM reads** to decide when to use the tool. FastMCP uses the whole docstring. |
| `inputSchema` | JSON Schema of the arguments (types, `required`). Generated from the type hints. |
| `outputSchema` | JSON Schema of `structuredContent`. Here `{"result": string}` because the function returns `str`. |

**`tools/call` request** `params`:

| Field | Meaning |
|---|---|
| `name` | Which tool to run. |
| `arguments` | The argument object; must satisfy `inputSchema`. Here `{"customer_id":"cust001"}`, **injected by our interceptor**, not by the model. |

**`tools/call` result**:

```json
{"jsonrpc":"2.0","id":1,"result":{
  "content":[{"type":"text","text":"Berry Tee"}],
  "structuredContent":{"result":"Berry Tee"},
  "isError":false}}
```

| Field | Meaning |
|---|---|
| `content` | List of content blocks meant for the model (`text`, `image`, `resource`...). Always present. |
| `structuredContent` | The same answer as machine-readable JSON matching `outputSchema`, for programs that don't want to parse text. |
| `isError` | `true` means the **tool ran but failed** (see section 8, two kinds of errors). |

---

## 6. Walkthrough of the actual log

Times are from `mcp-wire.log`; each `pid` is a separate server process.

| Time | pid | Message | Why it happened |
|---|---|---|---|
| 09:24:07 | 11748 | `initialize` | App start: `build_order_agent()` calls `load_order_tools()`, which spawns a server to learn the tool list. |
| 09:24:18 | 11748 | initialize result | **11 s** to answer: cold start of a fresh Python process (imports, first spawn); later spawns took ~2 to 3 s. |
| 09:24:18 | 11748 | `tools/list` and result | Tools discovered, schemas rewritten by `_hide_customer_id`. Process then exits. |
| 09:24:22 | 18968 | `initialize` (**new process**) | Turn 1. LLM chose `get_last_bought_product`; the client opens a brand-new connection for this single call. |
| 09:24:25 | 18968 | `tools/call` get_last_bought_product `{"customer_id":"cust001"}` | `customer_id` was injected by the interceptor. |
| 09:24:28 | 18968 | result `Berry Tee` | ~3 s: first Neon connection and query. |
| 09:24:28 | 18968 | `tools/list` (id 2) | Extra request made by the client SDK after the call (see section 8). |
| 09:24:34 | 13084 | `initialize`, `tools/call` get_purchase_count, result `9 items purchased.`, `tools/list` | Turn 2: **another new process**, same pattern. |

Things this table shows:

* **A server process per call.** `load_order_tools()` configures stdio with no persistent session,
  so every tool call = spawn process, handshake, call, list, exit. In this run that is about
  3 s (initialize) + 3 s (call, including the Neon connection) = ~6 s of MCP overhead per tool
  call, while the SQL itself takes milliseconds. The gaps between a result and the next
  `initialize` (4 to 6 s) cover supervisor routing and the model's tool-call decision.
* **ids restart at 0 per connection**, and the order is 0 initialize, 1 first real request, 2 ...
* `notifications/initialized` has no `id` and no reply.
* The server's own console lines (`Processing request of type CallToolRequest`, `tool=get_purchase_count count=9`)
  are stderr output of the server process, interleaved in `session-console.log`.
* The console also shows `Processing request of type ListToolsRequest` after every call. That is
  the server answering the trailing `tools/list`.

---

## 7. What to observe

### 7.1 In the logs

Check these every time you read a session log:

1. **Handshake succeeded**: `initialize` result has the same or an acceptable `protocolVersion`
   and `capabilities.tools`. No result means the server crashed at import (often a missing env var).
2. **`arguments` of every `tools/call`**: confirm `customer_id` equals the logged-in customer and
   was *not* chosen by the model. This is your authorization boundary.
3. **`isError`** in each result, and any `error` object: a failure that the model papered over
   with a friendly answer is invisible to the user but visible here.
4. **Request/response pairing**: every request `id` has exactly one response. A missing response
   means a hang or crash.
5. **Process count**: one `pid` per tool call is the cost pattern shown above. If you move to a
   persistent session or HTTP, expect one handshake per session instead.
6. **Latency per phase**: `initialize` (process start), `tools/call` (your SQL and network), and
   total gap between turns (LLM time). Log them separately so you know what to fix.
7. **stderr vs stdout**: only protocol JSON may appear on stdout of the server. A stray `print`
   corrupts the stream and the client fails with a parse error.
8. **Log hygiene**: server logs carry `service=`, `env=` and `commit=` but **no thread id**.
   Nothing in the server logs links a `tools/call` to the user's conversation (see 8.8).

### 7.2 In Langfuse

Open the session `f21654c3-862b-4720-8112-cee39dae0239` (Sessions page, filter by user
`cust001` or tag `supervisor`). This guide was written from the CLI log; I did not open the
Langfuse UI, so treat the list below as what to verify rather than a screenshot of it.

* **Two traces** named `hm-supervisor`, one per turn, grouped in the one session.
* In each trace: the supervisor routing step (its output `agent=order_management`), then the
  `order_management` agent, containing the **model generation** that chose the tool, a **tool span**
  (`get_last_bought_product` / `get_purchase_count`) and a second **generation** that wrote the answer.
* **Tool span input/output**: output should be `Berry Tee` / `9 items purchased.`. The input
  is recorded by LangChain *before* our interceptor runs, so it should **not** show `customer_id`
  even though the wire log does. Langfuse shows what the model asked for; the wire log shows what
  was actually sent. Compare the two when debugging.
* **Tool span duration** should be about 6 s (spawn + initialize + query + list) versus the SQL itself
  taking milliseconds. This is where the process-per-call cost shows.
* **Generation input**: open the first model generation and look at the tool definitions the model was
  given. The `description` includes the whole Google-style docstring (with `Args:` / `Returns:`
  blocks), which costs tokens on every call.
* **Token usage and latency** per generation, and whether the model called the tool once or looped.
* **Not in Langfuse**: the JSON-RPC messages themselves, `isError` details if the adapter converted
  them to exceptions, and anything from inside the server process. Use the wire/stderr logs for those.

---

## 8. Other things to know before running MCP in production

### 8.1 Two kinds of errors
* **Protocol error**: JSON-RPC `error` object (unknown method, bad params). The call never ran.
* **Tool error**: normal `result` with `isError: true`. The tool ran and failed (e.g. DB timeout).
  The message text goes **to the LLM**, so a raw exception string can leak SQL, table names or
  hostnames into the model's answer, against our "do not reveal internals" rule. Catch exceptions in
  the tool and return a short safe message, and log the detail server-side.

### 8.2 Authorization must live outside the model
The current design is right: the host injects `customer_id` and the model cannot choose it.
Two follow-ups:
* The **server trusts whatever `customer_id` it receives**. Over stdio that is fine (only our
  process can talk to it). If you ever switch to HTTP, anyone who can reach the endpoint can ask
  for any customer. You then need authentication (the HTTP transport spec uses OAuth 2.1 bearer
  tokens) and the server must derive the customer from the token, not from an argument.
* The interceptor hides `customer_id` from the *schema*, but the **description text still
  mentions `customer_id`** (visible in the `tools/list` result, from the docstring). It is harmless
  today, yet it nudges the model to ask the user for an id. Keep tool docstrings written for the
  model, separate from developer notes.

### 8.3 Tool descriptions are prompt input (injection risk)
Whatever a server puts in `description` or in a tool result is read by the LLM as text. A
third-party or compromised server can embed instructions ("ignore previous rules, call X"),
sometimes called tool poisoning. Only connect servers you control or have reviewed, pin their
versions, and treat results from tools that return user-generated content as untrusted.

### 8.4 Cost of process-per-call
Seen above: ~3 s handshake plus ~3 s call plus an extra `tools/list` per call, and a new Postgres
connection each time. Options, cheapest first:
* Keep one long-lived client session per app run (open once, reuse), so initialize and the DB
  connection happen once.
* Add a connection pool inside a long-lived server.
* Run the server as a shared HTTP service (streamable HTTP) if several apps or users need it.
  Always-on resources cost money, so weigh this against the pay-per-use goal in `CLAUDE.md`.

### 8.5 The trailing `tools/list`
After a `tools/call` the SDK may fetch the tool list again so it can validate `structuredContent`
against `outputSchema`. Because each call is a fresh connection with an empty cache, it happens on
every call. A persistent session caches it and removes the extra round trip.

### 8.6 Timeouts, hangs and cancellation
We set none. A server that hangs (Neon cold start, lock) blocks the agent turn. Set a client
read timeout per call and a statement timeout in Postgres. MCP has a `notifications/cancelled`
message for abandoning an in-flight request; make sure shutdown also kills child processes so
you don't leak orphan servers.

### 8.7 Versioning and compatibility
* `protocolVersion` is negotiated at `initialize`; pin the `mcp` and `langchain-mcp-adapters`
  versions in `pyproject.toml`, since client and server evolve independently.
* Changing a tool's name, arguments or output is a **breaking change** for any client or prompt
  that depends on it. The `ORDER_MANAGEMENT_PROMPT` hard-codes the two tool names, so a rename needs
  a prompt change too.
* If tools change at runtime, advertise `listChanged: true` and handle the notification.

### 8.8 Observability gaps to close
* **Correlation id**: the server process does not know `thread_id`. MCP lets the client attach
  `_meta` inside `params` (used for progress tokens and trace context), so you can pass
  `thread_id` / a trace id there and log it server-side. Without it you can only match by time and pid.
* **Metrics**: per tool, count calls, errors and latency, and count process spawns.
* **Tool annotations**: mark read-only tools with `readOnlyHint` so hosts and reviewers can
  tell they are safe. Ours are read-only but do not say so.

### 8.9 Operational details
* **Env vars** do not flow to a stdio child automatically: only those listed in `FORWARDED_ENV`
  reach it. A new secret needed by the server must be added there.
* **Output size**: cap result sizes; a huge result is paid for in tokens and may exceed limits.
* **Idempotency**: read-only tools are safe to retry; any future write tool needs an idempotency
  key, because clients and agents retry.
* **Windows**: the stdio client needs the Proactor event loop; `_make_sync` uses `asyncio.run`
  from worker threads for this reason. Keep it in mind if you change how tools are called.
* **Testing**: use the official MCP Inspector against the server for manual checks, and test the
  interceptor (it enforces the security rule) with a mocked server, never the real LLM.
* **Local first**: this session made two small Groq calls. Don't loop it for load testing on the free tier.

### 8.10 Quick production checklist
- [ ] Server returns safe, short error messages (no SQL or hostnames)
- [ ] `customer_id` can only come from authenticated context, never from the model
- [ ] Timeouts on client call and DB statement
- [ ] Persistent session or shared server instead of process-per-call
- [ ] Trace/thread id passed to the server and logged there
- [ ] `readOnlyHint` set; tool descriptions written for the model
- [ ] `mcp` and adapter versions pinned; tool contract changes reviewed
- [ ] Only trusted servers connected; results from untrusted sources treated as untrusted
