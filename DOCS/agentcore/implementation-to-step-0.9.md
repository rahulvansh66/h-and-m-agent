# AgentCore preparation: what is implemented (steps 0.1 to 0.9)

This document explains the local-only groundwork for running the assistant on AWS Bedrock AgentCore: who the
user is, how that identity travels between services, how each agent is protected, and how everything is packaged.
Nothing here needs an AWS account, and every behaviour is off by default, so the existing local workflow is unchanged.

Status at the time of writing: steps 0.1 to 0.8 are implemented and covered by tests (84 passing, LLM mocked).
Step 0.9 (Docker images) is written but **not yet built or run**; see [section 9](#9-docker-images-step-09).

## Contents

1. [The big picture](#1-the-big-picture)
2. [Settings and switches (0.1)](#2-settings-and-switches-01)
3. [JWT verification (0.2)](#3-jwt-verification-02)
4. [SigV4 signing of agent calls (0.3)](#4-sigv4-signing-of-agent-calls-03)
5. [Carrying the user token to the agents (0.4)](#5-carrying-the-user-token-to-the-agents-04)
6. [Customer scoping (0.5)](#6-customer-scoping-05)
7. [The AgentCore entrypoint (0.6)](#7-the-agentcore-entrypoint-06)
8. [Agent bind address and stream-only shipment (0.7, 0.8)](#8-agent-bind-address-and-stream-only-shipment-07-08)
9. [Docker images (0.9)](#9-docker-images-step-09)
10. [Switch reference](#10-switch-reference)
11. [How to run and test](#11-how-to-run-and-test)

---

## 1. The big picture

Two different questions are answered by two different mechanisms:

| Question | Mechanism | Where |
|---|---|---|
| Is this **service** allowed to call that agent? | AWS IAM, request signed with SigV4 | supervisor to agent hop |
| Which **customer** is asking, and may they see this data? | OIDC access token (JWT) with a `customer_id` claim | end user to supervisor to agents |

```
 end user ──(JWT)──▶ supervisor entrypoint ──(SigV4 + JWT in metadata)──▶ agent services ──▶ data (Neon, Qdrant)
                      runtime/entrypoint.py        a2a/client.py            a2a/server.py       filtered by customer_id
```

The principle behind the token work: **an agent never trusts a customer id because the supervisor said so**. It
re-verifies the user's token itself and uses the verified id. On top of that, every data query filters by that id
(step 0.5), so even a confused model cannot read another customer's rows.

Everything is switched on by deployment settings. Locally both switches are off and the code path is the same as before.

## 2. Settings and switches (0.1)

Files: [config/base.yaml](../../config/base.yaml), [core/settings.py](../../src/hm_agent/core/settings.py), [.env.example](../../.env.example).

Non-secret values live in `config/base.yaml`; the Auth0 tenant (which gives the token issuer) comes from `.env`.

```yaml
auth:
  enabled: false                  # verify user tokens (env AUTH_ENABLED overrides)
  audience: https://hm-agent/api
  customer_claim: https://hm-agent/customer_id   # namespaced claim carrying the customer id
  jwks_cache_ttl_s: 3600
aws:
  region: us-east-1
  sigv4_enabled: false            # env SIGV4_ENABLED overrides
```

Boolean switches are read **at call time**, because `.env` is loaded after `settings` is imported:

```python
# core/settings.py
def _flag(env_var: str, default: bool) -> bool:
    raw = os.environ.get(env_var)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in _TRUE_VALUES
```

`auth_enabled()`, `sigv4_enabled()` and `push_receiver_enabled()` all use it. `auth_issuer()` derives the issuer URL
(`https://<tenant>/`) from `AUTH0_TENANT`.

## 3. JWT verification (0.2)

Files: [auth/jwt_verifier.py](../../src/hm_agent/auth/jwt_verifier.py), [auth/\_\_init\_\_.py](../../src/hm_agent/auth/__init__.py),
tests in [tests/test_auth_jwt.py](../../tests/test_auth_jwt.py).

`verify_token(token)` is the single trust check for end-user identity. It checks signature (against the issuer's
JWKS), issuer, audience, expiry, and reads the namespaced `customer_id` claim. Any problem raises `AuthError` with a
short `reason` code that is safe to log.

Design points:

- **RS256 only.** `_ALGORITHMS = ["RS256"]` rules out `none` and HMAC key-confusion attacks.
- **JWKS cached in memory** with a TTL, and refreshed once when an unknown `kid` appears (key rotation).
- **The fetcher is injectable**, so tests serve a stub JWKS and never touch the network.
- **Logs carry only `status` and `reason`**, never the token or claim values.

```python
claims = jwt.decode(
    token, key.key,
    algorithms=_ALGORITHMS, audience=self._audience, issuer=self._issuer,
    leeway=_LEEWAY_S, options={"require": ["exp", "iss", "aud"]},
)
customer_id = claims.get(self._customer_claim)
if not isinstance(customer_id, str) or not customer_id:
    raise AuthError("missing_claim")
return VerifiedUser(customer_id=customer_id)
```

The module is shared by the supervisor entrypoint (7) and every agent service (5).

## 4. SigV4 signing of agent calls (0.3)

Files: [a2a/sigv4.py](../../src/hm_agent/a2a/sigv4.py), [a2a/client.py](../../src/hm_agent/a2a/client.py),
tests in [tests/test_a2a_sigv4.py](../../tests/test_a2a_sigv4.py).

AgentCore Runtime agent endpoints use IAM (SigV4) inbound auth, so each HTTP request from the supervisor (card fetch,
`SendMessage`, `GetTask`) must be signed. `SigV4Auth` is an `httpx.Auth` that signs with botocore:

```python
class SigV4Auth(httpx.Auth):
    requires_request_body = True          # the exact body bytes are signed

    def auth_flow(self, request):
        signed_headers = {n: v for n, v in request.headers.items() if n.lower() in _SIGNED_REQUEST_HEADERS}
        aws_request = AWSRequest(method=request.method, url=str(request.url),
                                 data=request.content, headers=signed_headers)
        BotocoreSigV4Auth(self._credentials_provider(), self._service, self._region).add_auth(aws_request)
        for name in _RESULT_HEADERS:                       # Authorization, X-Amz-Date, X-Amz-Security-Token
            if name in aws_request.headers:
                request.headers[name] = aws_request.headers[name]
        yield request
```

Only `content-type` (plus the host and the `x-amz-*` headers botocore adds) is signed. Proxies and HTTP clients add or
rewrite headers such as `Accept-Encoding`, which would break a signature that covered them.

The client turns signing on only when the switch is set; locally `auth=None` and nothing changes:

```python
async with httpx.AsyncClient(timeout=timeout_s, event_hooks=wire_log_hooks(), auth=request_auth()) as http:
```

`request_auth()` returns `SigV4Auth(settings.AWS_REGION)` when `sigv4_enabled()`, else `None`. Credentials come from the
standard AWS chain and are fetched per request, so rotated temporary credentials work.

**Not yet proven:** that Runtime accepts these signed requests, including streaming. That needs a deployed Runtime (step 1.5).

## 5. Carrying the user token to the agents (0.4)

Files: [turns.py](../../src/hm_agent/turns.py), [a2a/remote_node.py](../../src/hm_agent/a2a/remote_node.py),
[a2a/client.py](../../src/hm_agent/a2a/client.py), [a2a/server.py](../../src/hm_agent/a2a/server.py),
[a2a/wire_log.py](../../src/hm_agent/a2a/wire_log.py), tests in [tests/test_a2a_streaming.py](../../tests/test_a2a_streaming.py).

The token travels: **entrypoint, then run config, then `remote_node`, then A2A request metadata, then the agent**.

1. `build_config(customer_id, thread_id, user_token)` stores the token in the run config under `__user_token`.
   The leading `__` matters: LangGraph copies other configurable keys into the checkpoint metadata, which would
   write the token to Neon. Keys starting with `__` are skipped.

   ```python
   configurable = {"customer_id": customer_id, "thread_id": thread_id}
   if user_token:
       configurable[USER_TOKEN_CONFIG_KEY] = user_token      # "__user_token"
   ```

2. `remote_node` reads it and passes it to the client, which adds it to the request metadata:

   ```python
   def _request_metadata(customer_id, thread_id, user_token):
       metadata = {"customer_id": customer_id, "thread_id": thread_id}
       if user_token:
           metadata[USER_TOKEN_METADATA_KEY] = user_token     # "user_token"
       return metadata
   ```

3. On the agent, `RunnerExecutor.execute` verifies it when auth is enabled. The token's customer must equal the
   metadata `customer_id`, and the verified value is what the runner receives. Any failure marks the task `failed`
   with a generic message and logs `status=auth_rejected`:

   ```python
   if auth_enabled():
       verified = await self._verify_customer(context, customer_id, task_id)
       if verified is None:
           await updater.failed(new_text_message(REJECTED_MESSAGE))
           return
       customer_id = verified
   ```

   `verify_token` may fetch keys over the network on first use, so it runs in `asyncio.to_thread`.

4. The optional A2A wire log (a debug aid that records bodies) masks `user_token` through `_redact`.

With auth disabled none of this runs, which keeps the local default unchanged.

## 6. Customer scoping (0.5)

Files: [core/mcp_tools.py](../../src/hm_agent/core/mcp_tools.py), [shipment_tracking/tools.py](../../src/hm_agent/shipment_tracking/tools.py),
[shipment_tracking/agent.py](../../src/hm_agent/shipment_tracking/agent.py), tests in
[tests/test_customer_scoping.py](../../tests/test_customer_scoping.py) (uses the `db` fixture, i.e. the test database).

Authentication says who the caller is; scoping makes sure they only reach their own rows. The audit of every data query path:

| Agent / tool | How it is scoped | Test |
|---|---|---|
| order_management: `get_last_bought_product`, `get_purchase_count`, `list_orders`, `get_order_details` | SQL `WHERE o.customer_id = %s` | `test_order_tools_return_only_own_data` |
| fashion: `get_purchase_history`, `get_style_profile`, `get_styling_guide` | SQL filter on `customer_id` | `test_fashion_tools_return_only_own_data` |
| fashion: `search_products` | catalog search, reads no customer data | (nothing to scope) |
| shipment_tracking carrier lookup | **was unscoped**, now gated by `order_belongs_to_customer` | `test_shipment_*` |
| the `customer_id` tool argument | overwritten from the run config by `inject_customer_id` | `test_interceptor_*` |

Two real gaps were fixed:

- **Shipment lookup trusted the message.** The order id comes from what the customer types, so anyone could track
  anyone's order. Now the runner checks ownership first and gives one generic reply for "unknown" and "not yours":

  ```python
  if not order_belongs_to_customer(order_id, customer_id):
      logger.warning("agent=shipment_tracking step=order_not_owned thread_id=%s", thread_id)
      yield answer(ORDER_NOT_FOUND_REPLY)
      return
  ```

- **Silent default customer.** `inject_customer_id` used to fall back to `cust001` when the run config had no id. With
  auth on that could turn a missing id into someone's data, so it now fails closed:

  ```python
  customer_id = config.get("configurable", {}).get(CUSTOMER_ID_ARG)
  if not customer_id:
      if auth_enabled():
          raise PermissionError("no verified customer id in the run config")
      customer_id = DEFAULT_CUSTOMER_ID      # local CLI default only
  ```

Consequence for deployment: the shipment agent now reads `order_management.orders`, so it needs the Neon connection string.
It is a read across schemas with no foreign key, so the "no cross-schema FKs" rule still holds.

## 7. The AgentCore entrypoint (0.6)

Files: [runtime/entrypoint.py](../../src/hm_agent/runtime/entrypoint.py), tests in
[tests/test_runtime_entrypoint.py](../../tests/test_runtime_entrypoint.py).

A `BedrockAgentCoreApp` that serves `POST /invocations` and `GET /ping`. It replaces the public FastAPI layer as the
front door (the FastAPI code still exists until step 0.10).

**Request:** `{"message": "..."}` for a turn, or `{"action": "history"}`. The Runtime session id header
(`X-Amzn-Bedrock-AgentCore-Runtime-Session-Id`) is the conversation thread id.

**Response:** a stream of events, the same contract as the API plus `thread` and `done`:

```
{"type": "thread", "thread_id": ...}
{"type": "agent_selected" | "started" | "progress" | "token" | "reply", ...}
{"type": "history", "thread_id": ..., "messages": [...]}     (history action only)
{"type": "error", "detail": ...}
{"type": "done"}                                              (always last)
```

**Identity.** With auth on, the customer is the verified token's claim and the request body is ignored; with auth off
(local) it is the body's `customer_id` or the default:

```python
def resolve_identity(payload, context):
    if not auth_enabled():
        return str(payload.get("customer_id") or DEFAULT_CUSTOMER_ID), None
    token = bearer_token(context)
    return verify_token(token).customer_id, token       # token is forwarded to the agents (section 5)
```

The `Authorization` header only reaches the code if the Runtime's request-header allowlist includes it (deployment step 1.4).

**Thread ownership.** A stored thread records the customer it was created with; another customer gets the same
generic "Conversation not found" as a missing thread, for turns and history alike:

```python
owner = await asyncio.to_thread(thread_owner, graph, thread_id)
if owner is not None and owner != customer_id:
    yield {"type": "error", "detail": THREAD_NOT_FOUND_DETAIL}
    return
```

**Blocking graph, async server.** The LangGraph run is synchronous, so it runs on a worker thread and hands events to the
event loop through a queue (`_run_blocking`). Failures become a generic `error` event (no internals), and the stream
always ends with `done`:

```python
@app.entrypoint
async def invoke(payload, context):
    async for event in _handle(payload, context):
        yield event
    yield {"type": "done"}
```

**Startup.** `lifespan` builds the graph once with a small Neon checkpointer pool (`runtime.pool_max_size`). No push
receiver is started; the entrypoint is stream-only. History sits behind `load_history()` so it can move to a Lambda later.

Locally it runs on `runtime.port` (8080); set `RUNTIME_PORT` if that port is taken (it is blocked on some Windows machines).

## 8. Agent bind address and stream-only shipment (0.7, 0.8)

**0.7: bind address.** Files: [core/settings.py](../../src/hm_agent/core/settings.py) and the four `*/a2a_service.py` mains.

Runtime requires agents to listen on `0.0.0.0:9000`. Defaults stay local (`127.0.0.1` and each agent's own port), so
`scripts/a2a/run_agents.py` is unchanged; env vars override:

```python
def bind_address(agent: str) -> tuple[str, int]:
    host = os.environ.get("A2A_BIND_HOST", "").strip() or A2A_HOST
    port = os.environ.get("A2A_BIND_PORT", "").strip()
    return host, int(port) if port else A2A_PORTS[agent]
```

The address an agent's card **advertises** is a separate setting: `<AGENT>_AGENT_URL` (via `agent_url()`). A deployed
agent must set it to its public address, because clients post to the card's URL, not to where the agent listens.

**0.8: shipment tracking over the stream only.** Files: [a2a/push_receiver.py](../../src/hm_agent/a2a/push_receiver.py),
[cli.py](../../src/hm_agent/cli.py), [api/app.py](../../src/hm_agent/api/app.py), config key `a2a.push_receiver_enabled`
(env `PUSH_RECEIVER_ENABLED`).

The slow carrier lookup (about 20 s) streams progress lines over the open connection. The webhook receiver on `:8010` was
a fallback for dropped connections, which Runtime cannot host. With the switch off, `get_receiver()` returns `None`,
the CLI and API do not start the listener, and a lost stream returns the existing "try again" reply. The receiver code
is kept because the webhook-versus-stream decision is still open.

## 9. Docker images (step 0.9)

Files: [Dockerfile](../../Dockerfile), [docker/start-agent.sh](../../docker/start-agent.sh),
[docker-compose.yml](../../docker-compose.yml), [.dockerignore](../../.dockerignore).

> **Status: written and the compose file validates (`docker compose config`), but no image has been built or run**,
> because the Docker daemon was not running when this was written. Treat it as unverified until
> `docker compose up --build` succeeds and a streamed turn works through port 8090.

One multi-stage `Dockerfile`, all `linux/arm64` (what Runtime runs):

| Stage | Contents | Command |
|---|---|---|
| `base` | Python 3.11 slim, `uv sync --frozen --no-dev`, `config/` and `src/`, non-root user | |
| `agent` | `base` + `AGENT` build arg, default bind `0.0.0.0:9000` | `start-agent` |
| `supervisor` | `base`, port 8080 | `python -m hm_agent.runtime.entrypoint` |

Decisions worth knowing:

- **Code runs from `/app/src` via `PYTHONPATH`**, not installed into site-packages, because `REPO_ROOT` (where `config/` is
  found) is computed from the source tree.
- **Embedding model baked in** for customer_support and fashion (`PRELOAD_EMBEDDING=true`), so a cold start does not download it:

  ```dockerfile
  RUN if [ "$PRELOAD_EMBEDDING" = "true" ]; then \
        python -c "from hm_agent.core.config import load_config; from fastembed import TextEmbedding; TextEmbedding(load_config()['embedding'])"; \
      fi
  ```

- **Fashion runs two processes in one container.** Its agent loads tools from a long-lived MCP server while building, and
  that server's URL is not configurable yet (step 2.1). `start-agent.sh` starts the MCP server first and waits for its port:

  ```sh
  if [ "$AGENT" = "fashion" ]; then
    python -m hm_agent.fashion.mcp_server &
    until python -c "import socket; ...; socket.create_connection((h, p), 2)" 2>/dev/null; do sleep 1; done
  fi
  exec python -m "hm_agent.${AGENT}.a2a_service"
  ```

- **No secrets in images.** `.env` is excluded by `.dockerignore`; compose passes it at run time with `env_file`. `ref/`,
  `.tmp/`, `data/` and `.git` are also excluded.
- **Compose wiring.** Each agent sets its own `*_AGENT_URL` (so its card advertises `http://<service>:9000`), and the supervisor
  gets the same URLs plus `PUSH_RECEIVER_ENABLED=false`. The supervisor is published on host port 8090.
- Building arm64 on an x86 machine uses QEMU and is slow. A missing aarch64 wheel would fail at `uv sync`.

## 10. Switch reference

| Setting | Env override | Default | Effect |
|---|---|---|---|
| `auth.enabled` | `AUTH_ENABLED` | false | verify user tokens in the entrypoint and agents |
| (issuer) | `AUTH0_TENANT` | unset | issuer URL used to verify tokens |
| `aws.sigv4_enabled` | `SIGV4_ENABLED` | false | sign supervisor-to-agent calls |
| `a2a.push_receiver_enabled` | `PUSH_RECEIVER_ENABLED` | true | run the `:8010` webhook receiver |
| (agent listen address) | `A2A_BIND_HOST`, `A2A_BIND_PORT` | `127.0.0.1`, per-agent port | where an agent service listens |
| (agent public address) | `<AGENT>_AGENT_URL` | `http://127.0.0.1:<port>` | URL the card advertises and the supervisor calls |
| `runtime.port` | `RUNTIME_PORT` | 8080 | entrypoint port |

## 11. How to run and test

```bash
uv run pytest -q                                   # 84 tests, LLM mocked; scoping tests use the test database
uv run python scripts/a2a/run_agents.py            # the four agents (and the fashion MCP server)
RUNTIME_PORT=8090 uv run python -m hm_agent.runtime.entrypoint
curl -N -X POST localhost:8090/invocations -H 'Content-Type: application/json' \
  -H 'X-Amzn-Bedrock-AgentCore-Runtime-Session-Id: local-session-000000000000000001' \
  -d '{"message": "Where is my order ORD-1001?", "customer_id": "cust001"}'
docker compose up --build                          # unverified, see section 9
```

The scoping tests run against `neondb_test` (`NEON_POSTGRES_TEST_CONNECTION_STRING`), never the main database. Note that
a manual turn through the entrypoint writes a thread to whichever database `NEON_POSTGRES_CONNECTION_STRING` points to.
