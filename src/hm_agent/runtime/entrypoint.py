"""AgentCore entrypoint of the supervisor: serves one customer turn as a stream of events.

Role in the system: the front door of the supervisor on AgentCore Runtime (``POST /invocations``,
``GET /ping``). It does not make routing decisions itself; it identifies the customer, checks the
thread belongs to them and streams ``hm_agent.turns.stream_turn`` back with the same event contract the
FastAPI layer uses:

    {"type": "thread", "thread_id": ...}                 first event: the conversation thread
    {"type": "agent_selected" | "started" | "progress" | "token" | "reply", ...}   the turn (see ``hm_agent.turns``)
    {"type": "history", "thread_id": ..., "messages": [...]}   answer to ``{"action": "history"}``
    {"type": "error", "detail": ...}                     the request or the turn failed
    {"type": "done"}                                     always the last event

Identity: with auth enabled the customer is the claim of the verified ``Authorization`` bearer token (the
request body is never trusted); with auth disabled (local) it is the body's ``customer_id`` or the default.
The Runtime session id (``X-Amzn-Bedrock-AgentCore-Runtime-Session-Id``) is the conversation thread id.

Request body: ``{"message": "..."}`` for a turn, or ``{"action": "history"}`` to read the session's messages.

Run locally (the four agents must be running, see ``scripts/a2a/run_agents.py``):
    uv run python -m hm_agent.runtime.entrypoint
"""

import asyncio
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from bedrock_agentcore.runtime import BedrockAgentCoreApp, RequestContext
from dotenv import load_dotenv
from langfuse import get_client
from langgraph.graph.state import CompiledStateGraph
from starlette.applications import Starlette

from hm_agent.auth import AuthError, verify_token
from hm_agent.core.config import REPO_ROOT
from hm_agent.core.db import neon_checkpointer_pool
from hm_agent.core.logging_setup import configure_logging
from hm_agent.core.settings import (
    DEFAULT_CUSTOMER_ID,
    RUNTIME_MAX_MESSAGE_CHARS,
    RUNTIME_POOL_MAX_SIZE,
    RUNTIME_PORT,
    auth_enabled,
)
from hm_agent.graph import build_graph
from hm_agent.runtime.threads import history_messages, thread_owner
from hm_agent.turns import stream_turn

logger = logging.getLogger(__name__)

ERROR_DETAIL = "The assistant could not process this request."
UNAUTHORIZED_DETAIL = "Unauthorized."
BAD_REQUEST_DETAIL = "Send a non-empty 'message' of reasonable length, or {'action': 'history'}."
THREAD_NOT_FOUND_DETAIL = "Conversation not found."  # same text for "missing" and "someone else's"
BEARER_PREFIX = "bearer "


@dataclass
class _State:
    """Process-wide state created at startup.

    Attributes:
        graph: Compiled supervisor graph with the Neon checkpointer, or None before startup.
    """

    graph: CompiledStateGraph | None = None


_state = _State()


@asynccontextmanager
async def lifespan(_app: Starlette):
    """Build the supervisor graph once at startup and close the checkpointer pool at shutdown.

    Args:
        _app (Starlette): The ASGI app (unused).

    Yields:
        None: Control while the server runs.
    """
    with neon_checkpointer_pool(RUNTIME_POOL_MAX_SIZE) as checkpointer:
        _state.graph = build_graph(checkpointer)
        logger.info("component=runtime status=started pool_max_size=%d auth_enabled=%s", RUNTIME_POOL_MAX_SIZE, auth_enabled())
        try:
            yield
        finally:
            _state.graph = None
            get_client().flush()


app = BedrockAgentCoreApp(lifespan=lifespan)


def bearer_token(context: RequestContext) -> str:
    """Extract the bearer token from the request headers the Runtime forwarded.

    Needs ``Authorization`` in the Runtime's request-header allowlist.

    Args:
        context (RequestContext): Request context of the invocation.

    Returns:
        str: The token without the ``Bearer`` prefix, or an empty string if absent.
    """
    value = (context.request_headers or {}).get("Authorization", "").strip()
    return value[len(BEARER_PREFIX):].strip() if value.lower().startswith(BEARER_PREFIX) else ""


def resolve_identity(payload: dict, context: RequestContext) -> tuple[str, str | None]:
    """Decide which customer the request acts for.

    Args:
        payload (dict): Invocation body.
        context (RequestContext): Request context of the invocation.

    Returns:
        tuple[str, str | None]: ``(customer_id, user_token)``. With auth enabled the id comes from the
        verified token and the token is returned so it can be forwarded to the agents; with auth
        disabled the id is the body's ``customer_id`` (or the default) and the token is None.

    Raises:
        AuthError: If auth is enabled and the token is missing or invalid.
    """
    if not auth_enabled():
        return str(payload.get("customer_id") or DEFAULT_CUSTOMER_ID), None
    token = bearer_token(context)
    return verify_token(token).customer_id, token


def _valid_message(payload: dict) -> str | None:
    """Return the customer's message if the body carries a usable one.

    Args:
        payload (dict): Invocation body.

    Returns:
        str | None: The message, or None if it is missing, not text, blank or too long.
    """
    message = payload.get("message")
    if not isinstance(message, str) or not message.strip() or len(message) > RUNTIME_MAX_MESSAGE_CHARS:
        return None
    return message


async def _run_blocking(events: Callable[[], Iterator[dict]], thread_id: str) -> AsyncIterator[dict]:
    """Run a blocking event generator on a worker thread and yield its events to async code as they happen.

    Args:
        events (Callable[[], Iterator[dict]]): Factory of the blocking generator (the graph is synchronous).
        thread_id (str): Conversation thread id, for log correlation.

    Yields:
        dict: The generator's events; a failure ends with ``{"type": "error", ...}``.
    """
    loop = asyncio.get_running_loop()
    outbox: asyncio.Queue[dict | None] = asyncio.Queue()

    def produce() -> None:
        """Worker thread: iterate the generator and hand each event to the event loop."""
        try:
            for event in events():
                loop.call_soon_threadsafe(outbox.put_nowait, event)
        except Exception:
            logger.exception("component=runtime status=turn_failed thread_id=%s", thread_id)
            loop.call_soon_threadsafe(outbox.put_nowait, {"type": "error", "detail": ERROR_DETAIL})
        finally:
            get_client().flush()  # each turn is its own short trace; send it before the session idles
            loop.call_soon_threadsafe(outbox.put_nowait, None)

    worker = asyncio.create_task(asyncio.to_thread(produce))
    while (event := await outbox.get()) is not None:
        yield event
    await worker


def load_history(graph: CompiledStateGraph, thread_id: str) -> dict:
    """Build the ``history`` event for a thread.

    Kept behind a function so it can move out of the entrypoint (for example to a Lambda) later.

    Args:
        graph (CompiledStateGraph): Compiled supervisor graph with a checkpointer.
        thread_id (str): Conversation thread id.

    Returns:
        dict: ``{"type": "history", "thread_id": ..., "messages": [{"role", "agent", "text"}, ...]}``.
    """
    return {"type": "history", "thread_id": thread_id, "messages": history_messages(graph, thread_id)}


async def _handle(payload: dict, context: RequestContext) -> AsyncIterator[dict]:
    """Serve one invocation, yielding events (see the module docstring for the contract).

    Args:
        payload (dict): Invocation body.
        context (RequestContext): Request context; ``session_id`` is the conversation thread id.

    Yields:
        dict: Events. Failures become an ``error`` event, never an exception.
    """
    start = time.perf_counter()
    thread_id = context.session_id or str(uuid.uuid4())
    status = "complete"
    try:
        try:
            customer_id, user_token = resolve_identity(payload, context)
        except AuthError:  # the reason is already logged by the verifier
            status = "unauthorized"
            yield {"type": "error", "detail": UNAUTHORIZED_DETAIL}
            return
        graph = _state.graph
        if graph is None:
            raise RuntimeError("the supervisor graph is not built (startup did not run)")
        yield {"type": "thread", "thread_id": thread_id}

        # A thread may only be continued (or read) by the customer who started it.
        owner = await asyncio.to_thread(thread_owner, graph, thread_id)
        if owner is not None and owner != customer_id:
            status = "thread_denied"
            logger.warning("component=runtime status=thread_denied thread_id=%s", thread_id)
            yield {"type": "error", "detail": THREAD_NOT_FOUND_DETAIL}
            return

        if payload.get("action") == "history":
            yield await asyncio.to_thread(load_history, graph, thread_id)
            return
        message = _valid_message(payload)
        if message is None:
            status = "bad_request"
            yield {"type": "error", "detail": BAD_REQUEST_DETAIL}
            return
        async for event in _run_blocking(lambda: stream_turn(graph, message, customer_id, thread_id, user_token), thread_id):
            if event["type"] == "error":
                status = "failed"
            yield event
    except Exception:
        status = "failed"
        logger.exception("component=runtime status=failed thread_id=%s", thread_id)
        yield {"type": "error", "detail": ERROR_DETAIL}
    finally:
        logger.info("component=runtime status=%s thread_id=%s duration_s=%.2f", status, thread_id, time.perf_counter() - start)


@app.entrypoint
async def invoke(payload: dict, context: RequestContext) -> AsyncIterator[dict]:
    """AgentCore entrypoint: stream the events of one invocation, always ending with ``done``.

    Args:
        payload (dict): Invocation body: ``{"message": ...}`` or ``{"action": "history"}`` (plus
            ``customer_id`` only when auth is disabled).
        context (RequestContext): Request context; ``session_id`` is the conversation thread id and
            ``request_headers`` carries ``Authorization`` when the Runtime allowlists it.

    Yields:
        dict: Events as described in the module docstring.
    """
    async for event in _handle(payload, context):
        yield event
    yield {"type": "done"}


def main() -> None:
    """Run the entrypoint on the Runtime port (the container uses the same command).

    Locally the port can be changed with the ``RUNTIME_PORT`` environment variable when 8080 is taken.
    """
    load_dotenv(REPO_ROOT / ".env")
    configure_logging("runtime")
    app.run(port=int(os.environ.get("RUNTIME_PORT") or RUNTIME_PORT))


if __name__ == "__main__":
    main()
