"""Chat routes: one-shot reply, streamed reply, thread history and pushed-result events.

Role in the system: the public conversation API. Each route runs the customer's message through
the supervisor graph (``hm_agent.turns``) for the customer named in ``X-Customer-Id``.
"""

import logging
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from sse_starlette.sse import EventSourceResponse

from hm_agent.api.deps import (
    enforce_rate_limit,
    get_customer_id,
    get_services,
    require_api_key,
    require_thread_access,
)
from hm_agent.api.schemas import ChatRequest, ChatResponse, HistoryResponse
from hm_agent.api.service import Services, aiter_turn_events, history_messages, sse_frame

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])


async def _prepare_thread(body: ChatRequest, customer_id: str, services: Services) -> str:
    """Pick the thread of a chat request and check the customer may use it.

    Args:
        body (ChatRequest): The request; its ``thread_id`` is used when given.
        customer_id (str): Customer making the request.
        services (Services): Shared services.

    Returns:
        str: The thread id (a new uuid4 when the request had none).

    Raises:
        HTTPException: 404 when the thread belongs to another customer.
    """
    if body.thread_id is None:
        return str(uuid.uuid4())
    await require_thread_access(body.thread_id, customer_id, services, must_exist=False)
    return body.thread_id


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(enforce_rate_limit)])
async def chat(
    body: ChatRequest,
    customer_id: str = Depends(get_customer_id),
    services: Services = Depends(get_services),
) -> ChatResponse:
    """Send one message and return the complete reply.

    Args:
        body (ChatRequest): Message and optional thread id.
        customer_id (str): Customer the request acts for (``X-Customer-Id``).
        services (Services): Shared services.

    Returns:
        ChatResponse: Thread id, answering agent and reply text.

    Raises:
        HTTPException: 404 for another customer's thread, 502 when the turn produced no reply.
    """
    thread_id = await _prepare_thread(body, customer_id, services)
    reply: dict | None = None
    async for event in aiter_turn_events(services, body.message, customer_id, thread_id):
        if event["type"] == "reply":
            reply = event
        elif event["type"] == "error":
            break
    if reply is None:
        raise HTTPException(status_code=502, detail="the assistant could not produce a reply")
    return ChatResponse(thread_id=thread_id, agent=reply["agent"], reply=reply["text"])


@router.post("/chat/stream", dependencies=[Depends(enforce_rate_limit)])
async def chat_stream(
    body: ChatRequest,
    customer_id: str = Depends(get_customer_id),
    services: Services = Depends(get_services),
) -> EventSourceResponse:
    """Send one message and stream the turn as Server-Sent Events.

    Events, in order: ``thread`` (the thread id), ``agent_selected``, then any of ``started`` /
    ``progress`` / ``token`` from the agent, ``reply`` (full text), and finally ``done``. A failed
    turn sends ``error`` followed by ``done``.

    Args:
        body (ChatRequest): Message and optional thread id.
        customer_id (str): Customer the request acts for (``X-Customer-Id``).
        services (Services): Shared services.

    Returns:
        EventSourceResponse: The event stream.

    Raises:
        HTTPException: 404 for another customer's thread (before the stream starts).
    """
    thread_id = await _prepare_thread(body, customer_id, services)

    async def frames() -> AsyncIterator[dict]:
        """Yield the SSE frames of the turn.

        Yields:
            dict: Frames for ``EventSourceResponse``.
        """
        yield sse_frame({"type": "thread", "thread_id": thread_id})
        async for event in aiter_turn_events(services, body.message, customer_id, thread_id):
            yield sse_frame(event)
        yield sse_frame({"type": "done"})

    return EventSourceResponse(frames())


@router.get("/threads/{thread_id}/messages", response_model=HistoryResponse)
async def thread_messages(
    thread_id: str,
    customer_id: str = Depends(get_customer_id),
    services: Services = Depends(get_services),
) -> HistoryResponse:
    """Return the stored messages of a conversation.

    Includes results added later by push notifications.

    Args:
        thread_id (str): Conversation thread id.
        customer_id (str): Customer the request acts for (``X-Customer-Id``).
        services (Services): Shared services.

    Returns:
        HistoryResponse: Messages, oldest first.

    Raises:
        HTTPException: 404 when the thread does not exist or belongs to another customer.
    """
    await require_thread_access(thread_id, customer_id, services, must_exist=True)
    messages = await run_in_threadpool(history_messages, services.graph, thread_id)
    return HistoryResponse(thread_id=thread_id, messages=messages)


@router.get("/threads/{thread_id}/events")
async def thread_events(
    thread_id: str,
    customer_id: str = Depends(get_customer_id),
    services: Services = Depends(get_services),
) -> EventSourceResponse:
    """Stream results that arrive after a turn ended (push notification fallback).

    Each ``update`` event carries ``agent`` and ``text``; the same message is also saved in the
    thread history, so a client that was not listening can read it from there.

    Args:
        thread_id (str): Conversation thread id.
        customer_id (str): Customer the request acts for (``X-Customer-Id``).
        services (Services): Shared services.

    Returns:
        EventSourceResponse: The event stream (kept open; a ping is sent periodically).

    Raises:
        HTTPException: 404 when the thread does not exist or belongs to another customer.
    """
    await require_thread_access(thread_id, customer_id, services, must_exist=True)
    queue = services.hub.subscribe(thread_id)

    async def frames() -> AsyncIterator[dict]:
        """Yield pushed results until the client disconnects.

        Yields:
            dict: Frames for ``EventSourceResponse``.
        """
        try:
            while True:
                yield sse_frame(await queue.get())
        finally:
            services.hub.unsubscribe(thread_id, queue)

    return EventSourceResponse(frames())
