"""Shared server-side state of the API: per-thread locks, rate limiting and pushed-result fan-out.

Role in the system: everything the routes need besides the compiled graph. ``Services`` is created
once in the app lifespan and read by the routes through ``hm_agent.api.deps``.
"""

import asyncio
import json
import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph

from hm_agent.turns import stream_turn

logger = logging.getLogger(__name__)

RATE_WINDOW_S = 60.0  # length of the sliding window used by the rate limiter


class ThreadLocks:
    """One lock per conversation thread.

    Two writes to the same thread (a turn and a pushed result) would overwrite each other's
    checkpoint, so each is done while holding the thread's lock.
    """

    def __init__(self) -> None:
        """Create an empty lock registry."""
        self._guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)

    def get(self, thread_id: str) -> threading.Lock:
        """Return the lock of a thread, creating it on first use.

        Args:
            thread_id (str): Conversation thread id.

        Returns:
            threading.Lock: The thread's lock (use it as a context manager).
        """
        with self._guard:
            return self._locks[thread_id]


class RateLimiter:
    """Sliding-window limiter: at most ``limit`` calls per customer per minute (in memory, per process)."""

    def __init__(self, limit: int) -> None:
        """Create a limiter.

        Args:
            limit (int): Calls allowed per customer within ``RATE_WINDOW_S`` seconds.
        """
        self._limit = limit
        self._calls: dict[str, deque[float]] = defaultdict(deque)
        self._guard = threading.Lock()

    def check(self, customer_id: str) -> float | None:
        """Record a call and say whether it is allowed.

        Args:
            customer_id (str): Customer making the call.

        Returns:
            float | None: None when the call is allowed, otherwise the seconds until it would be.
        """
        now = time.monotonic()
        with self._guard:
            calls = self._calls[customer_id]
            while calls and now - calls[0] >= RATE_WINDOW_S:
                calls.popleft()
            if len(calls) >= self._limit:
                return RATE_WINDOW_S - (now - calls[0])
            calls.append(now)
            return None


class PushEventHub:
    """Fan-out of pushed results to clients listening on a thread's ``/events`` stream.

    Results are published from worker threads (the push receiver), so delivery to each
    listener's queue goes through the event loop.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        """Create a hub bound to the server's event loop.

        Args:
            loop (asyncio.AbstractEventLoop): Loop the listeners' queues belong to.
        """
        self._loop = loop
        self._listeners: dict[str, set[asyncio.Queue]] = defaultdict(set)

    def subscribe(self, thread_id: str) -> asyncio.Queue:
        """Start listening to a thread.

        Args:
            thread_id (str): Conversation thread id.

        Returns:
            asyncio.Queue: Queue that receives each published event dict.
        """
        queue: asyncio.Queue = asyncio.Queue()
        self._listeners[thread_id].add(queue)
        return queue

    def unsubscribe(self, thread_id: str, queue: asyncio.Queue) -> None:
        """Stop listening to a thread.

        Args:
            thread_id (str): Conversation thread id.
            queue (asyncio.Queue): The queue returned by ``subscribe``.
        """
        self._listeners[thread_id].discard(queue)

    def publish(self, thread_id: str, event: dict) -> None:
        """Send an event to every listener of a thread (safe to call from any thread).

        Args:
            thread_id (str): Conversation thread id.
            event (dict): Event to deliver. Dropped (it is still in the thread history) if nobody listens.
        """
        for queue in list(self._listeners.get(thread_id, ())):
            self._loop.call_soon_threadsafe(queue.put_nowait, event)


@dataclass
class Services:
    """Everything the routes share.

    Attributes:
        graph: Compiled supervisor graph (with checkpointer).
        limiter: Per-customer rate limiter.
        hub: Pushed-result fan-out.
        locks: Per-thread locks.
    """

    graph: CompiledStateGraph
    limiter: RateLimiter
    hub: PushEventHub
    locks: ThreadLocks = field(default_factory=ThreadLocks)


def thread_owner(graph: CompiledStateGraph, thread_id: str) -> str | None:
    """Return the customer a stored thread belongs to.

    Every checkpoint written through the API records the ``customer_id`` it was written with.

    Args:
        graph (CompiledStateGraph): Compiled supervisor graph with a checkpointer.
        thread_id (str): Conversation thread id.

    Returns:
        str | None: The owning customer id; an empty string for a thread with no recorded owner
        (so it matches nobody); None when the thread has no stored state.
    """
    snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
    if not snapshot.values:
        return None
    return (snapshot.metadata or {}).get("customer_id", "")


def run_turn_events(services: Services, query: str, customer_id: str, thread_id: str) -> Iterator[dict]:
    """Run one turn while holding the thread lock and yield its events (blocking generator).

    Args:
        services (Services): Shared server state.
        query (str): The customer's message.
        customer_id (str): Customer the conversation belongs to.
        thread_id (str): Conversation thread id.

    Yields:
        dict: Turn events from ``hm_agent.turns.stream_turn``.
    """
    with services.locks.get(thread_id):
        yield from stream_turn(services.graph, query, customer_id, thread_id)


async def aiter_turn_events(services: Services, query: str, customer_id: str, thread_id: str) -> AsyncIterator[dict]:
    """Run one turn on a worker thread and yield its events to async code as they happen.

    Args:
        services (Services): Shared server state.
        query (str): The customer's message.
        customer_id (str): Customer the conversation belongs to.
        thread_id (str): Conversation thread id.

    Yields:
        dict: Turn events; a failed turn ends with ``{"type": "error", "detail": ...}``.
    """
    loop = asyncio.get_running_loop()
    outbox: asyncio.Queue[dict | None] = asyncio.Queue()

    def produce() -> None:
        """Worker thread: run the turn and hand each event to the event loop."""
        try:
            for event in run_turn_events(services, query, customer_id, thread_id):
                loop.call_soon_threadsafe(outbox.put_nowait, event)
        except Exception:
            logger.exception("component=api status=turn_failed thread_id=%s", thread_id)
            error = {"type": "error", "detail": "The assistant could not process this request."}
            loop.call_soon_threadsafe(outbox.put_nowait, error)
        finally:
            loop.call_soon_threadsafe(outbox.put_nowait, None)

    worker = asyncio.create_task(asyncio.to_thread(produce))
    while (event := await outbox.get()) is not None:
        yield event
    await worker


def sse_frame(event: dict) -> dict:
    """Format an event dict as a Server-Sent-Events frame for ``EventSourceResponse``.

    Args:
        event (dict): Event with a ``type`` key.

    Returns:
        dict: ``{"event": type, "data": json}``.
    """
    return {"event": event["type"], "data": json.dumps(event, ensure_ascii=False)}


def history_messages(graph: CompiledStateGraph, thread_id: str) -> list[dict]:
    """Read a thread's messages from the checkpointer.

    Args:
        graph (CompiledStateGraph): Compiled supervisor graph with a checkpointer.
        thread_id (str): Conversation thread id.

    Returns:
        list[dict]: ``{"role", "agent", "text"}`` dicts, oldest first.
    """
    values = graph.get_state({"configurable": {"thread_id": thread_id}}).values
    return [
        {
            "role": "user" if isinstance(m, HumanMessage) else "assistant",
            "agent": None if isinstance(m, HumanMessage) else m.name,
            "text": m.text,
        }
        for m in values.get("messages", [])
    ]
