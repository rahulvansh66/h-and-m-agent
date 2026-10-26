"""FastAPI application factory.

Role in the system: assembles the API. The lifespan opens the pooled Neon checkpointer, builds the
supervisor graph once, and starts the push-notification receiver so results pushed by agents are
saved into their thread and fanned out to listeners. Tests pass ready-made ``Services`` to skip all
of that.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langfuse import get_client

from hm_agent.a2a.push_receiver import PendingTask, PushReceiver, set_receiver
from hm_agent.api.routes import chat, health
from hm_agent.api.service import PushEventHub, RateLimiter, Services
from hm_agent.core.db import neon_checkpointer_pool
from hm_agent.core.settings import (
    A2A_WEBHOOK_PORT,
    API_CORS_ORIGINS,
    API_POOL_MAX_SIZE,
    API_RATE_LIMIT_PER_MIN,
    push_receiver_enabled,
)
from hm_agent.graph import build_graph

logger = logging.getLogger(__name__)


def _make_deliver(services: Services) -> Callable[[PendingTask, str], None]:
    """Create the callback that saves a pushed result into its thread and tells listeners.

    Args:
        services (Services): Shared services (graph, locks and hub).

    Returns:
        Callable[[PendingTask, str], None]: Function ``(task, text) -> None`` run on a worker thread.
    """

    def deliver(task: PendingTask, text: str) -> None:
        """Add the result to the thread, then publish it to the thread's listeners.

        Args:
            task (PendingTask): The finished background task.
            text (str): Result text to show the customer.
        """
        # customer_id is recorded in the checkpoint metadata, which is how thread ownership is checked.
        config: RunnableConfig = {"configurable": {"thread_id": task.thread_id, "customer_id": task.customer_id}}
        with services.locks.get(task.thread_id):
            # as_node makes the message look like it came from the agent's node, so no other node runs.
            services.graph.update_state(config, {"messages": [AIMessage(text, name=task.agent)]}, as_node=task.agent)
        services.hub.publish(task.thread_id, {"type": "update", "agent": task.agent, "text": text})
        logger.info("component=api status=push_delivered thread_id=%s agent=%s", task.thread_id, task.agent)

    return deliver


def create_app(services: Services | None = None) -> FastAPI:
    """Build the FastAPI app.

    Args:
        services (Services | None): Ready-made services (used by tests). When None, the lifespan
            builds them from Neon and starts the push receiver.

    Returns:
        FastAPI: App to pass to ``uvicorn.run``.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Create the shared services at startup and release them at shutdown.

        Args:
            app (FastAPI): The app being started.

        Yields:
            None: Control while the app serves requests.
        """
        if services is not None:
            app.state.services = services
            yield
            return
        with neon_checkpointer_pool(API_POOL_MAX_SIZE) as checkpointer:
            built = Services(
                graph=build_graph(checkpointer),
                limiter=RateLimiter(API_RATE_LIMIT_PER_MIN),
                hub=PushEventHub(asyncio.get_running_loop()),
            )
            app.state.services = built
            # Receives push notifications from agents whose stream broke and delivers their results.
            receiver: PushReceiver | None = None
            if push_receiver_enabled():
                receiver = PushReceiver(A2A_WEBHOOK_PORT, _make_deliver(built))
                receiver.start()
                set_receiver(receiver)
            logger.info("component=api status=started")
            try:
                yield
            finally:
                if receiver is not None:
                    set_receiver(None)
                    receiver.stop()
                get_client().flush()

    app = FastAPI(title="H&M Assistant API", version="0.1.0", lifespan=lifespan)
    if API_CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=API_CORS_ORIGINS,
            allow_methods=["GET", "POST"],
            allow_headers=["X-API-Key", "X-Customer-Id", "Content-Type"],
        )

    @app.middleware("http")
    async def log_requests(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        """Tag each request with an id, return it as ``X-Request-Id`` and log one line per request.

        Message bodies and headers are never logged.

        Args:
            request (Request): Incoming request.
            call_next (Callable[[Request], Awaitable[Response]]): Next handler in the chain.

        Returns:
            Response: The route's response with the ``X-Request-Id`` header added.
        """
        request_id = uuid.uuid4().hex[:12]
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        logger.info(
            "component=api request_id=%s method=%s path=%s status=%d duration_s=%.2f",
            request_id, request.method, request.url.path, response.status_code, time.perf_counter() - start,
        )
        return response

    app.include_router(health.router)
    app.include_router(chat.router)
    return app
