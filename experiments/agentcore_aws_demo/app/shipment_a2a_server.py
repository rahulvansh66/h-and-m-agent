"""Shipment tracking agent exposed over A2A (simulated carrier lookup)."""

import asyncio
import hashlib
import logging
import os
import re

import uvicorn
from a2a.helpers.proto_helpers import new_task_from_user_message, new_text_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue_v2 import EventQueue
from a2a.server.request_handlers import DefaultRequestHandlerV2
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types.a2a_pb2 import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, TaskState
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH, PROTOCOL_VERSION_CURRENT, TransportProtocol
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

logging.basicConfig(level=logging.INFO, format="[shipment-a2a] %(message)s")
logger = logging.getLogger(__name__)

# Defaults are the local setup; on AgentCore Runtime Terraform sets HOST=0.0.0.0, PORT=9000.
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "9000"))
ORDER_ID_PATTERN = re.compile(r"\bORD-\d+\b", re.IGNORECASE)
CARRIERS = [
    ("DHL Express", "Hamburg sorting hub", "In transit, expected delivery in 2 days"),
    ("PostNord", "Stockholm distribution centre", "Out for delivery, expected today"),
    ("UPS", "Rotterdam depot", "Held at the depot, delivery attempt tomorrow"),
]


def build_card() -> AgentCard:
    """Build the Agent Card: the public self-description clients read before talking to us.

    Returns:
        AgentCard: Card with one skill; `streaming=True` tells clients they can watch progress live.
    """
    return AgentCard(
        name="shipment_tracking",
        description="Checks the carrier for the shipping status of an order. The lookup is slow, so it streams progress.",
        version="1.0.0",
        supported_interfaces=[
            AgentInterface(
                url=os.environ.get("AGENT_URL", f"http://127.0.0.1:{PORT}"),  # the client overrides this on AWS
                protocol_binding=TransportProtocol.JSONRPC,
                protocol_version=PROTOCOL_VERSION_CURRENT,
            )
        ],
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="shipment-status", name="Shipment status lookup",
                description="Carrier, last scan and delivery outlook for an order id.",
                tags=["shipping"], examples=["Where is my order ORD-1007?"],
            )
        ],
    )


class ShipmentExecutor(AgentExecutor):
    """The agent's work: turn one incoming message into a streamed task."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Run one task: announce it, report progress, publish the answer, complete.

        Args:
            context: Incoming message plus task/context ids and request metadata.
            event_queue: Where task events are published (the SDK streams them to the caller).
        """
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if context.current_task is None:
            await event_queue.enqueue_event(new_task_from_user_message(context.message))  # state: submitted
        await updater.start_work()                                                         # state: working

        match = ORDER_ID_PATTERN.search(context.get_user_input())
        if not match:
            await updater.add_artifact([new_text_part("Which order should I track? Send an id like ORD-1007.")],
                                       artifact_id="reply", name="reply", last_chunk=True)
            await updater.complete()
            return

        order_id = match.group(0).upper()
        logger.info("task=%s order=%s status=start", context.task_id, order_id)
        for step in ["Contacting the carrier", "Waiting for the carrier system", "Reading the latest scan events"]:
            await updater.update_status(TaskState.TASK_STATE_WORKING, new_text_message(step))  # progress line
            await asyncio.sleep(1)                                                             # the "slow" part

        digest = int(hashlib.sha256(order_id.encode()).hexdigest()[:8], 16)
        carrier, location, outlook = CARRIERS[digest % len(CARRIERS)]
        reply = f"Order {order_id} is with {carrier}. Last scan: {location}. {outlook}."
        await updater.add_artifact([new_text_part(reply)], artifact_id="reply", name="reply", last_chunk=True)
        await updater.complete()                                                               # state: completed
        logger.info("task=%s status=complete", context.task_id)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Mark the task cancelled (our runs are short, so there is nothing to stop).

        Args:
            context: Request context holding the task to cancel.
            event_queue: Queue the cancellation event is published to.
        """
        await TaskUpdater(event_queue, context.task_id, context.context_id).cancel()


async def ping(_request) -> JSONResponse:
    """Health check required by AgentCore Runtime (GET /ping).

    Args:
        _request: The incoming request (unused).

    Returns:
        JSONResponse: Status "Healthy".
    """
    return JSONResponse({"status": "Healthy"})


class DefaultA2AVersion:
    """ASGI middleware: treat requests without an `A2A-Version` header as protocol 1.0.

    AgentCore Runtime's A2A proxy does not forward that header, and the SDK reads a missing header as the old
    0.3 protocol and rejects the call. Locally the client's header arrives untouched, so this changes nothing there.
    """

    def __init__(self, inner) -> None:
        """Wrap the real app.

        Args:
            inner: The ASGI application to protect.
        """
        self._inner = inner

    async def __call__(self, scope, receive, send) -> None:
        """Add the version header when it is missing, then hand the request on.

        Args:
            scope: ASGI connection scope.
            receive: ASGI receive channel.
            send: ASGI send channel.
        """
        if scope["type"] == "http" and not any(name == b"a2a-version" for name, _ in scope["headers"]):
            scope["headers"] = [*scope["headers"], (b"a2a-version", b"1.0")]
        await self._inner(scope, receive, send)


card = build_card()
handler = DefaultRequestHandlerV2(agent_executor=ShipmentExecutor(), task_store=InMemoryTaskStore(), agent_card=card)
starlette_app = Starlette(routes=[
    Route("/ping", ping),                                                  # GET /ping  (Runtime health check)
    *create_agent_card_routes(card, card_url=AGENT_CARD_WELL_KNOWN_PATH),  # GET /.well-known/agent-card.json
    *create_jsonrpc_routes(handler, rpc_url="/"),                          # POST /  (JSON-RPC messages)
])
app = DefaultA2AVersion(starlette_app)

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
