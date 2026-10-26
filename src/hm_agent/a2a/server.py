"""Generic A2A server: serves any streaming "text in, chunks out" runner as an A2A agent.

Role in the system: every agent service (order management, customer support, shipment tracking) is
this wrapper around its own runner. It turns an incoming ``SendMessage`` into a Task, runs the
runner off the event loop (LangGraph graphs and the MCP tool wrapper are blocking), and streams its
output through the task lifecycle: submitted -> working (progress messages) -> artifact chunks ->
completed (or failed). Streaming clients see each event as it happens; blocking clients get the
assembled task at the end.
"""

import asyncio
import logging
import time
from collections.abc import Callable, Iterator

import httpx
from a2a.helpers.proto_helpers import (
    get_data_parts,
    new_task_from_user_message,
    new_text_message,
    new_text_part,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue_v2 import EventQueue
from a2a.server.request_handlers import DefaultRequestHandlerV2
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import (
    BasePushNotificationSender,
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
    TaskUpdater,
)
from a2a.types.a2a_pb2 import AgentCard, TaskState
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
from langchain_core.messages import AIMessage, AIMessageChunk, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.graph.state import CompiledStateGraph
from starlette.applications import Starlette

from hm_agent.a2a.chunks import Chunk, answer, progress
from hm_agent.auth import USER_TOKEN_METADATA_KEY, AuthError, verify_token
from hm_agent.core.settings import TRACING_TAGS, auth_enabled

logger = logging.getLogger(__name__)

RPC_PATH = "/"  # JSON-RPC endpoint; the Agent Card tells clients where it is
REJECTED_MESSAGE = "The agent could not process this request."  # generic on purpose: no hint about why
TOKEN_FLUSH_CHARS = 24  # LLM tokens are grouped into pieces of at least this size, one A2A event per piece

# Runner signature: (latest user text, prior messages, customer_id, thread_id) -> stream of chunks.
Runner = Callable[[str, list[dict], str, str], Iterator[Chunk]]


def graph_runner(
    graph: CompiledStateGraph,
    agent: str,
    progress_labels: dict[str, str] | None = None,
    stream_tokens: bool = False,
) -> Runner:
    """Wrap a compiled LangGraph graph as a streaming ``Runner``.

    The prior messages sent by the supervisor are rebuilt into LangChain messages and the graph is
    streamed with Langfuse tracing (session = thread id, user = customer id). A finished node listed
    in ``progress_labels`` yields a progress chunk. With ``stream_tokens`` the text the model
    writes is yielded as it is generated; otherwise (or when the model wrote no plain text) the last
    message of the final state is yielded as one answer chunk.

    Args:
        graph (CompiledStateGraph): Agent graph that reads and writes ``messages``.
        agent (str): Agent name, used as the Langfuse run name and tag.
        progress_labels (dict[str, str] | None): Graph node name -> status line shown when that node finishes.
        stream_tokens (bool): Stream model tokens as answer chunks. Use only when the model's text is
            the final reply (not for nodes whose output is validated or rewritten afterwards).

    Returns:
        Runner: Generator function ``(text, history, customer_id, thread_id) -> Iterator[Chunk]``.
    """
    labels = progress_labels or {}

    def run(text: str, history: list[dict], customer_id: str, thread_id: str) -> Iterator[Chunk]:
        """Stream the graph for one A2A request.

        Args:
            text (str): Latest customer message.
            history (list[dict]): Prior messages as ``{"role", "text", "name"}`` dicts, oldest first.
            customer_id (str): Customer the conversation belongs to (read by the MCP tool interceptor).
            thread_id (str): Conversation thread id (Langfuse session).

        Yields:
            Chunk: Progress lines and answer pieces, in order.
        """
        messages: list[AnyMessage] = [
            HumanMessage(m["text"]) if m["role"] == "user" else AIMessage(m["text"], name=m.get("name"))
            for m in history
        ]
        messages.append(HumanMessage(text))
        config: RunnableConfig = {
            "configurable": {"customer_id": customer_id, "thread_id": thread_id},
            "callbacks": [CallbackHandler()],
            "run_name": f"a2a-{agent}",
            "metadata": {
                "langfuse_session_id": thread_id,
                "langfuse_user_id": customer_id,
                "langfuse_tags": [*TRACING_TAGS, f"a2a-{agent}"],
            },
        }
        buffer, streamed, final_text = "", False, ""
        try:
            for mode, data in graph.stream({"messages": messages}, config, stream_mode=["messages", "updates", "values"]):
                if mode == "messages":
                    token, _meta = data
                    # Only plain model text; tool-call fragments are not part of the reply.
                    if stream_tokens and isinstance(token, AIMessageChunk) and token.text and not token.tool_call_chunks:
                        buffer += token.text
                        if len(buffer) >= TOKEN_FLUSH_CHARS:
                            yield answer(buffer)
                            buffer, streamed = "", True
                elif mode == "updates":
                    for node in data:
                        if node in labels:
                            yield progress(labels[node])
                else:
                    final_text = data["messages"][-1].text
            if buffer:
                yield answer(buffer)
                streamed = True
            if not streamed:
                yield answer(final_text)
        finally:
            # Each request is its own short trace; send it before replying so it is not lost on shutdown.
            get_client().flush()

    return run


class RunnerExecutor(AgentExecutor):
    """A2A executor that streams the chunks of a blocking ``Runner`` as task events."""

    def __init__(self, agent: str, runner: Runner) -> None:
        """Store the agent name and runner.

        Args:
            agent (str): Agent name, used in log fields.
            runner (Runner): Generator function that produces the reply as chunks.
        """
        self._agent = agent
        self._runner = runner

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Run one task: publish submitted/working, stream the runner's chunks, then complete.

        Reads the customer id and thread id from the request metadata and the prior messages from
        the first data part of the message. The runner is blocking, so it runs on a worker thread
        and hands its chunks to the event loop through a queue. A ``progress`` chunk becomes a
        ``working`` status update with a message; ``answer`` chunks become appends to one artifact,
        the last of them flagged ``last_chunk``. A runner failure marks the task ``failed``.

        When auth is enabled the user token in the request metadata is verified first and its
        customer id must equal the metadata ``customer_id``; the verified id is what the runner gets.
        A missing, invalid or mismatching token marks the task ``failed`` without running the agent.

        Args:
            context (RequestContext): Incoming message, task id, context id and request metadata.
            event_queue (EventQueue): Queue the task events are published to.
        """
        task_id, context_id = context.task_id, context.context_id
        updater = TaskUpdater(event_queue, task_id, context_id)
        if context.current_task is None:
            # First event of a task must be the Task itself (state submitted).
            await event_queue.enqueue_event(new_task_from_user_message(context.message))
        await updater.start_work()

        text = context.get_user_input()
        data = get_data_parts(context.message.parts)
        history = data[0].get("history", []) if data else []
        customer_id = context.metadata.get("customer_id", "")
        thread_id = context.metadata.get("thread_id", context_id)
        if auth_enabled():
            verified = await self._verify_customer(context, customer_id, task_id)
            if verified is None:
                await updater.failed(new_text_message(REJECTED_MESSAGE))
                return
            customer_id = verified
        logger.info(
            "agent=%s status=start task_id=%s context_id=%s customer_id=%s history=%d",
            self._agent, task_id, context_id, customer_id, len(history),
        )
        start = time.perf_counter()
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

        worker = asyncio.create_task(asyncio.to_thread(produce))
        held: Chunk | None = None  # newest answer chunk, held back until we know whether it is the last
        sent_answer, events, reply_chars = False, 0, 0

        async def flush(last: bool) -> None:
            """Publish the held answer chunk as an artifact append.

            Args:
                last (bool): True when no answer chunk follows, so it is flagged ``last_chunk``.
            """
            nonlocal held, sent_answer, reply_chars
            if held is None:
                return
            await updater.add_artifact(
                [new_text_part(held.text)], artifact_id="reply", name="reply", append=sent_answer, last_chunk=last
            )
            sent_answer, reply_chars, held = True, reply_chars + len(held.text), None

        while (item := await outbox.get()) is not None:
            if isinstance(item, Exception):
                await worker
                logger.error(
                    "agent=%s status=failed task_id=%s context_id=%s error=%s",
                    self._agent, task_id, context_id, type(item).__name__,
                )
                await updater.failed(new_text_message("The agent could not process this request."))
                return
            events += 1
            if item.kind == "progress":
                await updater.update_status(TaskState.TASK_STATE_WORKING, new_text_message(item.text))
            else:
                await flush(last=False)
                held = item
        await worker
        await flush(last=True)
        await updater.complete()
        logger.info(
            "agent=%s status=complete task_id=%s context_id=%s duration_s=%.2f chunks=%d reply_chars=%d",
            self._agent, task_id, context_id, time.perf_counter() - start, events, reply_chars,
        )

    async def _verify_customer(self, context: RequestContext, claimed_id: str, task_id: str) -> str | None:
        """Verify the user token of a request and check it belongs to the claimed customer.

        Args:
            context (RequestContext): Incoming request; its metadata may carry the user token.
            claimed_id (str): ``customer_id`` the caller put in the request metadata.
            task_id (str): Task id, for log correlation.

        Returns:
            str | None: The verified customer id, or None when the token is missing, invalid or
            belongs to a different customer than ``claimed_id``.
        """
        token = context.metadata.get(USER_TOKEN_METADATA_KEY, "")
        try:
            # The first call may fetch the signing keys over the network, so keep it off the event loop.
            verified = await asyncio.to_thread(verify_token, token)
        except AuthError as exc:
            logger.warning("agent=%s status=auth_rejected task_id=%s reason=%s", self._agent, task_id, exc.reason)
            return None
        if verified.customer_id != claimed_id:
            logger.warning("agent=%s status=auth_rejected task_id=%s reason=customer_mismatch", self._agent, task_id)
            return None
        return verified.customer_id

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel a task. Runs are short and blocking, so cancellation only marks the task cancelled.

        Args:
            context (RequestContext): Request context holding the task to cancel.
            event_queue (EventQueue): Queue the cancellation event is published to.
        """
        await TaskUpdater(event_queue, context.task_id, context.context_id).cancel()


def build_a2a_app(card: AgentCard, agent: str, runner: Runner) -> Starlette:
    """Build the ASGI app that serves an agent over A2A (JSON-RPC plus the Agent Card).

    Args:
        card (AgentCard): Card served at ``/.well-known/agent-card.json``.
        agent (str): Agent name, used in log fields.
        runner (Runner): Generator function that produces the reply as chunks.

    Returns:
        Starlette: App to pass to ``uvicorn.run``. Tasks and push configs are kept in memory, so
        they are lost on restart.
    """
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
        **push_kwargs,
    )
    routes = [
        *create_agent_card_routes(card, card_url=AGENT_CARD_WELL_KNOWN_PATH),
        *create_jsonrpc_routes(handler, rpc_url=RPC_PATH),
    ]
    return Starlette(routes=routes)
