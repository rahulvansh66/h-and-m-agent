"""Generic A2A server: serves any streaming "text in, chunks out" runner as an A2A agent.

Role in the system: every agent service (order management, customer support, shipment tracking) is
this wrapper around its own runner. It turns an incoming ``SendMessage`` into a Task, runs the
async runner on the event loop, and streams its
output through the task lifecycle: submitted -> working (progress messages) -> artifact chunks ->
completed (or failed). Streaming clients see each event as it happens; blocking clients get the
assembled task at the end.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable

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
from langgraph.graph.state import CompiledStateGraph
from starlette.applications import Starlette
from starlette.types import Lifespan

from hm_agent.a2a.chunks import Chunk, answer, progress
from hm_agent.auth import USER_TOKEN_METADATA_KEY, AuthError, verify_token
from hm_agent.core.logging_setup import trace_id_var
from hm_agent.core.settings import auth_enabled
from hm_agent.core.tracing import (
    TRACE_ID_CONFIG_KEY,
    TRACE_ID_METADATA_KEY,
    build_langfuse_config,
    incoming_trace_id,
)
from hm_agent.evals.online import compute_agent_scores, publish_scores
from hm_agent.evals.toolcalls import extract_tool_calls
from hm_agent.evals.usage import summarise_usage, usage_handler

logger = logging.getLogger(__name__)

RPC_PATH = "/"  # JSON-RPC endpoint; the Agent Card tells clients where it is
REJECTED_MESSAGE = "The agent could not process this request."  # generic on purpose: no hint about why
TOKEN_FLUSH_CHARS = 24  # LLM tokens are grouped into pieces of at least this size, one A2A event per piece

# Runner signature: (latest user text, prior messages, customer_id, thread_id) -> stream of chunks.
Runner = Callable[[str, list[dict], str, str], AsyncIterator[Chunk]]


def _score_run(agent: str, config: RunnableConfig, new_messages: list[AnyMessage], duration_s: float) -> None:
    """Write the agent-run scores (latency, tool health, tokens, notional cost) onto the turn's trace.

    Args:
        agent (str): Agent name.
        config (RunnableConfig): Run config of the finished run (holds the trace id and the usage handler).
        new_messages (list[AnyMessage]): Messages the run added after the supervisor's history.
        duration_s (float): Wall time of the run.
    """
    trace_id = config["configurable"].get(TRACE_ID_CONFIG_KEY) or getattr(config["callbacks"][0], "last_trace_id", None)
    handler = usage_handler(config)
    usage = summarise_usage(handler.usage_metadata) if handler else {}
    publish_scores(trace_id, compute_agent_scores(agent, extract_tool_calls(new_messages), duration_s, usage))


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
        Runner: Async generator function ``(text, history, customer_id, thread_id) -> AsyncIterator[Chunk]``.
    """
    labels = progress_labels or {}

    async def run(text: str, history: list[dict], customer_id: str, thread_id: str) -> AsyncIterator[Chunk]:
        """Stream the graph for one A2A request.

        Args:
            text (str): Latest customer message.
            history (list[dict]): Prior messages as ``{"role", "text", "name"}`` dicts, oldest first.
            customer_id (str): Customer the conversation belongs to (read by the MCP tool interceptor).
            thread_id (str): Conversation thread id (Langfuse session).

        Yields:
            Chunk: Progress lines and answer pieces, in order.
        """
        # The agent is stateless: the supervisor sends the recent turns on every call.
        messages: list[AnyMessage] = [
            HumanMessage(m["text"]) if m["role"] == "user" else AIMessage(m["text"], name=m.get("name"))
            for m in history
        ]
        messages.append(HumanMessage(text))

        # Joining the supervisor's trace makes one turn a single trace across services.
        trace_id = incoming_trace_id.get() or None
        if trace_id:
            trace_id_var.set(trace_id)
        config: RunnableConfig = build_langfuse_config(
            customer_id, thread_id, f"a2a-{agent}", extra_tags=[f"a2a-{agent}"], trace_id=trace_id
        )

        # ``final_text`` is the fallback reply when no tokens were streamed.
        buffer, streamed, final_text = "", False, ""
        final_messages: list[AnyMessage] = []  # used for tool-call scoring
        run_start = time.perf_counter()
        try:
            async for mode, data in graph.astream(
                {"messages": messages}, config, stream_mode=["messages", "updates", "values"]
            ):
                if mode == "messages":
                    token, _meta = data

                    # Tool-call fragments are not part of the reply, so only plain text is kept.
                    if stream_tokens and isinstance(token, AIMessageChunk) and token.text and not token.tool_call_chunks:
                        buffer += token.text

                        # Batching avoids one A2A event per token.
                        if len(buffer) >= TOKEN_FLUSH_CHARS:
                            yield answer(buffer)
                            buffer, streamed = "", True
                elif mode == "updates":
                    for node in data:
                        if node in labels:
                            yield progress(labels[node])
                else:
                    final_text = data["messages"][-1].text
                    final_messages = data["messages"]
            if buffer:
                yield answer(buffer)
                streamed = True
            if not streamed:
                yield answer(final_text)
            await asyncio.to_thread(
                _score_run, agent, config, final_messages[len(messages):], time.perf_counter() - run_start
            )
        finally:
            # Flush now: each request is a short trace that could be lost on shutdown.
            # The flush is a blocking network call, so keep it off the event loop.
            await asyncio.to_thread(get_client().flush)

    return run


def deferred_runner(get_runner: Callable[[], Runner]) -> Runner:
    """Wrap a runner that does not exist yet, so the app can be built before the service's lifespan runs.

    The agent's MCP session (and so its graph) is only created when the server starts, but the app is
    built earlier. The real runner is looked up on every request.

    Args:
        get_runner (Callable[[], Runner]): Returns the live runner; raises if the service is not started.

    Returns:
        Runner: Async generator function that forwards each request to ``get_runner()``.
    """

    async def run(text: str, history: list[dict], customer_id: str, thread_id: str) -> AsyncIterator[Chunk]:
        async for chunk in get_runner()(text, history, customer_id, thread_id):
            yield chunk

    return run


class RunnerExecutor(AgentExecutor):
    """A2A executor that streams the chunks of an async ``Runner`` as task events."""

    def __init__(self, agent: str, runner: Runner) -> None:
        """Store the agent name and runner.

        Args:
            agent (str): Agent name, used in log fields.
            runner (Runner): Async generator function that produces the reply as chunks.
        """
        self._agent = agent
        self._runner = runner

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Run one task: publish submitted/working, stream the runner's chunks, then complete.

        Reads the customer id and thread id from the request metadata and the prior messages from
        the first data part of the message. The runner is an async generator iterated on the event
        loop. A ``progress`` chunk becomes a
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

        # The first event of a task must be the Task itself (state submitted).
        if context.current_task is None:
            await event_queue.enqueue_event(new_task_from_user_message(context.message))
        await updater.start_work()

        # Text part = latest question; data part = prior turns (see the client).
        text = context.get_user_input()
        data = get_data_parts(context.message.parts)
        history = data[0].get("history", []) if data else []
        customer_id = context.metadata.get("customer_id", "")
        thread_id = context.metadata.get("thread_id", context_id)
        trace_id = context.metadata.get(TRACE_ID_METADATA_KEY, "")

        # SECURITY: the claimed customer_id is replaced by the one proven by the token.
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

        # Same task as the runner, so the ContextVar is visible to it without copying.
        incoming_trace_id.set(trace_id)
        held: Chunk | None = None  # newest answer chunk, not yet sent
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

        # Progress goes out at once; an answer chunk waits one step, since only the next item shows it was not last.
        try:
            async for item in self._runner(text, history, customer_id, thread_id):
                events += 1
                if item.kind == "progress":
                    await updater.update_status(TaskState.TASK_STATE_WORKING, new_text_message(item.text))
                else:
                    await flush(last=False)
                    held = item
        except Exception as exc:  # reported to the task, never to the caller
            logger.error(
                "agent=%s status=failed task_id=%s context_id=%s error=%s",
                self._agent, task_id, context_id, type(exc).__name__,
            )

            # Generic text on purpose; details stay in the server log.
            await updater.failed(new_text_message(REJECTED_MESSAGE))
            return
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
            # The first call may fetch signing keys over the network, so keep it off the event loop.
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


def build_a2a_app(card: AgentCard, agent: str, runner: Runner, lifespan: Lifespan[Starlette] | None = None) -> Starlette:
    """Build the ASGI app that serves an agent over A2A (JSON-RPC plus the Agent Card).

    Args:
        card (AgentCard): Card served at ``/.well-known/agent-card.json``.
        agent (str): Agent name, used in log fields.
        runner (Runner): Async generator function that produces the reply as chunks.
        lifespan (Lifespan[Starlette] | None): Startup/shutdown context run on the server's event loop,
            used to hold resources (such as an MCP session) open while the app serves.

    Returns:
        Starlette: App to pass to ``uvicorn.run``. Tasks and push configs are kept in memory, so
        they are lost on restart.
    """
    push_kwargs = {}
    if card.capabilities.push_notifications:
        # The caller registers a webhook per task; the sender POSTs every update to it.
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
    return Starlette(routes=routes, lifespan=lifespan)
