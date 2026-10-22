"""Generic A2A server: serves any "text in, text out" runner as an A2A agent.

Role in the system: every agent service (order management, customer support, out of scope) is
this wrapper around its own runner. It turns an incoming ``SendMessage`` into a Task, runs the
runner off the event loop (LangGraph graphs and the MCP tool wrapper are blocking), and reports
progress through the task lifecycle: submitted -> working -> artifact -> completed (or failed).
"""

import asyncio
import logging
import time
from collections.abc import Callable

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
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types.a2a_pb2 import AgentCard
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.graph.state import CompiledStateGraph
from starlette.applications import Starlette

from hm_agent.core.settings import TRACING_TAGS

logger = logging.getLogger(__name__)

RPC_PATH = "/"  # JSON-RPC endpoint; the Agent Card tells clients where it is

# Runner signature: (latest user text, prior messages, customer_id, thread_id) -> reply text.
Runner = Callable[[str, list[dict], str, str], str]


def graph_runner(graph: CompiledStateGraph, agent: str) -> Runner:
    """Wrap a compiled LangGraph graph as a ``Runner``.

    The prior messages sent by the supervisor are rebuilt into LangChain messages, the graph is
    invoked with Langfuse tracing (session = thread id, user = customer id), and the last message
    of the result is returned as the reply.

    Args:
        graph (CompiledStateGraph): Agent graph that reads and writes ``messages``.
        agent (str): Agent name, used as the Langfuse run name and tag.

    Returns:
        Runner: Function ``(text, history, customer_id, thread_id) -> reply``.
    """

    def run(text: str, history: list[dict], customer_id: str, thread_id: str) -> str:
        """Invoke the graph for one A2A request.

        Args:
            text (str): Latest customer message.
            history (list[dict]): Prior messages as ``{"role", "text", "name"}`` dicts, oldest first.
            customer_id (str): Customer the conversation belongs to (read by the MCP tool interceptor).
            thread_id (str): Conversation thread id (Langfuse session).

        Returns:
            str: Text of the last message produced by the graph.
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
        try:
            result = graph.invoke({"messages": messages}, config)
        finally:
            # Each request is its own short trace; send it before replying so it is not lost on shutdown.
            get_client().flush()
        return result["messages"][-1].text

    return run


class RunnerExecutor(AgentExecutor):
    """A2A executor that delegates the work of a task to a blocking ``Runner``."""

    def __init__(self, agent: str, runner: Runner) -> None:
        """Store the agent name and runner.

        Args:
            agent (str): Agent name, used in log fields.
            runner (Runner): Blocking function that produces the reply text.
        """
        self._agent = agent
        self._runner = runner

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Run one task: publish submitted/working, run the runner, publish artifact and completion.

        Reads the customer id and thread id from the request metadata and the prior messages from
        the first data part of the message. A runner failure marks the task ``failed`` instead of
        crashing the server.

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
        logger.info(
            "agent=%s status=start task_id=%s context_id=%s customer_id=%s history=%d",
            self._agent, task_id, context_id, customer_id, len(history),
        )
        start = time.perf_counter()
        try:
            reply = await asyncio.to_thread(self._runner, text, history, customer_id, thread_id)
        except Exception:
            logger.exception("agent=%s status=failed task_id=%s context_id=%s", self._agent, task_id, context_id)
            await updater.failed(new_text_message("The agent could not process this request."))
            return
        await updater.add_artifact([new_text_part(reply)], name="reply")
        await updater.complete()
        logger.info(
            "agent=%s status=complete task_id=%s context_id=%s duration_s=%.2f reply_chars=%d",
            self._agent, task_id, context_id, time.perf_counter() - start, len(reply),
        )

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
        runner (Runner): Blocking function that produces the reply text.

    Returns:
        Starlette: App to pass to ``uvicorn.run``. Tasks are kept in memory, so they are lost on restart.
    """
    handler = DefaultRequestHandlerV2(
        agent_executor=RunnerExecutor(agent, runner),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    routes = [
        *create_agent_card_routes(card, card_url=AGENT_CARD_WELL_KNOWN_PATH),
        *create_jsonrpc_routes(handler, rpc_url=RPC_PATH),
    ]
    return Starlette(routes=routes)
