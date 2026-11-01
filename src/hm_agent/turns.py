"""One supervisor turn as a stream of events, shared by the CLI and the runtime entrypoint.

Role in the system: both front ends (``hm_agent.cli`` and ``hm_agent.runtime``) run a customer message
through the compiled supervisor graph the same way. This module builds the per-turn run config
(thread, customer, Langfuse tracing) and turns ``graph.stream`` output into small event dicts:

    {"type": "agent_selected", "agent": ...}   the supervisor chose an agent
    {"type": "started",  "agent": ..., "task_id": ...}   the agent accepted the task
    {"type": "progress", "agent": ..., "text": ...}      a status line while the agent works
    {"type": "token",    "agent": ..., "text": ...}      a piece of the reply, as it is produced
    {"type": "reply",    "agent": ..., "text": ...}      the full reply (always the last event)

A message split between several agents gets one ``agent_selected`` per agent and ``started`` /
``progress`` events labelled by agent. Their answers stream as ``token`` events with agent
``multi_agent``, one section at a time (each behind a short lead-in), starting as soon as the first
agent answers; the last event is one ``reply`` with the whole combined text.

Each turn also logs one ``turn status=done ... ttft_s=... total_s=...`` line (time to the first token
and total time) for log-based latency metrics.
"""

import logging
import time
from collections.abc import Iterator

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langfuse.langchain import CallbackHandler
from langgraph.graph.state import CompiledStateGraph

from hm_agent.auth import USER_TOKEN_CONFIG_KEY
from hm_agent.core.settings import TRACING_TAGS

logger = logging.getLogger(__name__)


def build_config(customer_id: str, thread_id: str, user_token: str | None = None) -> RunnableConfig:
    """Build the per-turn run config (thread, customer and Langfuse tracing metadata).

    A fresh ``CallbackHandler`` is created per turn so each turn is its own trace, while the
    shared ``langfuse_session_id`` groups all turns of the thread into one Langfuse session.

    Args:
        customer_id (str): Customer the conversation belongs to, e.g. ``cust001``.
        thread_id (str): Conversation thread id used by the checkpointer and Langfuse.
        user_token (str | None): The end user's access token, forwarded to the agents so they can verify
            it. Kept out of the checkpoint and Langfuse metadata; never log it.

    Returns:
        RunnableConfig: Config to pass to ``graph.stream`` / ``graph.invoke``.
    """
    configurable = {"customer_id": customer_id, "thread_id": thread_id}
    if user_token:
        configurable[USER_TOKEN_CONFIG_KEY] = user_token
    return {
        "configurable": configurable,
        # The handler traces every graph node, LLM call and tool call of this turn.
        "callbacks": [CallbackHandler()],
        "run_name": "hm-supervisor",
        "metadata": {
            "langfuse_session_id": thread_id,
            "langfuse_user_id": customer_id,
            "langfuse_tags": TRACING_TAGS,
        },
    }


def stream_turn(
    graph: CompiledStateGraph, query: str, customer_id: str, thread_id: str, user_token: str | None = None
) -> Iterator[dict]:
    """Run one customer message through the graph and yield events as they happen.

    Args:
        graph (CompiledStateGraph): Compiled supervisor graph (with a checkpointer for multi-turn use).
        query (str): The customer's message for this turn.
        customer_id (str): Customer the conversation belongs to.
        thread_id (str): Conversation thread id; earlier turns are loaded from the checkpointer.
        user_token (str | None): End user's access token to forward to the agents (None when auth is off).

    Yields:
        dict: Event dicts as described in the module docstring. The last one has ``type == "reply"``
        (it is missing only when the graph produced no AI message).
    """
    start = time.perf_counter()
    first_token_s: float | None = None  # time to first token (ttft), set once on the first "token" event
    agents = 0  # number of agents the supervisor picked; more than one means a split turn
    # Two stream modes at once: "updates" gives per-node state changes (routing, final replies), "custom"
    # gives the events the nodes write themselves (started / progress / token).
    stream = graph.stream(
        {"messages": [HumanMessage(query)]},
        build_config(customer_id, thread_id, user_token),
        stream_mode=["updates", "custom"],
    )
    for mode, data in stream:
        if mode == "custom":
            if data.get("type") == "token" and first_token_s is None:
                first_token_s = time.perf_counter() - start
            yield data  # written by the remote-agent node: started / progress / token
            continue
        # "updates" payload: {node_name: state_update}.
        for node, update in data.items():
            if node == "supervisor":
                for task in update["tasks"]:
                    agents += 1
                    yield {"type": "agent_selected", "agent": task.agent}
            elif update and update.get("messages"):
                # Branch nodes of a split turn write ``results`` (no reply); ``synthesize`` writes the combined one.
                message = update["messages"][-1]
                yield {"type": "reply", "agent": message.name or node, "text": message.text}
    # One line per turn, shaped for log-based metrics: alert on a high p95 of ttft_s, split-turn share, etc.
    logger.info(
        "turn status=done thread_id=%s split=%s agents=%d ttft_s=%s total_s=%.2f",
        thread_id,
        agents > 1,
        agents,
        f"{first_token_s:.2f}" if first_token_s is not None else "none",
        time.perf_counter() - start,
    )
