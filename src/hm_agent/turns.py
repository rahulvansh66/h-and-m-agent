"""One supervisor turn as a stream of events, shared by the CLI and the API.

Role in the system: both front ends (``hm_agent.cli`` and ``hm_agent.api``) run a customer message
through the compiled supervisor graph the same way. This module builds the per-turn run config
(thread, customer, Langfuse tracing) and turns ``graph.stream`` output into small event dicts:

    {"type": "agent_selected", "agent": ...}   the supervisor chose an agent
    {"type": "started",  "agent": ..., "task_id": ...}   the agent accepted the task
    {"type": "progress", "agent": ..., "text": ...}      a status line while the agent works
    {"type": "token",    "agent": ..., "text": ...}      a piece of the reply, as it is produced
    {"type": "reply",    "agent": ..., "text": ...}      the full reply (always the last event)
"""

import logging
from collections.abc import Iterator

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langfuse.langchain import CallbackHandler
from langgraph.graph.state import CompiledStateGraph

from hm_agent.core.settings import TRACING_TAGS

logger = logging.getLogger(__name__)


def build_config(customer_id: str, thread_id: str) -> RunnableConfig:
    """Build the per-turn run config (thread, customer and Langfuse tracing metadata).

    A fresh ``CallbackHandler`` is created per turn so each turn is its own trace, while the
    shared ``langfuse_session_id`` groups all turns of the thread into one Langfuse session.

    Args:
        customer_id (str): Customer the conversation belongs to, e.g. ``cust001``.
        thread_id (str): Conversation thread id used by the checkpointer and Langfuse.

    Returns:
        RunnableConfig: Config to pass to ``graph.stream`` / ``graph.invoke``.
    """
    return {
        "configurable": {"customer_id": customer_id, "thread_id": thread_id},
        # The handler traces every graph node, LLM call and tool call of this turn.
        "callbacks": [CallbackHandler()],
        "run_name": "hm-supervisor",
        "metadata": {
            "langfuse_session_id": thread_id,
            "langfuse_user_id": customer_id,
            "langfuse_tags": TRACING_TAGS,
        },
    }


def stream_turn(graph: CompiledStateGraph, query: str, customer_id: str, thread_id: str) -> Iterator[dict]:
    """Run one customer message through the graph and yield events as they happen.

    Args:
        graph (CompiledStateGraph): Compiled supervisor graph (with a checkpointer for multi-turn use).
        query (str): The customer's message for this turn.
        customer_id (str): Customer the conversation belongs to.
        thread_id (str): Conversation thread id; earlier turns are loaded from the checkpointer.

    Yields:
        dict: Event dicts as described in the module docstring. The last one has ``type == "reply"``
        (it is missing only when the graph produced no AI message).
    """
    stream = graph.stream(
        {"messages": [HumanMessage(query)]},
        build_config(customer_id, thread_id),
        stream_mode=["updates", "custom"],
    )
    for mode, data in stream:
        if mode == "custom":
            yield data  # written by the remote-agent node: started / progress / token
            continue
        for node, update in data.items():
            if node == "supervisor":
                yield {"type": "agent_selected", "agent": update["next_agent"]}
            elif update and update.get("messages"):
                yield {"type": "reply", "agent": node, "text": update["messages"][-1].text}
