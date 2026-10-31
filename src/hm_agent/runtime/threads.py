"""Read-only views of stored conversation threads for the AgentCore entrypoint.

Role in the system: the supervisor graph's checkpointer holds every conversation. The entrypoint uses
these helpers to check which customer owns a thread (authorization) and to return its history.
"""

from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph


def thread_owner(graph: CompiledStateGraph, thread_id: str) -> str | None:
    """Return the customer a stored thread belongs to.

    Every checkpoint written by a turn records the ``customer_id`` it was written with.

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
