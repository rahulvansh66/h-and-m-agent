"""Tests for the thread helpers of the AgentCore entrypoint, using a fake graph (no Neon)."""

from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from hm_agent.runtime.threads import history_messages, thread_owner


class FakeGraph:
    """Graph stub whose ``get_state`` returns a fixed snapshot."""

    def __init__(self, values: dict, metadata: dict | None = None) -> None:
        """Store the snapshot to return.

        Args:
            values (dict): State values of the snapshot.
            metadata (dict | None): Checkpoint metadata of the snapshot.
        """
        self._snapshot = SimpleNamespace(values=values, metadata=metadata)

    def get_state(self, config: dict) -> SimpleNamespace:
        """Return the fixed snapshot.

        Args:
            config (dict): Run config (ignored).

        Returns:
            SimpleNamespace: Snapshot with ``values`` and ``metadata``.
        """
        return self._snapshot


def test_thread_owner_returns_recorded_customer() -> None:
    """The owner is the customer id stored in the checkpoint metadata."""
    graph = FakeGraph({"messages": [1]}, {"customer_id": "cust001"})
    assert thread_owner(graph, "t1") == "cust001"


def test_thread_owner_unknown_and_unowned_threads() -> None:
    """No stored state gives None; stored state without an owner gives an empty string."""
    assert thread_owner(FakeGraph({}), "t1") is None
    assert thread_owner(FakeGraph({"messages": [1]}, {}), "t1") == ""


def test_history_messages_maps_roles_and_agents() -> None:
    """Human messages become user turns; others carry the answering agent's name."""
    graph = FakeGraph({"messages": [HumanMessage("hi"), AIMessage("hello", name="fashion")]})
    assert history_messages(graph, "t1") == [
        {"role": "user", "agent": None, "text": "hi"},
        {"role": "assistant", "agent": "fashion", "text": "hello"},
    ]
