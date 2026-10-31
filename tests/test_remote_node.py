"""Tests for the remote-agent graph node: streamed result, push fallback and failure.

Role in the system: checks how ``make_remote_agent_node`` turns the client's event stream into a
graph reply, forwards custom stream events, and falls back to the push receiver when the stream
breaks. The client and receiver are replaced by fakes.
"""

import pytest
from langchain_core.messages import HumanMessage

from hm_agent.a2a import remote_node
from hm_agent.a2a.client import PushTarget, RemoteEvent
from hm_agent.a2a.push_receiver import COMPLETED_STATE, PendingTask

AGENT = "shipment_tracking"


class FakeReceiver:
    """Stand-in for ``PushReceiver`` that records what the node does with it."""

    def __init__(self) -> None:
        """Create a receiver with empty records."""
        self.target = PushTarget("http://hook", "tok")
        self.registered: list[PendingTask] = []
        self.streamed: list[str] = []

    def register(self, task: PendingTask) -> None:
        """Record a registered task.

        Args:
            task (PendingTask): The task handed over.
        """
        self.registered.append(task)

    def mark_streamed(self, task_id: str) -> None:
        """Record a task whose result arrived over the stream.

        Args:
            task_id (str): Task id.
        """
        self.streamed.append(task_id)


@pytest.fixture
def setup(monkeypatch: pytest.MonkeyPatch):
    """Patch the node's collaborators and return handles to inspect them.

    Args:
        monkeypatch (pytest.MonkeyPatch): Used to replace the client, receiver and stream writer.

    Returns:
        tuple: ``(receiver, written, run)`` where ``run(events_or_exc)`` calls the node.
    """
    receiver, written = FakeReceiver(), []
    monkeypatch.setattr(remote_node, "get_receiver", lambda: receiver)
    monkeypatch.setattr(remote_node, "get_stream_writer", lambda: written.append)

    def run(script) -> dict:
        def fake_stream(*args, **kwargs):
            for item in script:
                if isinstance(item, Exception):
                    raise item
                yield item

        monkeypatch.setattr(remote_node, "stream_remote_agent", fake_stream)
        node = remote_node.make_remote_agent_node(AGENT)
        config = {"configurable": {"customer_id": "cust001", "thread_id": "t1"}}
        return node({"messages": [HumanMessage("Where is ORD-1007?")]}, config)

    return receiver, written, run


def test_streamed_reply_is_joined_forwarded_and_webhook_ignored(setup) -> None:
    """Pieces are forwarded as custom events, joined into one reply, and the later webhook is ignored."""
    receiver, written, run = setup
    out = run(
        [
            RemoteEvent("started", task_id="task-1", push_registered=True),
            RemoteEvent("progress", text="Contacting"),
            RemoteEvent("answer", text="It is "),
            RemoteEvent("answer", text="on its way."),
            RemoteEvent("state", state=COMPLETED_STATE),
        ]
    )
    assert out["messages"][0].text == "It is on its way."
    assert out["messages"][0].name == AGENT
    assert [w["type"] for w in written] == ["started", "progress", "token", "token"]
    assert receiver.streamed == ["task-1"] and receiver.registered == []


def test_stream_break_falls_back_to_push(setup) -> None:
    """If the stream dies before a terminal state, the task is handed to the push receiver."""
    receiver, _, run = setup
    out = run([RemoteEvent("started", task_id="task-2", push_registered=True), ConnectionError("dropped")])
    assert out["messages"][0].text == remote_node.PENDING_REPLY
    assert [t.task_id for t in receiver.registered] == ["task-2"]
    assert receiver.streamed == []


def test_failure_without_push_returns_apology(setup) -> None:
    """Without a push registration there is nothing to wait for, so the user gets the apology."""
    receiver, _, run = setup
    out = run([RemoteEvent("started", task_id="task-3", push_registered=False), ConnectionError("dropped")])
    assert out["messages"][0].text == remote_node.UNAVAILABLE_REPLY
    assert receiver.registered == []


def test_failed_task_returns_apology_and_ignores_webhook(setup) -> None:
    """A task that ends failed is not retried through push; its notifications are ignored."""
    receiver, _, run = setup
    out = run([RemoteEvent("started", task_id="task-4", push_registered=True), RemoteEvent("state", state="TASK_STATE_FAILED")])
    assert out["messages"][0].text == remote_node.UNAVAILABLE_REPLY
    assert receiver.streamed == ["task-4"] and receiver.registered == []


def test_sub_query_replaces_latest_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a split turn the agent receives its own sub-query, with the earlier turns as history."""
    sent: list[tuple] = []

    def fake_stream(url, text, history, *args, **kwargs):
        sent.append((text, history))
        yield RemoteEvent("state", state=COMPLETED_STATE)

    monkeypatch.setattr(remote_node, "get_receiver", lambda: None)
    monkeypatch.setattr(remote_node, "get_stream_writer", lambda: lambda event: None)
    monkeypatch.setattr(remote_node, "stream_remote_agent", fake_stream)
    node = remote_node.make_remote_agent_node(AGENT)
    config = {"configurable": {"customer_id": "cust001", "thread_id": "t1"}}
    messages = [HumanMessage("hi"), HumanMessage("Where is ORD-1007, and what goes with my jacket?")]
    node({"messages": messages, "query": "Where is ORD-1007?"}, config)
    node({"messages": messages, "query": None}, config)
    assert sent[0][0] == "Where is ORD-1007?" and [h["text"] for h in sent[0][1]] == ["hi"]
    assert sent[1][0] == "Where is ORD-1007, and what goes with my jacket?"


def test_user_token_from_run_config_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    """The token in the run config reaches the client call; without one an empty token is passed."""
    from hm_agent.auth import USER_TOKEN_CONFIG_KEY

    seen: list[str] = []

    def fake_stream(*args, user_token="", **kwargs):
        seen.append(user_token)
        yield RemoteEvent("state", state=COMPLETED_STATE)

    monkeypatch.setattr(remote_node, "get_receiver", lambda: None)
    monkeypatch.setattr(remote_node, "get_stream_writer", lambda: lambda event: None)
    monkeypatch.setattr(remote_node, "stream_remote_agent", fake_stream)
    node = remote_node.make_remote_agent_node(AGENT)
    state = {"messages": [HumanMessage("hi")]}
    node(state, {"configurable": {"customer_id": "cust001", "thread_id": "t1", USER_TOKEN_CONFIG_KEY: "tok"}})
    node(state, {"configurable": {"customer_id": "cust001", "thread_id": "t1"}})
    assert seen == ["tok", ""]
