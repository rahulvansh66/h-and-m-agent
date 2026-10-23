"""Tests for A2A streaming: executor -> SSE -> client, run in-process over an ASGI transport.

Role in the system: checks that a runner's chunks become the right A2A events (progress as status
updates, answer pieces as appended artifact chunks) and that both the streaming and the blocking
client read them back. No network, no LLM.
"""

import httpx
import pytest
from a2a.types.a2a_pb2 import AgentCapabilities, AgentCard

from hm_agent.a2a import client as a2a_client
from hm_agent.a2a.cards import build_agent_card
from hm_agent.a2a.chunks import Chunk, answer, progress
from hm_agent.a2a.server import build_a2a_app
from hm_agent.core.state import CUSTOMER_SUPPORT, ORDER_MANAGEMENT, SHIPMENT_TRACKING

AGENT_URL = "http://agent.test"


def fake_runner(text: str, history: list[dict], customer_id: str, thread_id: str):
    """Runner that reports one progress line and a two-piece answer that echoes its inputs.

    Args:
        text (str): Latest customer message.
        history (list[dict]): Prior messages.
        customer_id (str): Customer id.
        thread_id (str): Thread id.

    Yields:
        Chunk: Progress, then two answer pieces.
    """
    yield progress("working on it")
    yield answer(f"{text}|{len(history)}|")
    yield answer(f"{customer_id}/{thread_id}")


def failing_runner(text: str, history: list[dict], customer_id: str, thread_id: str):
    """Runner that fails after one progress line.

    Args:
        text (str): Latest customer message.
        history (list[dict]): Prior messages.
        customer_id (str): Customer id.
        thread_id (str): Thread id.

    Yields:
        Chunk: One progress chunk, before raising.

    Raises:
        RuntimeError: Always.
    """
    yield progress("starting")
    raise RuntimeError("boom")


@pytest.fixture
def serve(monkeypatch: pytest.MonkeyPatch):
    """Serve a runner as an in-process A2A app and point the client at it.

    Args:
        monkeypatch (pytest.MonkeyPatch): Used to swap the client's ``httpx.AsyncClient``.

    Returns:
        Callable: Function ``(runner, push=False) -> None`` that starts serving the runner.
    """

    def start(runner, push: bool = False) -> None:
        card = build_agent_card(SHIPMENT_TRACKING)
        card.supported_interfaces[0].url = AGENT_URL
        card.capabilities.CopyFrom(AgentCapabilities(streaming=True, push_notifications=push))
        app = build_a2a_app(card, "fake", runner)
        real_client = httpx.AsyncClient

        def in_process_client(**kwargs):
            return real_client(transport=httpx.ASGITransport(app=app), base_url=AGENT_URL, **kwargs)

        monkeypatch.setattr(a2a_client.httpx, "AsyncClient", in_process_client)

    return start


def test_cards_advertise_streaming_and_push_only_for_shipment() -> None:
    """Every card streams; only the shipment agent also advertises push notifications."""
    for agent in (ORDER_MANAGEMENT, CUSTOMER_SUPPORT, SHIPMENT_TRACKING):
        card: AgentCard = build_agent_card(agent)
        assert card.capabilities.streaming
        assert card.capabilities.push_notifications == (agent == SHIPMENT_TRACKING)


def test_stream_yields_progress_then_answer_pieces_then_terminal_state(serve) -> None:
    """The client sees started, progress, both answer pieces in order, then the terminal state."""
    serve(fake_runner)
    events = list(a2a_client.stream_remote_agent(AGENT_URL, "hi", [{"role": "user", "text": "x", "name": None}], "cust001", "t1", 10))
    assert [e.kind for e in events] == ["started", "progress", "answer", "answer", "state"]
    assert events[1].text == "working on it"
    assert "".join(e.text for e in events if e.kind == "answer") == "hi|1|cust001/t1"
    assert events[-1].state == "TASK_STATE_COMPLETED"


def test_blocking_client_gets_the_assembled_reply(serve) -> None:
    """A non-streaming caller still receives the pieces merged into one artifact."""
    serve(fake_runner)
    result = a2a_client.call_remote_agent(AGENT_URL, "hi", [], "cust001", "t1", 10)
    assert result.reply == "hi|0|cust001/t1"


def test_runner_failure_ends_in_failed_state(serve) -> None:
    """An exception in the runner marks the task failed instead of crashing the server."""
    serve(failing_runner)
    events = list(a2a_client.stream_remote_agent(AGENT_URL, "hi", [], "cust001", "t1", 10))
    assert events[-1].kind == "state"
    assert events[-1].state == "TASK_STATE_FAILED"


def test_chunk_helpers() -> None:
    """The helper constructors set the right kind."""
    assert progress("a") == Chunk("progress", "a")
    assert answer("b") == Chunk("answer", "b")
