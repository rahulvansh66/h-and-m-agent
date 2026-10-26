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
from hm_agent.core.state import CUSTOMER_SUPPORT, FASHION, ORDER_MANAGEMENT, SHIPMENT_TRACKING

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
    for agent in (ORDER_MANAGEMENT, CUSTOMER_SUPPORT, SHIPMENT_TRACKING, FASHION):
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


@pytest.fixture
def auth_on(monkeypatch: pytest.MonkeyPatch):
    """Switch agent-side token verification on with a fake verifier.

    The fake accepts the token ``good-<customer>`` and rejects everything else.

    Args:
        monkeypatch (pytest.MonkeyPatch): Used to replace the server's auth hooks.
    """
    from hm_agent.a2a import server
    from hm_agent.auth import AuthError, VerifiedUser

    def fake_verify(token: str) -> VerifiedUser:
        if not token.startswith("good-"):
            raise AuthError("invalid_token")
        return VerifiedUser(customer_id=token.removeprefix("good-"))

    monkeypatch.setattr(server, "auth_enabled", lambda: True)
    monkeypatch.setattr(server, "verify_token", fake_verify)


def run_stream(token: str, customer_id: str = "cust001") -> list:
    """Stream one message to the served fake agent, claiming ``customer_id`` with ``token``.

    Args:
        token (str): User token to forward ("" sends none).
        customer_id (str): Customer id placed in the request metadata.

    Returns:
        list: The events the client received.
    """
    return list(a2a_client.stream_remote_agent(AGENT_URL, "hi", [], customer_id, "t1", 10, user_token=token))


def test_matching_token_runs_the_agent(serve, auth_on) -> None:
    """A token for the claimed customer is accepted and the agent runs for that customer."""
    serve(fake_runner)
    events = run_stream("good-cust001")
    assert events[-1].state == "TASK_STATE_COMPLETED"
    assert "".join(e.text for e in events if e.kind == "answer").endswith("cust001/t1")


def test_token_for_another_customer_is_rejected(serve, auth_on) -> None:
    """A valid token for cust002 cannot be used to act as cust001: the task fails and the agent never runs."""
    serve(fake_runner)
    events = run_stream("good-cust002", customer_id="cust001")
    assert [e.kind for e in events if e.kind == "answer"] == []
    assert events[-1].state == "TASK_STATE_FAILED"


def test_missing_token_is_rejected_when_auth_is_enabled(serve, auth_on) -> None:
    """No token at all fails the task when auth is on."""
    serve(fake_runner)
    assert run_stream("")[-1].state == "TASK_STATE_FAILED"


def test_missing_token_is_fine_when_auth_is_disabled(serve) -> None:
    """With auth off (the local default) behaviour is unchanged: no token needed."""
    serve(fake_runner)
    assert run_stream("")[-1].state == "TASK_STATE_COMPLETED"


def test_wire_log_redacts_the_user_token(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The optional wire log never records the user token."""
    from hm_agent.a2a import wire_log

    log = tmp_path / "wire.jsonl"
    monkeypatch.setenv(wire_log.WIRE_LOG_ENV, str(log))
    wire_log._write_wire("C->S", "POST http://x", '{"params": {"metadata": {"user_token": "secret-abc", "customer_id": "c"}}}')
    text = log.read_text(encoding="utf-8")
    assert "secret-abc" not in text and "customer_id" in text


def test_bind_address_defaults_and_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local defaults are the configured host and per-agent port; env vars override them for Runtime."""
    from hm_agent.core import settings

    monkeypatch.delenv("A2A_BIND_HOST", raising=False)
    monkeypatch.delenv("A2A_BIND_PORT", raising=False)
    assert settings.bind_address(ORDER_MANAGEMENT) == (settings.A2A_HOST, settings.agent_port(ORDER_MANAGEMENT))
    monkeypatch.setenv("A2A_BIND_HOST", "0.0.0.0")
    monkeypatch.setenv("A2A_BIND_PORT", "9000")
    assert settings.bind_address(ORDER_MANAGEMENT) == ("0.0.0.0", 9000)


def test_card_advertises_the_configured_agent_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Agent Card URL follows ``<AGENT>_AGENT_URL``, independent of the bind address."""
    monkeypatch.setenv("FASHION_AGENT_URL", "https://runtime.example/agent/")
    assert build_agent_card(FASHION).supported_interfaces[0].url == "https://runtime.example/agent"
