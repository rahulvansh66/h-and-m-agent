"""Tests for the AgentCore entrypoint (``hm_agent.runtime.entrypoint``).

Role in the system: drives ``POST /invocations`` and ``GET /ping`` of the entrypoint app in-process with a
fake graph, so event order, identity resolution, thread ownership, history and error handling are checked
without Neon, agents or an LLM.
"""

import json

import pytest
from starlette.testclient import TestClient

from hm_agent.auth import AuthError, VerifiedUser
from hm_agent.runtime import entrypoint

SESSION_HEADER = "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"
SESSION = "session-" + "x" * 30  # Runtime session ids are at least 33 characters


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch):
    """Install a fake graph and fakes for turn streaming and thread ownership, and return a test client.

    Args:
        monkeypatch (pytest.MonkeyPatch): Used to replace the entrypoint's collaborators.

    Returns:
        SimpleNamespace-like object with ``client`` (TestClient), ``owners`` (thread -> customer, edit
        to set up stored threads) and ``calls`` (list of ``(message, customer_id, thread_id, user_token)``).
    """
    owners: dict[str, str] = {}
    calls: list[tuple] = []

    def fake_stream_turn(graph, query, customer_id, thread_id, user_token=None):
        calls.append((query, customer_id, thread_id, user_token))
        yield {"type": "agent_selected", "agent": "order_management"}
        yield {"type": "token", "agent": "order_management", "text": "Hel"}
        yield {"type": "reply", "agent": "order_management", "text": "Hello"}

    monkeypatch.setattr(entrypoint, "stream_turn", fake_stream_turn)
    monkeypatch.setattr(entrypoint, "thread_owner", lambda graph, thread_id: owners.get(thread_id))
    monkeypatch.setattr(entrypoint, "history_messages", lambda graph, thread_id: [{"role": "user", "agent": None, "text": "hi"}])
    monkeypatch.setattr(entrypoint, "auth_enabled", lambda: False)
    monkeypatch.setattr(entrypoint, "get_client", lambda: type("C", (), {"flush": lambda self: None})())
    monkeypatch.setattr(entrypoint._state, "graph", object())

    class Served:
        client = TestClient(entrypoint.app)  # no `with`: the lifespan (Neon) is not started

    Served.owners, Served.calls = owners, calls
    return Served


def events(response) -> list[dict]:
    """Parse the SSE body of an invocation into event dicts.

    Args:
        response: HTTP response of the test client.

    Returns:
        list[dict]: The decoded ``data:`` payloads, in order.
    """
    return [json.loads(line[5:]) for line in response.text.splitlines() if line.startswith("data:")]


def invoke(served, body: dict, headers: dict | None = None):
    """POST an invocation with the session header set.

    Args:
        served: The ``served`` fixture value.
        body (dict): Invocation body.
        headers (dict | None): Extra headers (for example ``Authorization``).

    Returns:
        The HTTP response.
    """
    return served.client.post("/invocations", json=body, headers={SESSION_HEADER: SESSION, **(headers or {})})


def test_ping_is_healthy(served) -> None:
    """The Runtime health probe answers."""
    assert served.client.get("/ping").json()["status"] == "Healthy"


def test_turn_streams_events_in_order_and_ends_with_done(served) -> None:
    """thread -> turn events -> done, with the session id used as the thread id."""
    out = events(invoke(served, {"message": "hello"}))
    assert [e["type"] for e in out] == ["thread", "agent_selected", "token", "reply", "done"]
    assert out[0]["thread_id"] == SESSION
    assert served.calls == [("hello", "cust001", SESSION, None)]  # local: default customer, no token


def test_local_customer_comes_from_the_body(served) -> None:
    """With auth disabled the body's customer_id selects the customer."""
    invoke(served, {"message": "hello", "customer_id": "cust002"})
    assert served.calls[0][1] == "cust002"


def test_auth_enabled_uses_the_verified_token_and_ignores_the_body(served, monkeypatch: pytest.MonkeyPatch) -> None:
    """With auth on, the customer is the token's claim, and the token is forwarded for the agents."""
    seen: list[str] = []

    def fake_verify(token: str) -> VerifiedUser:
        seen.append(token)
        return VerifiedUser(customer_id="cust003")

    monkeypatch.setattr(entrypoint, "auth_enabled", lambda: True)
    monkeypatch.setattr(entrypoint, "verify_token", fake_verify)
    out = events(invoke(served, {"message": "hello", "customer_id": "cust999"}, {"Authorization": "Bearer abc"}))
    assert out[-1]["type"] == "done" and "error" not in [e["type"] for e in out]
    assert seen == ["abc"]
    assert served.calls == [("hello", "cust003", SESSION, "abc")]


def test_auth_enabled_rejects_a_bad_or_missing_token(served, monkeypatch: pytest.MonkeyPatch) -> None:
    """An invalid token yields an unauthorized error and the graph is never run."""

    def reject(token: str) -> VerifiedUser:
        raise AuthError("invalid_token")

    monkeypatch.setattr(entrypoint, "auth_enabled", lambda: True)
    monkeypatch.setattr(entrypoint, "verify_token", reject)
    for headers in ({"Authorization": "Bearer bad"}, {}):
        out = events(invoke(served, {"message": "hello"}, headers))
        assert out == [{"type": "error", "detail": entrypoint.UNAUTHORIZED_DETAIL}, {"type": "done"}]
    assert served.calls == []


def test_other_customers_thread_is_denied(served) -> None:
    """Continuing a thread owned by someone else is refused with the generic not-found error."""
    served.owners[SESSION] = "cust002"
    out = events(invoke(served, {"message": "hello", "customer_id": "cust001"}))
    assert [e["type"] for e in out] == ["thread", "error", "done"]
    assert out[1]["detail"] == entrypoint.THREAD_NOT_FOUND_DETAIL
    assert served.calls == []


def test_own_existing_thread_can_continue(served) -> None:
    """The owner can continue their thread."""
    served.owners[SESSION] = "cust001"
    assert events(invoke(served, {"message": "again"}))[-2]["type"] == "reply"


def test_history_action_returns_messages(served) -> None:
    """``{"action": "history"}`` returns the stored messages and runs no turn."""
    out = events(invoke(served, {"action": "history"}))
    assert [e["type"] for e in out] == ["thread", "history", "done"]
    assert out[1]["messages"] == [{"role": "user", "agent": None, "text": "hi"}]
    assert served.calls == []


def test_history_of_another_customers_thread_is_denied(served) -> None:
    """History obeys the same ownership rule as turns."""
    served.owners[SESSION] = "cust002"
    out = events(invoke(served, {"action": "history", "customer_id": "cust001"}))
    assert [e["type"] for e in out] == ["thread", "error", "done"]


@pytest.mark.parametrize("body", [{}, {"message": ""}, {"message": "   "}, {"message": 5}, {"message": "x" * 100_000}])
def test_invalid_message_is_a_bad_request_error(served, body: dict) -> None:
    """Missing, blank, non-text or oversized messages produce an error event, not a turn."""
    out = events(invoke(served, body))
    assert [e["type"] for e in out] == ["thread", "error", "done"]
    assert out[1]["detail"] == entrypoint.BAD_REQUEST_DETAIL
    assert served.calls == []


def test_failing_turn_ends_with_error_then_done(served, monkeypatch: pytest.MonkeyPatch) -> None:
    """An exception in the graph is reported as a generic error event (no internals) and the stream still ends."""

    def boom(*args, **kwargs):
        yield {"type": "agent_selected", "agent": "fashion"}
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(entrypoint, "stream_turn", boom)
    out = events(invoke(served, {"message": "hello"}))
    assert [e["type"] for e in out] == ["thread", "agent_selected", "error", "done"]
    assert out[2]["detail"] == entrypoint.ERROR_DETAIL
    assert "secret" not in json.dumps(out)
