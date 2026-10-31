"""Tests for the FastAPI layer using a fake compiled graph (no Neon, no agents, no LLM).

Role in the system: checks authentication, customer identity, thread ownership, the chat and
streaming routes, rate limiting and health endpoints of ``hm_agent.api``.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from hm_agent import turns
from hm_agent.api import deps
from hm_agent.api.app import create_app
from hm_agent.api.routes import health
from hm_agent.api.service import PushEventHub, RateLimiter, Services
from hm_agent.core.state import AgentTask

KEY = "test-key"
HEADERS = {"X-API-Key": KEY, "X-Customer-Id": "cust001"}


class FakeGraph:
    """Minimal stand-in for the compiled supervisor graph."""

    def __init__(self) -> None:
        """Create a graph with one stored thread (``t-old``, owned by cust001)."""
        self.threads = {
            "t-old": (
                [HumanMessage("hi"), AIMessage("hello", name="order_management")],
                {"customer_id": "cust001"},
            )
        }

    def stream(self, inputs: dict, config: dict, stream_mode: list[str]):
        """Yield the events a real turn on the order agent would produce.

        Args:
            inputs (dict): Graph input (unused).
            config (dict): Run config; its thread id is remembered.
            stream_mode (list[str]): Requested modes (unused).

        Yields:
            tuple: ``(mode, data)`` pairs as ``graph.stream`` does.
        """
        text = inputs["messages"][0].text
        if text == "explode":
            raise RuntimeError("graph failure")
        thread_id = config["configurable"]["thread_id"]
        yield "updates", {"supervisor": {"tasks": [AgentTask(agent="order_management")]}}
        yield "custom", {"type": "started", "agent": "order_management", "task_id": "x"}
        yield "custom", {"type": "token", "agent": "order_management", "text": "You bought "}
        yield "custom", {"type": "token", "agent": "order_management", "text": "a dress."}
        reply = AIMessage("You bought a dress.", name="order_management")
        yield "updates", {"order_management": {"messages": [reply]}}
        owner = config["configurable"]["customer_id"]
        self.threads[thread_id] = ([HumanMessage(text), reply], {"customer_id": owner})

    def get_state(self, config: dict) -> SimpleNamespace:
        """Return the stored state of a thread.

        Args:
            config (dict): Config holding the thread id.

        Returns:
            SimpleNamespace: With ``values`` and ``metadata`` like a real state snapshot.
        """
        messages, metadata = self.threads.get(config["configurable"]["thread_id"], ([], {}))
        return SimpleNamespace(values={"messages": messages} if messages else {}, metadata=metadata)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    """Create a test client around an app with fake services and a known API key.

    Args:
        monkeypatch (pytest.MonkeyPatch): Used to set the key and avoid Langfuse in the run config.

    Yields:
        TestClient: Client with the app lifespan running.
    """
    monkeypatch.setenv(deps.API_KEY_ENV, KEY)
    monkeypatch.setattr(
        turns, "build_config", lambda customer_id, thread_id, user_token=None: {"configurable": {"customer_id": customer_id, "thread_id": thread_id}}
    )
    services = Services(graph=FakeGraph(), limiter=RateLimiter(limit=100), hub=PushEventHub(asyncio.new_event_loop()))
    with TestClient(create_app(services)) as test_client:
        test_client.services = services
        yield test_client


def test_missing_or_wrong_api_key_is_rejected(client: TestClient) -> None:
    """Without the right X-API-Key the API answers 401, whatever the route."""
    body = {"message": "hi"}
    assert client.post("/v1/chat", json=body, headers={"X-Customer-Id": "cust001"}).status_code == 401
    assert client.post("/v1/chat", json=body, headers={**HEADERS, "X-API-Key": "nope"}).status_code == 401


def test_customer_header_is_required_and_validated(client: TestClient) -> None:
    """A missing or malformed customer id is a 422."""
    body = {"message": "hi"}
    assert client.post("/v1/chat", json=body, headers={"X-API-Key": KEY}).status_code == 422
    assert client.post("/v1/chat", json=body, headers={**HEADERS, "X-Customer-Id": "admin'--"}).status_code == 422


def test_chat_returns_reply_and_new_thread(client: TestClient) -> None:
    """A chat without a thread id starts one and returns the answer with the answering agent."""
    response = client.post("/v1/chat", json={"message": "what did I buy?"}, headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["reply"] == "You bought a dress." and data["agent"] == "order_management"
    assert data["thread_id"] and "X-Request-Id" in response.headers


def test_message_length_is_limited(client: TestClient) -> None:
    """Over-long and empty messages are rejected before reaching the graph."""
    assert client.post("/v1/chat", json={"message": "x" * 5000}, headers=HEADERS).status_code == 422
    assert client.post("/v1/chat", json={"message": ""}, headers=HEADERS).status_code == 422


def test_graph_failure_is_a_502_without_details(client: TestClient) -> None:
    """A crashing graph yields a generic 502, never the exception text."""
    response = client.post("/v1/chat", json={"message": "explode"}, headers=HEADERS)
    assert response.status_code == 502
    assert "graph failure" not in response.text


def test_stream_sends_events_in_order(client: TestClient) -> None:
    """The SSE stream carries thread, agent_selected, started, tokens, reply and done, in that order."""
    with client.stream("POST", "/v1/chat/stream", json={"message": "what did I buy?"}, headers=HEADERS) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        names = [line[len("event:"):].strip() for line in response.iter_lines() if line.startswith("event:")]
    assert names == ["thread", "agent_selected", "started", "token", "token", "reply", "done"]


def test_stream_error_event_when_graph_fails(client: TestClient) -> None:
    """A failing turn is reported as an error event, then done; the connection ends cleanly."""
    with client.stream("POST", "/v1/chat/stream", json={"message": "explode"}, headers=HEADERS) as response:
        lines = list(response.iter_lines())
    assert [line[len("event:"):].strip() for line in lines if line.startswith("event:")] == ["thread", "error", "done"]
    assert "graph failure" not in "\n".join(lines)


def test_history_of_own_thread(client: TestClient) -> None:
    """The owner can read a stored thread."""
    response = client.get("/v1/threads/t-old/messages", headers=HEADERS)
    assert response.status_code == 200
    assert [(m["role"], m["agent"], m["text"]) for m in response.json()["messages"]] == [
        ("user", None, "hi"),
        ("assistant", "order_management", "hello"),
    ]


def test_other_customers_thread_looks_missing(client: TestClient) -> None:
    """Another customer gets 404 for reading, and 404 for chatting on, someone else's thread."""
    other = {**HEADERS, "X-Customer-Id": "cust002"}
    assert client.get("/v1/threads/t-old/messages", headers=other).status_code == 404
    assert client.post("/v1/chat", json={"message": "hi", "thread_id": "t-old"}, headers=other).status_code == 404
    assert client.get("/v1/threads/does-not-exist/messages", headers=HEADERS).status_code == 404


def test_thread_without_recorded_owner_is_denied(client: TestClient) -> None:
    """A stored thread with no owner recorded (legacy data) matches nobody."""
    client.services.graph.threads["legacy"] = ([HumanMessage("hi")], {})
    assert client.get("/v1/threads/legacy/messages", headers=HEADERS).status_code == 404


def test_rate_limit(client: TestClient) -> None:
    """Over the per-customer limit the API answers 429 with Retry-After; other customers are unaffected."""
    client.services.limiter = RateLimiter(limit=2)
    for _ in range(2):
        assert client.post("/v1/chat", json={"message": "hi"}, headers=HEADERS).status_code == 200
    limited = client.post("/v1/chat", json={"message": "hi"}, headers=HEADERS)
    assert limited.status_code == 429 and int(limited.headers["Retry-After"]) >= 1
    assert client.post("/v1/chat", json={"message": "hi"}, headers={**HEADERS, "X-Customer-Id": "cust002"}).status_code == 200


def test_health_and_readiness(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """healthz is always ok; readyz is 503 and names the agents when none is reachable."""
    monkeypatch.setattr(health, "agent_url", lambda agent: "http://127.0.0.1:9")  # nothing listens here
    assert client.get("/healthz").json() == {"status": "ok"}
    ready = client.get("/readyz")
    assert ready.status_code == 503
    assert not ready.json()["ready"] and len(ready.json()["agents"]) == 4


def test_pushed_result_reaches_listeners() -> None:
    """The hub delivers an event published from another thread to a subscribed queue."""

    async def scenario() -> dict:
        hub = PushEventHub(asyncio.get_running_loop())
        queue = hub.subscribe("t1")
        await asyncio.to_thread(hub.publish, "t1", {"type": "update", "text": "arrived"})
        return await asyncio.wait_for(queue.get(), timeout=2)

    assert json.dumps(asyncio.run(scenario()), sort_keys=True) == json.dumps({"text": "arrived", "type": "update"}, sort_keys=True)
