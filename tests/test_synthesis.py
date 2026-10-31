"""Tests for the synthesis node and the split-turn graph flow (fan-out, join, merge).

Role in the system: checks ``hm_agent.supervisor.synthesis`` (cheapest-path merge, citation guard,
fallbacks) and, in a real compiled LangGraph with fake agent nodes, that two parallel branches meet
in ``synthesize`` exactly once while a single-agent turn skips it. No LLM or agent service is called.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from hm_agent import graph as graph_module
from hm_agent.core.state import AgentResult, AgentTask
from hm_agent.supervisor import synthesis


class FakeModel:
    """Stand-in chat model that records calls and returns a canned reply or raises."""

    def __init__(self, reply: str | Exception) -> None:
        """Create a model.

        Args:
            reply (str | Exception): Text to return, or an exception to raise on invoke.
        """
        self.reply, self.calls = reply, 0

    def invoke(self, _messages: list) -> AIMessage:
        """Return the canned reply (or raise it) and count the call.

        Args:
            _messages (list): Prompt messages (unused).

        Returns:
            AIMessage: The canned reply.
        """
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return AIMessage(self.reply)


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch):
    """Install a fake synthesis model and return a function that sets its reply.

    Args:
        monkeypatch (pytest.MonkeyPatch): Used to replace ``load_model``.

    Returns:
        Callable[[str | Exception], FakeModel]: Installs and returns the fake model.
    """

    def install(reply: str | Exception) -> FakeModel:
        fake = FakeModel(reply)
        monkeypatch.setattr(synthesis, "load_model", lambda _size: fake)
        return fake

    return install


@pytest.fixture
def written(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Capture the custom stream events the node writes.

    Args:
        monkeypatch (pytest.MonkeyPatch): Used to replace the stream writer.

    Returns:
        list[dict]: Events written so far.
    """
    events: list[dict] = []
    monkeypatch.setattr(synthesis, "get_stream_writer", lambda: events.append)
    return events


def _result(agent: str, status: str = "complete", text: str = "answer") -> AgentResult:
    """Build a branch result.

    Args:
        agent (str): Agent name.
        status (str): complete, pending or failed.
        text (str): Answer text.

    Returns:
        AgentResult: The result with a fixed sub-query.
    """
    return AgentResult(agent=agent, query="q", status=status, text=text)


def _state(*results: AgentResult) -> dict:
    """Build a synthesis input state with the plan in the order given.

    Args:
        *results (AgentResult): Branch results, in arrival order.

    Returns:
        dict: State with messages, tasks and results.
    """
    return {
        "messages": [HumanMessage("Where is ORD-1 and what goes with my jacket?")],
        "tasks": [AgentTask(agent=r.agent, query="q") for r in results],
        "results": list(results),
    }


def test_no_results_is_a_noop(model, written) -> None:
    """A single-agent turn has no results, so nothing is merged or streamed."""
    fake = model("unused")
    assert synthesis.synthesize_node({"messages": [HumanMessage("hi")]}) == {}
    assert fake.calls == 0 and written == []


def test_all_failed_apologises_without_llm(model, written) -> None:
    """When every branch failed there is one apology and no model call."""
    fake = model("unused")
    out = synthesis.synthesize_node(_state(_result("fashion", "failed"), _result("shipment_tracking", "failed")))
    assert out["messages"][0].text == synthesis.ALL_FAILED_REPLY
    assert fake.calls == 0 and out["results"] == []


def test_one_complete_adds_note_for_the_rest_without_llm(model, written) -> None:
    """One good answer is kept as is; a failed and a pending branch each get a plain note."""
    fake = model("unused")
    out = synthesis.synthesize_node(
        _state(_result("fashion", text="Wear it with jeans."), _result("shipment_tracking", "pending", "STAND-IN ACK"))
    )
    text = out["messages"][0].text
    assert text.startswith("Wear it with jeans.") and "still being checked" in text and "STAND-IN ACK" not in text
    failed = synthesis.synthesize_node(_state(_result("fashion", text="Jeans."), _result("order_management", "failed")))
    assert "couldn't get an answer about orders and purchases" in failed["messages"][0].text
    assert fake.calls == 0


def test_two_complete_are_merged_by_the_model_and_streamed_once(model, written) -> None:
    """Two good answers cost one model call; the merged reply is named multi_agent and streamed as one token."""
    fake = model("It is on its way. Wear it with jeans.")
    out = synthesis.synthesize_node(_state(_result("shipment_tracking", text="On its way."), _result("fashion", text="Jeans.")))
    message = out["messages"][0]
    assert message.text == "It is on its way. Wear it with jeans."
    assert message.name == "multi_agent" and message.additional_kwargs["agents"] == ["shipment_tracking", "fashion"]
    assert fake.calls == 1
    assert written == [{"type": "token", "agent": "multi_agent", "text": message.text}]


def test_results_are_merged_in_plan_order_not_arrival_order(model, written) -> None:
    """The plan order decides the order of the answers, whichever branch finished first."""
    model("x")
    state = _state(_result("fashion"), _result("order_management"))
    state["results"].reverse()
    out = synthesis.synthesize_node(state)
    assert out["messages"][0].additional_kwargs["agents"] == ["fashion", "order_management"]


@pytest.mark.parametrize("bad", ["", "Dropped the citation.", RuntimeError("groq down")])
def test_bad_model_output_falls_back_to_plain_join(model, written, bad: str | Exception) -> None:
    """An empty reply, a lost citation marker or a model error all fall back to joining the answers."""
    model(bad)
    out = synthesis.synthesize_node(
        _state(_result("customer_support", text="Return it within 30 days [1].\n\nSources:\n[1] Returns"), _result("fashion", text="Jeans."))
    )
    text = out["messages"][0].text
    assert "Return it within 30 days [1]." in text and "Jeans." in text


def test_citations_kept_checks_markers() -> None:
    """The guard passes when every marker survives and fails when one is dropped."""
    assert synthesis._citations_kept(["A [1] B [2]"], "keep [2] and [1]")
    assert not synthesis._citations_kept(["A [1] B [2]"], "only [1]")
    assert synthesis._citations_kept(["no markers"], "anything")


def _fake_remote_node_factory(calls: list[str]):
    """Create a stand-in for ``make_remote_agent_node`` that mimics a branch or a single-agent node.

    Args:
        calls (list[str]): Receives the name of each node run.

    Returns:
        Callable[[str], Callable]: Factory taking an agent name and returning the node.
    """

    def make(agent: str):
        def node(state: dict) -> dict:
            calls.append(agent)
            sub_query = state.get("query")
            if sub_query:
                return {"results": [AgentResult(agent=agent, query=sub_query, status="complete", text=f"{agent} says ok")]}
            return {"messages": [AIMessage(f"{agent} says ok", name=agent)]}

        return node

    return make


def _build(monkeypatch: pytest.MonkeyPatch, tasks: list[AgentTask]):
    """Compile the real graph with a fixed plan and fake agent nodes.

    Args:
        monkeypatch (pytest.MonkeyPatch): Used to replace the supervisor and remote nodes.
        tasks (list[AgentTask]): The plan the fake supervisor returns.

    Returns:
        tuple: ``(graph, calls)`` where ``calls`` records which agent nodes ran.
    """
    calls: list[str] = []
    monkeypatch.setattr(graph_module, "supervisor_node", lambda state: {"tasks": tasks, "results": []})
    monkeypatch.setattr(graph_module, "make_remote_agent_node", _fake_remote_node_factory(calls))
    return graph_module.build_graph(InMemorySaver()), calls


def test_split_turn_runs_both_branches_then_synthesize_once(monkeypatch: pytest.MonkeyPatch, model) -> None:
    """Two parallel branches produce one merged AI message and leave no results behind."""
    fake = model("Merged answer.")
    graph, calls = _build(monkeypatch, [AgentTask(agent="fashion", query="a"), AgentTask(agent="order_management", query="b")])
    config = {"configurable": {"thread_id": "t1"}}
    out = graph.invoke({"messages": [HumanMessage("two things")]}, config)
    assert sorted(calls) == ["fashion", "order_management"]
    assert [m.name for m in out["messages"]] == [None, "multi_agent"]
    assert out["messages"][-1].text == "Merged answer."
    assert out["results"] == [] and fake.calls == 1


def test_single_turn_skips_synthesize(monkeypatch: pytest.MonkeyPatch, model) -> None:
    """A one-agent turn writes the agent's own message and never calls the merge model."""
    fake = model("unused")
    graph, calls = _build(monkeypatch, [AgentTask(agent="fashion")])
    out = graph.invoke({"messages": [HumanMessage("one thing")]}, {"configurable": {"thread_id": "t2"}})
    assert calls == ["fashion"]
    assert [m.name for m in out["messages"]] == [None, "fashion"]
    assert fake.calls == 0


def test_pushed_message_does_not_trigger_synthesize(monkeypatch: pytest.MonkeyPatch, model) -> None:
    """A result added later with update_state(as_node=agent) leaves no merge step pending."""
    fake = model("unused")
    graph, _ = _build(monkeypatch, [AgentTask(agent="shipment_tracking")])
    config = {"configurable": {"thread_id": "t3"}}
    graph.invoke({"messages": [HumanMessage("where is it")]}, config)
    graph.update_state(config, {"messages": [AIMessage("late result", name="shipment_tracking")]}, as_node="shipment_tracking")
    assert graph.get_state(config).next == ()
    assert fake.calls == 0
