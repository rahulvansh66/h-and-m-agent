"""Tests for the synthesis node and the split-turn graph flow (fan-out, join, tail, composer).

Role in the system: checks ``hm_agent.supervisor.synthesis`` (streams what is left of a split turn, saves
exactly what was shown, optional streaming composer) and, in a real compiled LangGraph with fake agent
nodes, that two parallel branches meet in ``synthesize`` exactly once while a single-agent turn skips it.
No LLM or agent service is called.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable
from langgraph.checkpoint.memory import InMemorySaver

from hm_agent import graph as graph_module
from hm_agent.core.state import AgentResult, AgentTask
from hm_agent.supervisor import split_stream, synthesis
from hm_agent.supervisor.split_stream import LEAD_INS

SOURCES = "\n\nSources:\n[1] Returns policy"


class FakeModel(Runnable):
    """Stand-in chat model that streams canned chunks (or fails) and counts calls."""

    def __init__(self, chunks: list[str] | Exception, fail_after: int | None = None) -> None:
        """Create a model.

        Args:
            chunks (list[str] | Exception): Chunks to stream, or an exception raised on the first call.
            fail_after (int | None): If set, raise after this many chunks have been yielded.
        """
        self.chunks, self.fail_after, self.calls = chunks, fail_after, 0

    def invoke(self, _input, config=None, **_kwargs):
        """Not used: the synthesis node only streams."""
        raise NotImplementedError

    def stream(self, _input, config=None, **_kwargs):
        """Yield the canned chunks as AI messages, or raise.

        Args:
            _input: Rendered prompt (unused).
            config: Run config (unused).

        Yields:
            AIMessage: One chunk.
        """
        self.calls += 1
        if isinstance(self.chunks, Exception):
            raise self.chunks
        for i, chunk in enumerate(self.chunks):
            if self.fail_after is not None and i == self.fail_after:
                raise RuntimeError("groq dropped the stream")
            yield AIMessage(chunk)


@pytest.fixture(autouse=True)
def clean_registry():
    """Keep the turn registry empty around every test."""
    split_stream._TURNS.clear()
    yield
    split_stream._TURNS.clear()


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch):
    """Install a fake synthesis model and return a function that sets its output.

    Args:
        monkeypatch (pytest.MonkeyPatch): Used to replace ``load_model``.

    Returns:
        Callable[..., FakeModel]: Installs and returns the fake model.
    """

    def install(chunks: list[str] | Exception, fail_after: int | None = None) -> FakeModel:
        fake = FakeModel(chunks, fail_after)
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


def _state(*results: AgentResult, **extra) -> dict:
    """Build a synthesis input state with the plan in the order given.

    Args:
        *results (AgentResult): Branch results, in arrival order.
        **extra: Extra state keys such as ``combine`` or ``turn_id``.

    Returns:
        dict: State with messages, tasks and results.
    """
    return {
        "messages": [HumanMessage("Where is ORD-1 and what goes with my jacket?")],
        "tasks": [AgentTask(agent=r.agent, query="q") for r in results],
        "results": list(results),
        **extra,
    }


def _text(events: list[dict]) -> str:
    """Join the token texts of captured events.

    Args:
        events (list[dict]): Captured stream events.

    Returns:
        str: What the customer saw.
    """
    return "".join(e["text"] for e in events if e["type"] == "token")


def test_no_results_is_a_noop(model, written) -> None:
    """A single-agent turn has no results, so nothing is streamed or saved."""
    fake = model(["unused"])
    assert synthesis.synthesize_node({"messages": [HumanMessage("hi")]}) == {}
    assert fake.calls == 0 and written == [] and split_stream.active_turns() == 0


def test_all_failed_apologises_once_without_llm(model, written) -> None:
    """When every branch failed there is one apology and no model call."""
    fake = model(["unused"])
    out = synthesis.synthesize_node(_state(_result("fashion", "failed"), _result("shipment_tracking", "failed")))
    assert out["messages"][0].text == synthesis.ALL_FAILED_REPLY == _text(written)
    assert fake.calls == 0 and out["results"] == []


def test_one_complete_shows_its_section_and_a_note_for_the_rest(model, written) -> None:
    """One good answer is shown under its lead-in; a failed and a pending branch each get a plain note."""
    fake = model(["unused"])
    out = synthesis.synthesize_node(
        _state(_result("fashion", text="Wear it with jeans."), _result("shipment_tracking", "pending", "STAND-IN ACK"))
    )
    text = out["messages"][0].text
    assert text.startswith(f"{LEAD_INS['fashion']}\nWear it with jeans.") and "still being checked" in text
    assert "STAND-IN ACK" not in text and text == _text(written)
    failed = synthesis.synthesize_node(_state(_result("fashion", text="Jeans."), _result("order_management", "failed")))
    assert "couldn't get an answer about orders and purchases" in failed["messages"][0].text
    assert fake.calls == 0


def test_two_complete_need_no_model_and_follow_plan_order(model, written) -> None:
    """By default two good answers are shown as two sections in plan order; the merge model is never called."""
    fake = model(["unused"])
    state = _state(_result("fashion", text="Jeans."), _result("order_management", text="You bought a jacket."))
    state["results"].reverse()  # arrival order differs from plan order
    out = synthesis.synthesize_node(state)
    message = out["messages"][0]
    assert message.text == f"{LEAD_INS['fashion']}\nJeans.\n\n{LEAD_INS['order_management']}\nYou bought a jacket."
    assert message.text == _text(written) and message.name == "multi_agent"
    assert message.additional_kwargs["agents"] == ["fashion", "order_management"]
    assert fake.calls == 0


def test_sections_already_streamed_are_not_repeated(model, written) -> None:
    """What the coordinator showed live is kept as is; only the missing part is added; history equals the stream."""
    model(["unused"])
    live = split_stream.get_split_stream("turn-1")
    live.piece("fashion", "Jeans.", written.append)
    live.finish("fashion", True, written.append)
    out = synthesis.synthesize_node(
        _state(_result("fashion", text="Jeans."), _result("shipment_tracking", "failed"), turn_id="turn-1")
    )
    assert out["messages"][0].text == _text(written)
    assert _text(written).count("Jeans.") == 1 and "delivery tracking" in _text(written)
    assert split_stream.active_turns() == 0


def test_partial_text_of_a_branch_that_died_is_kept_with_a_note(model, written) -> None:
    """A branch that failed after streaming leaves its text in the history, followed by the failure note."""
    model(["unused"])
    live = split_stream.get_split_stream("turn-2")
    live.piece("fashion", "Wear it with", written.append)
    live.finish("fashion", False, written.append)
    out = synthesis.synthesize_node(
        _state(_result("fashion", "failed", "sorry"), _result("order_management", text="You bought a jacket."), turn_id="turn-2")
    )
    text = out["messages"][0].text
    assert "Wear it with" in text and "You bought a jacket." in text and "outfit and product suggestions" in text
    assert synthesis.ALL_FAILED_REPLY not in text


def test_composer_streams_tokens_and_appends_sources_by_code(model, written) -> None:
    """A flagged turn streams the model's tokens live and appends the Sources list itself."""
    fake = model(["It arrives ", "Friday, so you can return it [1]."])
    out = synthesis.synthesize_node(
        _state(
            _result("shipment_tracking", text="Arrives Friday."),
            _result("customer_support", text="Return within 30 days [1]." + SOURCES),
            combine=True,
        )
    )
    token_events = [e for e in written if e["type"] == "token"]
    assert [e["text"] for e in token_events][:2] == ["It arrives ", "Friday, so you can return it [1]."]
    assert out["messages"][0].text == _text(written)
    assert out["messages"][0].text.endswith("Sources:\n[1] Returns policy") and out["messages"][0].text.count("Sources:") == 1
    assert fake.calls == 1


@pytest.mark.parametrize("failure", [RuntimeError("groq down"), []])
def test_composer_failing_before_first_token_falls_back_to_sections(model, written, failure) -> None:
    """An error or an empty stream before any token shows the plain sections instead."""
    model(failure)
    out = synthesis.synthesize_node(
        _state(_result("shipment_tracking", text="Arrives Friday."), _result("customer_support", text="Return it."), combine=True)
    )
    text = out["messages"][0].text
    assert "Arrives Friday." in text and "Return it." in text and LEAD_INS["customer_support"] in text


def test_composer_breaking_mid_stream_keeps_what_was_shown(model, written) -> None:
    """If the model dies after some tokens, the customer's text stays and nothing is added or crashes."""
    model(["It arrives ", "Friday"], fail_after=1)
    out = synthesis.synthesize_node(
        _state(_result("shipment_tracking", text="Arrives Friday."), _result("customer_support", text="Return it."), combine=True)
    )
    assert out["messages"][0].text == "It arrives " == _text(written)


def test_composer_skipped_when_two_answers_have_sources(model, written) -> None:
    """Two Sources lists would clash on [n] numbering, so the plain sections are shown instead."""
    fake = model(["unused"])
    out = synthesis.synthesize_node(
        _state(_result("customer_support", text="A [1]." + SOURCES), _result("fashion", text="B [1]." + SOURCES), combine=True)
    )
    assert fake.calls == 0 and out["messages"][0].text.count("Sources:") == 2


def test_combine_with_one_complete_answer_does_not_call_the_model(model, written) -> None:
    """There is nothing to combine when only one branch answered."""
    fake = model(["unused"])
    out = synthesis.synthesize_node(
        _state(_result("fashion", text="Jeans."), _result("order_management", "failed"), combine=True)
    )
    assert fake.calls == 0 and "Jeans." in out["messages"][0].text


def test_split_sources_and_citation_check() -> None:
    """The Sources list is separated from the body, and the citation check notices a dropped marker."""
    assert synthesis._split_sources("Body [1]." + SOURCES) == ("Body [1].", "Sources:\n[1] Returns policy")
    assert synthesis._split_sources("No list.") == ("No list.", "")
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
    monkeypatch.setattr(graph_module, "supervisor_node", lambda state: {"tasks": tasks, "results": [], "turn_id": "g", "combine": False})
    monkeypatch.setattr(graph_module, "make_remote_agent_node", _fake_remote_node_factory(calls))
    return graph_module.build_graph(InMemorySaver()), calls


def test_split_turn_runs_both_branches_then_synthesize_once(monkeypatch: pytest.MonkeyPatch, model) -> None:
    """Two parallel branches produce one saved message with both sections and leave no results or turn behind."""
    fake = model(["unused"])
    graph, calls = _build(monkeypatch, [AgentTask(agent="fashion", query="a"), AgentTask(agent="order_management", query="b")])
    config = {"configurable": {"thread_id": "t1"}}
    out = graph.invoke({"messages": [HumanMessage("two things")]}, config)
    assert sorted(calls) == ["fashion", "order_management"]
    assert [m.name for m in out["messages"]] == [None, "multi_agent"]
    text = out["messages"][-1].text
    assert "fashion says ok" in text and "order_management says ok" in text
    assert out["results"] == [] and fake.calls == 0 and split_stream.active_turns() == 0


def test_single_turn_skips_synthesize_and_the_coordinator(monkeypatch: pytest.MonkeyPatch, model) -> None:
    """A one-agent turn writes the agent's own message; no results, no coordinator, no model call."""
    fake = model(["unused"])
    graph, calls = _build(monkeypatch, [AgentTask(agent="fashion")])
    out = graph.invoke({"messages": [HumanMessage("one thing")]}, {"configurable": {"thread_id": "t2"}})
    assert calls == ["fashion"]
    assert [m.name for m in out["messages"]] == [None, "fashion"]
    assert out["results"] == [] and fake.calls == 0 and split_stream.active_turns() == 0


def test_pushed_message_does_not_trigger_synthesize(monkeypatch: pytest.MonkeyPatch, model) -> None:
    """A result added later with update_state(as_node=agent) leaves no merge step pending."""
    fake = model(["unused"])
    graph, _ = _build(monkeypatch, [AgentTask(agent="shipment_tracking")])
    config = {"configurable": {"thread_id": "t3"}}
    graph.invoke({"messages": [HumanMessage("where is it")]}, config)
    graph.update_state(config, {"messages": [AIMessage("late result", name="shipment_tracking")]}, as_node="shipment_tracking")
    assert graph.get_state(config).next == ()
    assert fake.calls == 0
