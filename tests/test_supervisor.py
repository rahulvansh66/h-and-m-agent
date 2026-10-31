"""Tests for the supervisor: plan clean-up, fan-out, per-agent sub-queries and sticky routing.

Role in the system: checks ``hm_agent.supervisor.agent`` and the event mapping in ``hm_agent.turns``
without any LLM call; the routing model is replaced by a fake.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from hm_agent import turns
from hm_agent.core.state import AgentTask, SupervisorDecision
from hm_agent.supervisor import agent as supervisor


class FakeRouter:
    """Stand-in for the structured-output routing model; counts how often it is called."""

    def __init__(self, decision: SupervisorDecision | None) -> None:
        """Create a router that always returns ``decision``.

        Args:
            decision (SupervisorDecision | None): The decision every call returns.
        """
        self.decision, self.calls = decision, 0

    def invoke(self, _messages: list) -> SupervisorDecision | None:
        """Return the canned decision and count the call.

        Args:
            _messages (list): Prompt messages (unused).

        Returns:
            SupervisorDecision | None: The canned decision.
        """
        self.calls += 1
        return self.decision


@pytest.fixture
def router(monkeypatch: pytest.MonkeyPatch):
    """Patch the model loader and return a function that installs a canned decision.

    Args:
        monkeypatch (pytest.MonkeyPatch): Used to replace ``load_model``.

    Returns:
        Callable[[SupervisorDecision | None], FakeRouter]: Installs and returns the fake router.
    """

    def install(decision: SupervisorDecision | None) -> FakeRouter:
        fake = FakeRouter(decision)
        monkeypatch.setattr(supervisor, "load_model", lambda _name: type("M", (), {"with_structured_output": lambda self, _s: fake})())
        return fake

    return install


def _decision(*tasks: AgentTask) -> SupervisorDecision:
    """Build a decision from tasks.

    Args:
        *tasks (AgentTask): Tasks of the plan.

    Returns:
        SupervisorDecision: Decision with a fixed reason.
    """
    return SupervisorDecision(tasks=list(tasks), reason="test")


def test_normalise_drops_duplicates_and_mixed_out_of_scope() -> None:
    """A repeated agent and an out_of_scope next to real agents are removed."""
    tasks = [
        AgentTask(agent="fashion", query="outfit for the jacket"),
        AgentTask(agent="fashion", query="again"),
        AgentTask(agent="out_of_scope", query="chat"),
        AgentTask(agent="shipment_tracking", query="where is ORD-1001"),
    ]
    out = supervisor.normalise_tasks(tasks)
    assert [t.agent for t in out] == ["fashion", "shipment_tracking"]


def test_normalise_caps_at_max_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    """The plan never fans out to more agents than the configured maximum."""
    monkeypatch.setattr(supervisor, "SUPERVISOR_MAX_AGENTS", 2)
    tasks = [AgentTask(agent=a, query="q") for a in ("fashion", "order_management", "customer_support")]
    assert [t.agent for t in supervisor.normalise_tasks(tasks)] == ["fashion", "order_management"]


def test_normalise_single_task_loses_its_query() -> None:
    """With one agent left the customer's own message is used, so the sub-query is cleared."""
    out = supervisor.normalise_tasks([AgentTask(agent="fashion", query="paraphrase")])
    assert out == [AgentTask(agent="fashion", query=None)]


def test_normalise_split_without_queries_falls_back_to_one_agent() -> None:
    """Several agents without sub-queries would each answer everything, so only the first is kept."""
    out = supervisor.normalise_tasks([AgentTask(agent="fashion"), AgentTask(agent="order_management")])
    assert out == [AgentTask(agent="fashion", query=None)]


def test_route_sends_one_branch_per_task_with_its_query() -> None:
    """Each task becomes a Send to its agent carrying its sub-query (None for a single task)."""
    messages = [HumanMessage("hi")]
    split = supervisor.route_to_agent(
        {"messages": messages, "tasks": [AgentTask(agent="fashion", query="q1"), AgentTask(agent="order_management", query="q2")]}
    )
    assert [(s.node, s.arg["query"]) for s in split] == [("fashion", "q1"), ("order_management", "q2")]
    single = supervisor.route_to_agent({"messages": messages, "tasks": [AgentTask(agent="fashion")]})
    assert [(s.node, s.arg["query"]) for s in single] == [("fashion", None)]


def test_supervisor_node_uses_model_plan(router) -> None:
    """A fresh message is planned by the model and the plan is normalised."""
    fake = router(_decision(AgentTask(agent="order_management", query="x")))
    out = supervisor.supervisor_node({"messages": [HumanMessage("What did I buy last?")]})
    assert out["tasks"] == [AgentTask(agent="order_management")]
    assert fake.calls == 1


def test_supervisor_node_no_decision_uses_default_agent(router) -> None:
    """If the model returns nothing the message goes to customer support."""
    router(None)
    out = supervisor.supervisor_node({"messages": [HumanMessage("hello")]})
    assert out["tasks"] == [AgentTask(agent="customer_support")]


def test_sticky_short_follow_up_skips_the_model(router) -> None:
    """A short follow-up stays with the previous agent and makes no LLM call."""
    fake = router(_decision(AgentTask(agent="fashion")))
    messages = [HumanMessage("my jacket is damaged"), AIMessage("Sorry...", name="customer_support"), HumanMessage("how long does it take?")]
    out = supervisor.supervisor_node({"messages": messages})
    assert out["tasks"] == [AgentTask(agent="customer_support")]
    assert fake.calls == 0


@pytest.mark.parametrize(
    "follow_up, previous_name",
    [
        ("how long does that take if I also want to know what goes with the jacket?", "customer_support"),  # too long
        ("what goes with it?", "order_management"),  # no follow-up cue
        ("how long does it take?", "out_of_scope"),  # previous turn was a decline
        ("how long does it take?", None),  # previous message has no agent label
    ],
)
def test_sticky_falls_through_to_model(router, follow_up: str, previous_name: str | None) -> None:
    """Long messages, messages without a cue and declined or unlabeled previous turns go to the model."""
    fake = router(_decision(AgentTask(agent="fashion")))
    messages = [HumanMessage("hi"), AIMessage("answer", name=previous_name), HumanMessage(follow_up)]
    supervisor.supervisor_node({"messages": messages})
    assert fake.calls == 1


def test_sticky_first_turn_and_kill_switch(router, monkeypatch: pytest.MonkeyPatch) -> None:
    """The first turn has nothing to stay with, and a limit of 0 disables the shortcut."""
    fake = router(_decision(AgentTask(agent="fashion")))
    supervisor.supervisor_node({"messages": [HumanMessage("how long does it take?")]})
    assert fake.calls == 1
    monkeypatch.setattr(supervisor, "SUPERVISOR_STICKY_MAX_WORDS", 0)
    messages = [HumanMessage("hi"), AIMessage("answer", name="fashion"), HumanMessage("how long does it take?")]
    supervisor.supervisor_node({"messages": messages})
    assert fake.calls == 2


def test_turn_emits_one_agent_selected_per_task() -> None:
    """The supervisor update becomes one agent_selected event per agent in the plan."""

    class Graph:
        def stream(self, *_args, **_kwargs):
            yield "updates", {"supervisor": {"tasks": [AgentTask(agent="fashion", query="a"), AgentTask(agent="order_management", query="b")]}}
            yield "updates", {"fashion": {"messages": [AIMessage("f", name="fashion")]}}
            yield "updates", {"order_management": {"messages": [AIMessage("o", name="order_management")]}}

    events = list(turns.stream_turn(Graph(), "q", "cust001", "t1"))
    assert [(e["type"], e["agent"]) for e in events] == [
        ("agent_selected", "fashion"),
        ("agent_selected", "order_management"),
        ("reply", "fashion"),
        ("reply", "order_management"),
    ]
