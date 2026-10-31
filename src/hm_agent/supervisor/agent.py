"""Supervisor agent: classifies the user message and picks one or more sub-agents.

Role in the system: first node of the graph; its plan drives the conditional edge, which fans out
to one agent (the usual case) or to several in parallel when the message mixes topics, giving each
its own sub-query. Short follow-ups skip the model and stay with the previous agent. It is
the only place that decides scope, so sub-agents can assume the messages they get are theirs.
Messages no agent handles are routed to the local ``out_of_scope`` node, which returns the fixed
reply (``OUT_OF_SCOPE_REPLY``) defined here. It is not an A2A agent: it has no LLM, tools or data.
"""

import logging

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import Send

from hm_agent.core.llm import load_model
from hm_agent.core.settings import SUPERVISOR_HISTORY, SUPERVISOR_MAX_AGENTS, SUPERVISOR_STICKY_MAX_WORDS
from hm_agent.core.state import (
    DEFAULT_AGENT,
    MULTI_AGENT,
    OUT_OF_SCOPE,
    SPECIALIST_AGENTS,
    AgentName,
    AgentState,
    AgentTask,
    SupervisorDecision,
)

logger = logging.getLogger(__name__)

OUT_OF_SCOPE_REPLY = (
    "I can help with questions about your orders and purchases, where your parcel is, "
    "damaged or defective items, and outfit or product suggestions. "
    "For anything else, please contact H&M customer service."
)

SUPERVISOR_PROMPT = """\
# ROLE
You are the supervisor of the H&M customer support team. You do not answer customers;
you only decide which specialist agent should handle the latest customer message.

# TASK
You are given the recent conversation. Choose the agent for the latest customer message:
- order_management: questions about the customer's own orders or purchases
  (what they bought, last purchase, how many purchases, order history).
- shipment_tracking: where an order or parcel is right now, shipping or delivery status,
  carrier and tracking questions.
- customer_support: anything about damaged, defective, broken or faulty items, and what to do
  about them (returns, replacements, refunds, contacting a seller, proof, timelines).
- fashion: product recommendations and styling advice (what goes with an item, an outfit for an
  occasion, something similar to a product).
- out_of_scope: anything else (general chat, unrelated topics).

# RULES
- Use the conversation to understand the latest message. A short follow-up ("what about the
  refund?", "and how long does it take?") belongs to the agent that handled the previous turn,
  shown as "Assistant (<agent>)", unless the customer clearly changed topic.
- If the message mentions a damaged or defective item, choose customer_support.
- If the message is about order or purchase information, choose order_management.
- If the message is about where an order is or when it arrives, choose shipment_tracking.
- Buying history is a fact (order_management); pairing, outfits and recommendations are advice (fashion).
  "What did I buy last?" is order_management; "what goes with what I bought?" is fashion.
- If it fits no agent, choose out_of_scope. If it is only unclear, prefer customer_support.
- Never invent another agent name.
- Use exactly one task unless the message clearly asks for different things that need different agents
  (e.g. "where is ORD-1001, and what goes with my jacket?"). Then return one task per topic, at most 2,
  and give each a `query`: a short, self-contained question holding only that agent's part, using the
  conversation to resolve "it" or "that". Keep order ids and product names exactly as written.
- With a single task leave `query` null; the agent then receives the customer's own words.
- Never combine out_of_scope with another agent.

# OUTPUT
Return the structured fields only: `tasks` (each with `agent`: order_management | customer_support |
shipment_tracking | fashion | out_of_scope, and `query`) and `reason` (one short sentence).
"""

# A short message starting with one of these is a follow-up on the topic of the previous answer.
FOLLOW_UP_CUES = ("what about", "how long", "how much", "how many", "why", "when", "and how long", "and why")


def _render_transcript(messages: list[AnyMessage]) -> str:
    """Render the most recent messages as a labelled transcript for the routing prompt.

    AI messages are labelled with the agent that wrote them (``Assistant (<agent>)``) so the
    supervisor can keep follow-ups with the agent that was already handling the topic.

    Args:
        messages: Conversation messages, oldest first.

    Returns:
        str: One ``Speaker: text`` line per message, limited to the last ``SUPERVISOR_HISTORY``.
    """
    lines = []
    for m in messages[-SUPERVISOR_HISTORY:]:
        if isinstance(m, HumanMessage):
            lines.append(f"Customer: {m.text}")
        else:
            if m.name == MULTI_AGENT:
                label = f"Assistant (multi: {', '.join(m.additional_kwargs.get('agents', []))})"
            else:
                label = f"Assistant ({m.name})" if m.name else "Assistant"
            lines.append(f"{label}: {m.text}")
    return "\n".join(lines)


def _sticky_agent(messages: list[AnyMessage]) -> AgentName | None:
    """Return the previous agent when the latest message is a short follow-up to its answer.

    Saves the routing LLM call for turns like "how long does it take?". Deliberately conservative:
    the message must be at most ``SUPERVISOR_STICKY_MAX_WORDS`` words, start with a follow-up cue,
    and the previous answer must come from a real agent. A short message that changes topic and
    starts with a cue would be misrouted; set ``supervisor.sticky_max_words`` to 0 to turn this off.

    Args:
        messages: Conversation messages, oldest first; the last one is the customer's new message.

    Returns:
        AgentName | None: The agent that wrote the previous answer, or None to ask the LLM.
    """
    if SUPERVISOR_STICKY_MAX_WORDS <= 0 or len(messages) < 2 or not isinstance(messages[-1], HumanMessage):
        return None
    previous = messages[-2]
    if not isinstance(previous, AIMessage) or previous.name not in SPECIALIST_AGENTS:
        return None  # no label, a decline, or a merged multi-agent answer: let the model decide
    text = messages[-1].text.strip().lower()
    if len(text.split()) > SUPERVISOR_STICKY_MAX_WORDS or not text.startswith(FOLLOW_UP_CUES):
        return None
    return previous.name


def normalise_tasks(tasks: list[AgentTask]) -> list[AgentTask]:
    """Clean up the model's plan so the graph can run it safely.

    Removes repeated agents, drops ``out_of_scope`` when real agents are present, caps the plan at
    ``SUPERVISOR_MAX_AGENTS``, and clears the sub-queries when only one task is left (that agent
    then gets the customer's own message).

    Args:
        tasks: Tasks returned by the routing model, in the order it gave them.

    Returns:
        list[AgentTask]: One to ``SUPERVISOR_MAX_AGENTS`` tasks, never empty if ``tasks`` was not.
    """
    seen: set[str] = set()
    unique = [t for t in tasks if not (t.agent in seen or seen.add(t.agent))]
    real = [t for t in unique if t.agent != OUT_OF_SCOPE]
    kept = (real or unique)[:SUPERVISOR_MAX_AGENTS]
    if len(kept) == 1:
        return [AgentTask(agent=kept[0].agent)]
    # Several agents need a sub-query each; without one the agent would answer the whole message.
    return [t for t in kept if t.query] or [AgentTask(agent=kept[0].agent)]


def supervisor_node(state: AgentState) -> dict:
    """Plan the turn: pick the agent(s) for the latest user message, in context.

    A short follow-up is kept with the previous agent without an LLM call (see ``_sticky_agent``).
    Otherwise the small model returns one task, or several with a sub-query each when the message
    mixes topics. If the model returns no decision, the message goes to ``DEFAULT_AGENT``
    (customer support): that agent's relevance gate handles off-topic text cheaply, whereas a wrong
    ``out_of_scope`` would turn a customer away.

    Reads: ``messages``. Writes: ``tasks`` and an empty ``results`` (clears leftovers of an earlier turn).

    Args:
        state: Current graph state.

    Returns:
        dict: ``{"tasks": list[AgentTask], "results": []}``.
    """
    sticky = _sticky_agent(state["messages"])
    if sticky:
        logger.info("node=supervisor status=sticky agent=%s", sticky)
        return {"tasks": [AgentTask(agent=sticky)], "results": []}
    llm = load_model("small").with_structured_output(SupervisorDecision)
    decision: SupervisorDecision | None = llm.invoke(
        [SystemMessage(SUPERVISOR_PROMPT), HumanMessage(_render_transcript(state["messages"]))]
    )
    if decision is None:
        logger.warning("node=supervisor status=no_decision fallback=%s", DEFAULT_AGENT)
        return {"tasks": [AgentTask(agent=DEFAULT_AGENT)], "results": []}
    tasks = normalise_tasks(decision.tasks)
    logger.info("node=supervisor agents=%s reason=%s", ",".join(t.agent for t in tasks), decision.reason)
    return {"tasks": tasks, "results": []}


def route_to_agent(state: AgentState) -> list[Send]:
    """Conditional edge: send the run to every agent in the plan, in parallel.

    Each agent node receives the message history plus its own ``query`` (None for a single-agent
    turn, so it falls back to the customer's message).

    Reads: ``tasks``, ``messages``. Writes: nothing.
    Possible next nodes: ``order_management``, ``customer_support``, ``shipment_tracking``, ``fashion``, ``out_of_scope``.

    Args:
        state: Current graph state.

    Returns:
        list[Send]: One ``Send`` per task.
    """
    return [Send(t.agent, {"messages": state["messages"], "query": t.query}) for t in state["tasks"]]


def route_after_agent(state: AgentState) -> str:
    """Conditional edge after an agent node: merge on a split turn, otherwise finish.

    A single-agent turn has no ``results``, so it ends exactly as before (which also keeps pushed
    results, added to the thread as the agent's node, from triggering a merge).

    Reads: ``results``. Writes: nothing.

    Args:
        state: Current graph state.

    Returns:
        str: ``"synthesize"`` when branch results exist, else ``END``.
    """
    return "synthesize" if state.get("results") else END


def out_of_scope_node(state: AgentState, config: RunnableConfig) -> dict:
    """Reply with the fixed decline message when no agent can handle the request.

    Runs inside the supervisor graph (no A2A call, no LLM), so a decline is instant and cannot
    fail because a remote service is down.

    Reads: nothing. Writes: ``messages`` (one AI reply named ``out_of_scope``).

    Args:
        state (AgentState): Current graph state (unused).
        config (RunnableConfig): Run config; only ``thread_id`` is read, for logging.

    Returns:
        dict: ``{"messages": [AIMessage]}`` carrying ``OUT_OF_SCOPE_REPLY``.
    """
    thread_id = config.get("configurable", {}).get("thread_id", "")
    logger.info("node=%s status=declined thread_id=%s", OUT_OF_SCOPE, thread_id)
    return {"messages": [AIMessage(OUT_OF_SCOPE_REPLY, name=OUT_OF_SCOPE)]}
