"""Supervisor agent: classifies the user message and picks a sub-agent.

Role in the system: first node of the graph; its decision drives the conditional edge. It is
the only place that decides scope, so sub-agents can assume the messages they get are theirs.
Messages no agent handles are routed to the local ``out_of_scope`` node, which returns the fixed
reply (``OUT_OF_SCOPE_REPLY``) defined here. It is not an A2A agent: it has no LLM, tools or data.
"""

import logging

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from hm_agent.core.llm import load_model
from hm_agent.core.settings import SUPERVISOR_HISTORY
from hm_agent.core.state import (
    DEFAULT_AGENT,
    OUT_OF_SCOPE,
    AgentName,
    AgentState,
    SupervisorDecision,
)

logger = logging.getLogger(__name__)

OUT_OF_SCOPE_REPLY = (
    "I can help with questions about your orders and purchases, where your parcel is, and "
    "damaged or defective items. For anything else, please contact H&M customer service."
)

SUPERVISOR_PROMPT = """\
# ROLE
You are the supervisor of the H&M customer support team. You do not answer customers;
you only decide which specialist agent should handle the latest customer message.

# TASK
You are given the recent conversation. Choose exactly one agent for the latest customer message:
- order_management: questions about the customer's own orders or purchases
  (what they bought, last purchase, how many purchases, order history).
- shipment_tracking: where an order or parcel is right now, shipping or delivery status,
  carrier and tracking questions.
- customer_support: anything about damaged, defective, broken or faulty items, and what to do
  about them (returns, replacements, refunds, contacting a seller, proof, timelines).
- out_of_scope: anything else (general chat, product advice, unrelated topics).

# RULES
- Use the conversation to understand the latest message. A short follow-up ("what about the
  refund?", "and how long does it take?") belongs to the agent that handled the previous turn,
  shown as "Assistant (<agent>)", unless the customer clearly changed topic.
- If the message mentions a damaged or defective item, choose customer_support.
- If the message is about order or purchase information, choose order_management.
- If the message is about where an order is or when it arrives, choose shipment_tracking.
- If it fits no agent, choose out_of_scope. If it is only unclear, prefer customer_support.
- Never invent another agent name.

# OUTPUT
Return the structured fields only: `agent` (order_management | customer_support | shipment_tracking | out_of_scope)
and `reason` (one short sentence).
"""


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
            label = f"Assistant ({m.name})" if m.name else "Assistant"
            lines.append(f"{label}: {m.text}")
    return "\n".join(lines)


def supervisor_node(state: AgentState) -> dict:
    """Classify the latest user message (in context) and choose the sub-agent.

    If the model returns no decision, the message goes to ``DEFAULT_AGENT`` (customer support):
    that agent's relevance gate handles off-topic text cheaply, whereas a wrong ``out_of_scope``
    would turn a customer away.

    Reads: ``messages``. Writes: ``next_agent``.

    Args:
        state: Current graph state.

    Returns:
        dict: ``{"next_agent": AgentName}``.
    """
    llm = load_model("small").with_structured_output(SupervisorDecision)
    decision: SupervisorDecision | None = llm.invoke(
        [SystemMessage(SUPERVISOR_PROMPT), HumanMessage(_render_transcript(state["messages"]))]
    )
    if decision is None:
        logger.warning("node=supervisor status=no_decision fallback=%s", DEFAULT_AGENT)
        return {"next_agent": DEFAULT_AGENT}
    logger.info("node=supervisor agent=%s reason=%s", decision.agent, decision.reason)
    return {"next_agent": decision.agent}


def route_to_agent(state: AgentState) -> AgentName:
    """Conditional edge: send the run to the agent the supervisor chose.

    Reads: ``next_agent``. Writes: nothing.
    Possible next nodes: ``order_management``, ``customer_support``, ``shipment_tracking``, ``out_of_scope``.

    Args:
        state: Current graph state.

    Returns:
        AgentName: Name of the next node.
    """
    return state["next_agent"]


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
