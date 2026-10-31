"""Graph state and structured-output schemas.

Role in the system: shared data contracts between the supervisor and the graph.
"""

from typing import Literal

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


# Single source of truth for agent names: graph node names, message names and routing values.
AgentName = Literal["order_management", "customer_support", "shipment_tracking", "fashion", "out_of_scope"]

ORDER_MANAGEMENT: AgentName = "order_management"
CUSTOMER_SUPPORT: AgentName = "customer_support"
SHIPMENT_TRACKING: AgentName = "shipment_tracking"
FASHION: AgentName = "fashion"
OUT_OF_SCOPE: AgentName = "out_of_scope"

# Where the supervisor sends a message when the routing model returns no decision.
DEFAULT_AGENT: AgentName = CUSTOMER_SUPPORT


class AgentTask(BaseModel):
    """One unit of work for one agent.

    Attributes:
        agent: The agent that should handle it.
        query: Standalone sub-question for that agent. Set only when the message was split between
            several agents; ``None`` means the agent gets the customer's original message.
    """

    agent: AgentName = Field(description="The sub-agent that should handle this part, or out_of_scope if none fits.")
    query: str | None = Field(
        default=None,
        description="Only when the message is split between several agents: a self-contained question for this "
        "agent, keeping order ids and product names verbatim. Leave null when there is a single agent.",
    )


class SupervisorDecision(BaseModel):
    """Structured routing decision produced by the supervisor LLM."""

    tasks: list[AgentTask] = Field(
        min_length=1,
        description="One task for a single-topic message; one per topic only when the message clearly asks for different things.",
    )
    reason: str = Field(description="One short sentence explaining the choice.")


class AgentState(MessagesState):
    """Graph state: the message history plus the supervisor's plan.

    Attributes:
        tasks: Set by ``supervisor``; read by ``route_to_agent`` to fan out to one or more agents.
        query: Per-branch override of the customer message. Not written by any node: it only arrives in
            the payload ``route_to_agent`` sends to an agent node.
    """

    tasks: list[AgentTask]
    query: str
