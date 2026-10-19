"""Graph state and structured-output schemas.

Role in the system: shared data contracts between the supervisor and the graph.
"""

from typing import Literal

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


# Single source of truth for agent names: graph node names, message names and routing values.
AgentName = Literal["order_management", "customer_support", "out_of_scope"]

ORDER_MANAGEMENT: AgentName = "order_management"
CUSTOMER_SUPPORT: AgentName = "customer_support"
OUT_OF_SCOPE: AgentName = "out_of_scope"

# Where the supervisor sends a message when the routing model returns no decision.
DEFAULT_AGENT: AgentName = CUSTOMER_SUPPORT


class SupervisorDecision(BaseModel):
    """Structured routing decision produced by the supervisor LLM."""

    agent: AgentName = Field(
        description="The sub-agent that should handle the message, or out_of_scope if none fits."
    )
    reason: str = Field(description="One short sentence explaining the choice.")


class AgentState(MessagesState):
    """Graph state: the message history plus the supervisor's chosen agent.

    Attributes:
        next_agent: Set by ``supervisor``; read by ``route_to_agent``.
    """

    next_agent: AgentName
