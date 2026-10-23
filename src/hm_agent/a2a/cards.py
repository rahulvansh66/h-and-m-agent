"""Agent Cards: the public self-description each A2A agent serves at its well-known URL.

Role in the system: the supervisor (client) discovers an agent's endpoint, protocol version and
skills from its card before sending a message. Cards are built here so the services and any
tooling describe the agents identically.
"""

from a2a.types.a2a_pb2 import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from a2a.utils.constants import PROTOCOL_VERSION_CURRENT, TransportProtocol

from hm_agent.core.settings import agent_url
from hm_agent.core.state import CUSTOMER_SUPPORT, ORDER_MANAGEMENT, SHIPMENT_TRACKING

CARD_VERSION = "0.4.0"  # bump when a card's skills or behaviour change

# A skill is (skill id, skill name, skill description, example questions).
_Skill = tuple[str, str, str, list[str]]

# name -> (agent description, skills)
_AGENT_SPECS: dict[str, tuple[str, list[_Skill]]] = {
    ORDER_MANAGEMENT: (
        "Answers questions about the customer's own orders using read-only order data.",
        [
            (
                "order-history",
                "Order history lookup",
                "Last purchase and total purchase count for the identified customer.",
                ["What was the last item I bought?", "How many purchases have I made?"],
            ),
            (
                "order-details",
                "Order contents and details",
                "Date, sales channel and items for a specific order, or a list of recent orders.",
                ["What was in order ORD-1007?", "Show my recent orders", "Did I buy that online or in store?"],
            ),
        ],
    ),
    CUSTOMER_SUPPORT: (
        "Helps with damaged or defective items using the H&M help articles, with citations.",
        [
            (
                "damaged-item-help",
                "Damaged and defective item help",
                "Returns, replacements, refunds and next steps for faulty items.",
                ["My jacket arrived with a broken zipper, what can I do?"],
            ),
        ],
    ),
    SHIPMENT_TRACKING: (
        "Checks the carrier or warehouse for the shipping status of an order. The lookup is slow: "
        "it streams progress while it works, and callers can register a push notification webhook "
        "as a fallback if their connection drops.",
        [
            (
                "shipment-status",
                "Shipment status lookup",
                "Carrier, tracking number, last scan and delivery outlook for an order id.",
                ["Where is my order ORD-1007?", "Has my parcel shipped yet?"],
            ),
        ],
    ),
}

# Agents whose tasks outlive a normal request, so their card advertises push notifications.
PUSH_AGENTS = frozenset({SHIPMENT_TRACKING})


def build_agent_card(agent: str) -> AgentCard:
    """Build the Agent Card for one agent.

    Args:
        agent (str): Agent name, one of ``order_management``, ``customer_support``, ``shipment_tracking``.

    Returns:
        AgentCard: Card whose JSON-RPC interface points at the agent's configured URL.

    Raises:
        KeyError: If ``agent`` is not a known agent name.
    """
    description, skills = _AGENT_SPECS[agent]
    return AgentCard(
        name=agent,
        description=description,
        version=CARD_VERSION,
        supported_interfaces=[
            AgentInterface(
                url=agent_url(agent),
                protocol_binding=TransportProtocol.JSONRPC,
                protocol_version=PROTOCOL_VERSION_CURRENT,
            )
        ],
        # Every agent streams progress and answer chunks. Push notifications only for agents with
        # slow tasks: the webhook is the durable fallback if the caller's stream connection drops.
        capabilities=AgentCapabilities(streaming=True, push_notifications=agent in PUSH_AGENTS),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(id=skill_id, name=skill_name, description=skill_desc, tags=[agent], examples=examples)
            for skill_id, skill_name, skill_desc, examples in skills
        ],
    )
