"""Graph assembly for the supervisor topology.

    START -> supervisor -> order_management  (A2A) -> END
                        -> customer_support  (A2A) -> END
                        -> shipment_tracking (A2A, push notifications) -> END
                        -> out_of_scope      (local) -> END

Role in the system: wires the supervisor and the agents into one compiled graph. The two real
agents run as their own A2A services (``hm_agent.<agent>.a2a_service``); here each is only a node that
calls it (``shipment_tracking`` returns an acknowledgement and its result is added to the thread
later, by the push receiver). ``out_of_scope`` is a local node returning the supervisor's fixed decline reply.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from hm_agent.a2a.remote_node import make_remote_agent_node
from hm_agent.supervisor.agent import out_of_scope_node, route_to_agent, supervisor_node
from hm_agent.core.state import CUSTOMER_SUPPORT, ORDER_MANAGEMENT, OUT_OF_SCOPE, SHIPMENT_TRACKING, AgentState

REMOTE_AGENTS = (ORDER_MANAGEMENT, CUSTOMER_SUPPORT, SHIPMENT_TRACKING)  # agents served as remote A2A services


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    """Assemble and compile the supervisor graph.

    The agent services must be running for a turn routed to them to get an answer (see
    ``scripts/a2a/run_agents.py``); out-of-scope replies are produced locally.
    Conversation state lives only here, in the supervisor's checkpointer.

    Args:
        checkpointer: Optional checkpointer that persists state per ``thread_id``.

    Returns:
        CompiledStateGraph: Graph with nodes ``supervisor``, ``order_management``,
        ``customer_support``, ``shipment_tracking`` and ``out_of_scope``.
    """
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node(OUT_OF_SCOPE, out_of_scope_node)
    graph.add_edge(OUT_OF_SCOPE, END)
    for agent in REMOTE_AGENTS:
        graph.add_node(agent, make_remote_agent_node(agent))
        graph.add_edge(agent, END)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route_to_agent)
    return graph.compile(checkpointer=checkpointer)
