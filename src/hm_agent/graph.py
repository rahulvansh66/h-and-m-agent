"""Graph assembly for the supervisor topology.

    START -> supervisor -> order_management  -> END
                        -> customer_support  -> END
                        -> out_of_scope      -> END

Role in the system: wires the supervisor and sub-agents into one compiled graph.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from hm_agent.agents.customer_support import build_customer_support_agent
from hm_agent.agents.order_management import build_order_agent
from hm_agent.agents.supervisor import out_of_scope_node, route_to_agent, supervisor_node
from hm_agent.state import CUSTOMER_SUPPORT, ORDER_MANAGEMENT, OUT_OF_SCOPE, AgentState


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    """Assemble and compile the supervisor graph.

    Args:
        checkpointer: Optional checkpointer that persists state per ``thread_id``.

    Returns:
        CompiledStateGraph: Graph with nodes ``supervisor``, ``order_management``,
        ``customer_support`` and ``out_of_scope``.
    """
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    # The order agent is a self-contained ReAct subgraph used directly as a node.
    graph.add_node(ORDER_MANAGEMENT, build_order_agent())
    # Customer support is a condense -> RAG -> respond subgraph with its own private state.
    graph.add_node(CUSTOMER_SUPPORT, build_customer_support_agent())
    graph.add_node(OUT_OF_SCOPE, out_of_scope_node)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route_to_agent)
    graph.add_edge(ORDER_MANAGEMENT, END)
    graph.add_edge(CUSTOMER_SUPPORT, END)
    graph.add_edge(OUT_OF_SCOPE, END)
    return graph.compile(checkpointer=checkpointer)
