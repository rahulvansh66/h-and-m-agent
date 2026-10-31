"""Graph assembly for the supervisor topology.

    START -> supervisor -> order_management  (A2A) -> END
                        -> customer_support  (A2A) -> END
                        -> shipment_tracking (A2A, push notifications) -> END
                        -> fashion           (A2A) -> END
                        -> out_of_scope      (local) -> END

    A message that mixes topics goes to two agents in parallel; their answers then meet in
    ``synthesize``:  supervisor -> (agent A, agent B) -> synthesize -> END

Role in the system: wires the supervisor and the agents into one compiled graph. The real
agents run as their own A2A services (``hm_agent.<agent>.a2a_service``); here each is only a node that
calls it (``shipment_tracking`` returns an acknowledgement and its result is added to the thread
later, by the push receiver). ``out_of_scope`` is a local node returning the supervisor's fixed decline reply.
``synthesize`` runs only after a split turn and merges the branch answers into one reply.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from hm_agent.a2a.remote_node import make_remote_agent_node
from hm_agent.supervisor.agent import out_of_scope_node, route_after_agent, route_to_agent, supervisor_node
from hm_agent.supervisor.synthesis import synthesize_node
from hm_agent.core.state import OUT_OF_SCOPE, SPECIALIST_AGENTS, AgentState

REMOTE_AGENTS = SPECIALIST_AGENTS  # agents served as remote A2A services


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    """Assemble and compile the supervisor graph.

    The agent services must be running for a turn routed to them to get an answer (see
    ``scripts/a2a/run_agents.py``); out-of-scope replies are produced locally.
    Conversation state lives only here, in the supervisor's checkpointer.

    Args:
        checkpointer: Optional checkpointer that persists state per ``thread_id``.

    Returns:
        CompiledStateGraph: Graph with nodes ``supervisor``, ``order_management``,
        ``customer_support``, ``shipment_tracking``, ``fashion``, ``out_of_scope`` and ``synthesize``.
    """
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node(OUT_OF_SCOPE, out_of_scope_node)
    graph.add_edge(OUT_OF_SCOPE, END)
    graph.add_node("synthesize", synthesize_node)
    graph.add_edge("synthesize", END)
    for agent in REMOTE_AGENTS:
        graph.add_node(agent, make_remote_agent_node(agent))
        # A single-agent turn ends here; a split turn goes on to merge the branch answers.
        graph.add_conditional_edges(agent, route_after_agent, {"synthesize": "synthesize", END: END})
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route_to_agent)
    return graph.compile(checkpointer=checkpointer)
