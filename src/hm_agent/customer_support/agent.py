"""Customer support agent: answers damaged / defective item questions from the help articles.

Role in the system: sub-agent used as a node of the supervisor graph. It is a fixed pipeline
(not a tool-calling agent) because every question it receives needs the knowledge base:

    condense_query -> rag (retrieve -> generate -> rewrite/retry) -> respond

Scope is decided by the supervisor, so this agent assumes the message is its own. Questions the
articles do not cover are caught by the RAG relevance gate. The RAG subgraph keeps its own
private state (documents, claims, retries); only ``messages`` flows back to the parent graph.
"""

import logging

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

from hm_agent.core.llm import load_model
from hm_agent.customer_support.rag.pipeline import build_rag_graph
from hm_agent.core.settings import RAG_CONDENSE_HISTORY
from hm_agent.core.state import CUSTOMER_SUPPORT

logger = logging.getLogger(__name__)

CONDENSE_PROMPT = """\
# ROLE
You rewrite the latest customer message into a standalone question.

# TASK
Using the conversation, rewrite the latest customer message so it can be understood without
the earlier messages (resolve references such as "it", "the second one", "that order").

# RULES
- Keep the customer's intent and wording as close as possible; do not add new facts.
- If the message is already standalone, return it unchanged.
- Do not answer the question.

# OUTPUT
The standalone question only, on one line.
"""


class CustomerSupportState(MessagesState):
    """State of the customer support subgraph.

    Attributes:
        query: Standalone question produced by ``condense_query`` and read by the RAG subgraph.
        answer: Cited answer produced by the RAG subgraph and read by ``respond``.
    """

    query: str
    answer: str


def _role_label(message: AnyMessage) -> str:
    """Return the speaker label used when rendering a message in the condensing prompt.

    Args:
        message: A conversation message.

    Returns:
        str: ``"Customer"`` for human messages, ``"Assistant"`` otherwise.
    """
    return "Customer" if isinstance(message, HumanMessage) else "Assistant"


def condense_query_node(state: CustomerSupportState) -> dict:
    """Turn the latest customer message into a standalone question for retrieval.

    On the first turn there is no history to resolve, so the message is used as is and no LLM
    call is made. Otherwise one small-model call rewrites it using the recent history.

    Reads: ``messages``. Writes: ``query``.

    Args:
        state: Current subgraph state.

    Returns:
        dict: ``{"query": str}`` with the standalone question.
    """
    messages = state["messages"]
    latest = messages[-1].text

    # The first customer message is already standalone.
    if sum(isinstance(m, HumanMessage) for m in messages) <= 1:
        return {"query": latest}

    history = "\n".join(f"{_role_label(m)}: {m.text}" for m in messages[-RAG_CONDENSE_HISTORY:])
    response = load_model("small").invoke(
        [SystemMessage(CONDENSE_PROMPT), HumanMessage(f"Conversation:\n{history}")]
    )
    query = str(response.content).strip() or latest
    logger.info("node=condense_query status=complete history_messages=%d", len(messages))
    return {"query": query}


def respond_node(state: CustomerSupportState) -> dict:
    """Turn the RAG answer into the AI message the parent graph expects.

    The message is named after the agent (``CUSTOMER_SUPPORT``, the same name as its graph
    node) so the supervisor can keep follow-up questions with it on the next turn.

    Reads: ``answer``. Writes: ``messages`` (one AI reply).

    Args:
        state: Current subgraph state.

    Returns:
        dict: ``{"messages": [AIMessage]}`` containing the cited answer.
    """
    return {"messages": [AIMessage(state["answer"], name=CUSTOMER_SUPPORT)]}


def build_customer_support_agent() -> CompiledStateGraph:
    """Create the customer support subgraph (a compiled graph used as a parent-graph node).

    Returns:
        CompiledStateGraph: Graph ``condense_query -> rag -> respond``. It reads and writes
        ``messages`` only as far as the parent graph is concerned.
    """
    graph = StateGraph(CustomerSupportState)
    graph.add_node("condense_query", condense_query_node)
    # The RAG graph shares only ``query`` / ``answer`` with this state, so its documents,
    # claims and retry counter stay private to it.
    graph.add_node("rag", build_rag_graph())
    graph.add_node("respond", respond_node)
    
    graph.add_edge(START, "condense_query")
    graph.add_edge("condense_query", "rag")
    graph.add_edge("rag", "respond")
    graph.add_edge("respond", END)
    return graph.compile()
