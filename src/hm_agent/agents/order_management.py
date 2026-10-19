"""Order management agent: ReAct agent over read-only order SQL tools.

Role in the system: sub-agent used directly as a node of the supervisor graph. The supervisor
routes questions about the customer's own orders and purchases here. The agent answers only
through its tools (last purchase, purchase count), which are served by an MCP server
(``hm_agent.mcp_servers.order_management``), so order data is never guessed. It is built
with ``create_agent``, which already returns a compiled graph that reads and writes ``messages``,
so no custom ``StateGraph`` is needed.
"""

from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph

from hm_agent.llm import load_model
from hm_agent.mcp_client import load_order_tools
from hm_agent.state import ORDER_MANAGEMENT

ORDER_MANAGEMENT_PROMPT = """\
# ROLE
You are the H&M order management agent. You answer questions about a customer's
purchases using read-only database tools.

# TASK
Answer the customer's question about their order history by calling the available tools:
- get_last_bought_product: the product name of the customer's most recent purchase.
- get_purchase_count: how many items the customer has purchased in total.

# RULES
- Use a tool for every factual claim; never guess or invent order data.
- The customer is already identified by the system; never ask for or accept another customer id.
- If a tool returns no data, say that no purchases were found.
- If the question needs something your tools cannot provide, say so briefly.
- Do not reveal SQL, table names or internal details.

# OUTPUT
A short, friendly plain-text reply (1-3 sentences) addressed to the customer.
"""


def build_order_agent() -> CompiledStateGraph:
    """Create the order management ReAct agent (a compiled subgraph).

    The tools are loaded from the MCP server (blocking, once at build time) and exposed as
    sync tools, so the agent runs with plain ``invoke``/``stream``.

    Returns:
        CompiledStateGraph: Agent named ``order_management`` with the two MCP order tools.
    """
    return create_agent(
        load_model("small"),
        tools=load_order_tools(),
        system_prompt=ORDER_MANAGEMENT_PROMPT,
        name=ORDER_MANAGEMENT,
    )
