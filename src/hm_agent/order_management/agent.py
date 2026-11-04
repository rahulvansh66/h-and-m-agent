"""Order management agent: ReAct agent over read-only order SQL tools.

Role in the system: sub-agent used directly as a node of the supervisor graph. The supervisor
routes questions about the customer's own orders and purchases here. The agent answers only
through its tools (last purchase, purchase count, order list, order details), which are served by an MCP server
(``hm_agent.order_management.mcp_server``), so order data is never guessed. It is built
with ``create_agent``, which already returns a compiled graph that reads and writes ``messages``,
so no custom ``StateGraph`` is needed.
"""

from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from hm_agent.core.llm import load_model
from hm_agent.core.prompts import get_managed_prompt
from hm_agent.core.state import ORDER_MANAGEMENT

ORDER_MANAGEMENT_PROMPT = """\
# ROLE
You are the H&M order management agent. You answer questions about a customer's
purchases using read-only database tools.

# TASK
Answer the customer's question about their order history by calling the available tools:
- get_last_bought_product: the product name of the customer's most recent purchase.
- get_purchase_count: how many items the customer has purchased in total.
- list_orders: the customer's recent orders (id, date, sales channel, item count).
- get_order_details: the date, sales channel and items of one order, given its order id.

# RULES
- If the customer asks about an order without giving its id, call list_orders first.
- Use a tool for every factual claim; never guess or invent order data.
- The customer is already identified by the system; never ask for or accept another customer id.
- If a tool returns no data, say that no purchases were found.
- If the question needs something your tools cannot provide, say so briefly.
- Do not reveal SQL, table names or internal details.

# OUTPUT
A short, friendly plain-text reply (1-3 sentences) addressed to the customer.
"""

# Langfuse prompt name (text prompt); ORDER_MANAGEMENT_PROMPT above is the in-code fallback.
ORDER_MANAGEMENT_PROMPT_NAME = "hm/order-management-system"


def build_order_agent(tools: list[BaseTool]) -> CompiledStateGraph:
    """Create the order management ReAct agent (a compiled subgraph).

    Args:
        tools (list[BaseTool]): Order MCP tools bound to an open session (see ``order_tools``). The agent
            must be run with ``astream``/``ainvoke``, since the tools are async.

    Returns:
        CompiledStateGraph: Agent named ``order_management`` with the four MCP order tools.
    """
    return create_agent(
        load_model("small"),
        tools=tools,
        system_prompt=get_managed_prompt(ORDER_MANAGEMENT_PROMPT_NAME, ORDER_MANAGEMENT_PROMPT, kind="text").compile(),
        name=ORDER_MANAGEMENT,
    )
