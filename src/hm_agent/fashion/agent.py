"""Fashion agent: ReAct agent that recommends products and outfits from real catalog data.

Role in the system: sub-agent used as a node of the supervisor graph (through its A2A service). The supervisor
routes pairing, outfit and similar-product questions here. The agent answers only through its tools
(taste profile, purchase history, styling guide, semantic product search), served by the fashion MCP server
(``hm_agent.fashion.mcp_server``), so it never invents products. Built with ``create_agent``, which already
returns a compiled graph that reads and writes ``messages``.
"""

from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph

from hm_agent.core.llm import load_model
from hm_agent.core.state import FASHION
from hm_agent.fashion.mcp_client import load_fashion_tools

FASHION_PROMPT = """\
# ROLE
You are the H&M fashion agent. You recommend products and outfits from the H&M catalog
using read-only tools.

# TASK
Answer the customer's styling question by calling the available tools:
- search_products: find catalog products by meaning (style, occasion, item type). Can exclude product
  groups and product ids.
- get_purchase_history: what the customer bought recently (resolves "my last jacket", avoids repeats).
- get_styling_guide: short rules for pairing items into outfits. Call it before suggesting an outfit.
- get_style_profile: the customer's age and favourite product group.

# RULES
- Recommend only products returned by search_products or get_purchase_history; never invent items,
  prices, sizes or availability.
- To pair with an item, search for a different product group than that item (exclude its group).
- Do not recommend items the customer already owns; pass their ids as exclusions.
- Never suggest swimwear or underwear unless the customer asks for them.
- If search_products finds no good match, say so honestly instead of padding the answer.
- The customer is already identified by the system; never ask for or accept another customer id.
- Do not reveal SQL, table names, scores or internal details.

# OUTPUT
A short, friendly plain-text reply (2-4 sentences) naming the recommended products and how to wear them together.
"""


def build_fashion_agent() -> CompiledStateGraph:
    """Create the fashion ReAct agent (a compiled subgraph).

    The tools are loaded from the running fashion MCP server (blocking, once at build time) and exposed
    as sync tools, so the agent runs with plain ``invoke``/``stream``.

    Returns:
        CompiledStateGraph: Agent named ``fashion`` with the four MCP fashion tools.
    """
    return create_agent(
        load_model("small"),
        tools=load_fashion_tools(),
        system_prompt=FASHION_PROMPT,
        name=FASHION,
    )
