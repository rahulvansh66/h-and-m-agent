"""MCP client: loads the fashion tools from the long-lived fashion MCP server as LangChain tools.

Role in the system: bridge between the fashion agent and its MCP server (``hm_agent.fashion.mcp_server``).
The server runs as its own process (streamable HTTP), so it holds the embedding model and the database and
Qdrant credentials; this client only needs its URL. ``customer_id`` handling is shared with the other agents
(see ``hm_agent.core.mcp_tools``).
"""

import asyncio
import logging

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from hm_agent.core.mcp_tools import hide_customer_id, inject_customer_id, make_sync
from hm_agent.core.settings import FASHION_MCP_HOST, FASHION_MCP_PORT

logger = logging.getLogger(__name__)

FASHION_SERVER = "fashion"
MCP_PATH = "/mcp"  # default path of FastMCP's streamable HTTP endpoint


def load_fashion_tools() -> list[BaseTool]:
    """Load the fashion tools from the running fashion MCP server.

    Blocks while connecting once to list the tools, so the server must already be running
    (``scripts/a2a/run_agents.py`` starts it before the fashion service).

    Returns:
        list[BaseTool]: Sync-callable tools ``get_style_profile``, ``get_purchase_history``,
            ``get_styling_guide`` and ``search_products``.
    """
    client = MultiServerMCPClient(
        {
            FASHION_SERVER: {
                "transport": "streamable_http",
                "url": f"http://{FASHION_MCP_HOST}:{FASHION_MCP_PORT}{MCP_PATH}",
            }
        },
        tool_interceptors=[inject_customer_id],
    )
    tools = [make_sync(hide_customer_id(tool)) for tool in asyncio.run(client.get_tools())]
    logger.info("component=mcp_client server=%s tools=%d", FASHION_SERVER, len(tools))
    return tools
