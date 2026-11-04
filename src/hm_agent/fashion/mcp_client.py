"""MCP client: loads the fashion tools from the long-lived fashion MCP server as LangChain tools.

Role in the system: bridge between the fashion agent and its MCP server (``hm_agent.fashion.mcp_server``).
The server runs as its own process (streamable HTTP), so it holds the embedding model and the database and
Qdrant credentials; this client only needs its URL. ``customer_id`` handling and session reuse are shared
with the other agents (see ``hm_agent.core.mcp_tools``).
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from hm_agent.core.mcp_tools import inject_customer_id, open_mcp_tools
from hm_agent.core.settings import FASHION_MCP_HOST, FASHION_MCP_PORT

logger = logging.getLogger(__name__)

FASHION_SERVER = "fashion"
MCP_PATH = "/mcp"  # default path of FastMCP's streamable HTTP endpoint


@asynccontextmanager
async def fashion_tools() -> AsyncGenerator[list[BaseTool]]:
    """Open one session to the fashion MCP server and yield its tools until exit.

    The server must already be running (``scripts/a2a/run_agents.py`` starts it before the fashion
    service). The session stays open for the whole ``async with``, so every tool call reuses it.

    Yields:
        list[BaseTool]: Async tools ``get_style_profile``, ``get_purchase_history``,
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
    async with open_mcp_tools(client, FASHION_SERVER) as tools:
        logger.info("component=mcp_client server=%s tools=%d", FASHION_SERVER, len(tools))
        yield tools
