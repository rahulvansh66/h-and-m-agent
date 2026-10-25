"""MCP client: loads the order management tools from their MCP server as LangChain tools.

Role in the system: bridge between the order_management agent and its MCP server
(``hm_agent.order_management.mcp_server``). The ``customer_id`` hiding, injection and sync wrapping
are shared with the other agents (see ``hm_agent.core.mcp_tools``), so the LLM can never read another
customer's data.
"""

import asyncio
import logging
import os
import sys

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from hm_agent.core.mcp_tools import hide_customer_id, inject_customer_id, make_sync

logger = logging.getLogger(__name__)

ORDER_SERVER = "order_management"
# Passed to the server explicitly: stdio subprocesses only inherit a minimal environment by default.
FORWARDED_ENV = ("NEON_POSTGRES_CONNECTION_STRING",)


def load_order_tools() -> list[BaseTool]:
    """Load the order management tools from the MCP server.

    Blocks while connecting once to list the tools. Afterwards the client starts the server as a
    stdio subprocess for each tool call, so nothing stays running between calls.

    Returns:
        list[BaseTool]: Sync-callable tools ``get_last_bought_product``, ``get_purchase_count``,
            ``list_orders`` and ``get_order_details``.
    """
    env = {key: os.environ[key] for key in FORWARDED_ENV if key in os.environ}
    client = MultiServerMCPClient(
        {
            ORDER_SERVER: {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "hm_agent.order_management.mcp_server"],
                "env": env,
            }
        },
        tool_interceptors=[inject_customer_id],
    )
    tools = [make_sync(hide_customer_id(tool)) for tool in asyncio.run(client.get_tools())]
    logger.info("component=mcp_client server=%s tools=%d", ORDER_SERVER, len(tools))
    return tools
