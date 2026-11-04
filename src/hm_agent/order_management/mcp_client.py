"""MCP client: loads the order management tools from their MCP server as LangChain tools.

Role in the system: bridge between the order_management agent and its MCP server
(``hm_agent.order_management.mcp_server``). The ``customer_id`` hiding, injection and session reuse
are shared with the other agents (see ``hm_agent.core.mcp_tools``), so the LLM can never read another
customer's data.
"""

import logging
import os
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from hm_agent.core.mcp_tools import inject_customer_id, open_mcp_tools

logger = logging.getLogger(__name__)

ORDER_SERVER = "order_management"
# Passed to the server explicitly: stdio subprocesses only inherit a minimal environment by default.
FORWARDED_ENV = ("NEON_POSTGRES_CONNECTION_STRING", "PYTHONPATH")  # PYTHONPATH: the container runs from /app/src


@asynccontextmanager
async def order_tools() -> AsyncGenerator[list[BaseTool]]:
    """Open one session to the order MCP server and yield its tools until exit.

    The session starts the server as a stdio subprocess once and keeps it for the whole ``async with``,
    so a tool call no longer pays a process start.

    Yields:
        list[BaseTool]: Async tools ``get_last_bought_product``, ``get_purchase_count``,
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
    async with open_mcp_tools(client, ORDER_SERVER) as tools:
        logger.info("component=mcp_client server=%s tools=%d", ORDER_SERVER, len(tools))
        yield tools
