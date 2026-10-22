"""MCP client: loads the order management tools from their MCP server as LangChain tools.

Role in the system: bridge between the order_management agent and its MCP server
(``hm_agent.order_management.mcp_server``). Two things happen here beyond plain loading:

* the ``customer_id`` argument is hidden from the model's tool schema, and
* an interceptor overwrites it on every call with the id from the run config,

so the LLM can never read another customer's data, same guarantee as the former in-process tools.
"""

import asyncio
import logging
import os
import sys
from collections.abc import Awaitable, Callable

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest, MCPToolCallResult

from hm_agent.core.settings import DEFAULT_CUSTOMER_ID

logger = logging.getLogger(__name__)

ORDER_SERVER = "order_management"
CUSTOMER_ID_ARG = "customer_id"
# Passed to the server explicitly: stdio subprocesses only inherit a minimal environment by default.
FORWARDED_ENV = ("NEON_POSTGRES_CONNECTION_STRING",)


async def inject_customer_id(
    request: MCPToolCallRequest,
    handler: Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]],
) -> MCPToolCallResult:
    """Tool-call interceptor that sets ``customer_id`` from the run config.

    Any value the model supplied is discarded. Falls back to the default customer when the
    config carries none, matching the CLI default.

    Args:
        request: The outgoing MCP tool call; ``request.runtime`` holds the LangGraph runtime.
        handler: Executes the (possibly modified) request.

    Returns:
        MCPToolCallResult: The tool result from the MCP server.
    """
    config = getattr(request.runtime, "config", None) or {}
    customer_id = config.get("configurable", {}).get(CUSTOMER_ID_ARG, DEFAULT_CUSTOMER_ID)
    return await handler(request.override(args={**request.args, CUSTOMER_ID_ARG: customer_id}))


def _hide_customer_id(tool: BaseTool) -> BaseTool:
    """Remove ``customer_id`` from a tool's model-facing JSON schema.

    Args:
        tool: A tool loaded from MCP, whose ``args_schema`` is a JSON-schema dict.

    Returns:
        BaseTool: The same tool, with the argument no longer advertised to the model.
    """
    schema = dict(tool.args_schema)
    schema["properties"] = {k: v for k, v in schema.get("properties", {}).items() if k != CUSTOMER_ID_ARG}
    schema["required"] = [name for name in schema.get("required", []) if name != CUSTOMER_ID_ARG]
    tool.args_schema = schema
    return tool


def _make_sync(tool: BaseTool) -> BaseTool:
    """Give an async-only MCP tool a sync entry point.

    The graph is run with ``invoke`` and a sync Postgres checkpointer, so tools must be callable
    synchronously. LangGraph runs sync tools in worker threads, which have no running event loop,
    so ``asyncio.run`` is safe there (and on Windows it uses the Proactor loop that the stdio
    subprocess transport needs).

    Args:
        tool: A tool loaded from MCP, which only defines a coroutine.

    Returns:
        BaseTool: The same tool with ``func`` set to run its coroutine to completion.
    """
    coroutine = tool.coroutine

    # ``runtime`` is injected by LangGraph and forwarded so interceptors can read the run config.
    def run(runtime: object | None = None, **arguments: object) -> object:
        return asyncio.run(coroutine(runtime=runtime, **arguments))

    tool.func = run
    return tool


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
    tools = [_make_sync(_hide_customer_id(tool)) for tool in asyncio.run(client.get_tools())]
    logger.info("component=mcp_client server=%s tools=%d", ORDER_SERVER, len(tools))
    return tools
