"""Shared helpers for loading MCP tools into an agent.

Role in the system: used by every agent's MCP client (``order_management``, ``fashion``). Three things
are common to all of them:

* the ``customer_id`` argument is hidden from the model's tool schema,
* an interceptor overwrites it on every call with the id from the run config, and
* one MCP session is opened at service startup and reused by every tool call,

so the LLM can never read another customer's data and a tool call does not pay a connect + ``initialize``.
"""

from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest, MCPToolCallResult
from langchain_mcp_adapters.tools import load_mcp_tools

from hm_agent.core.settings import DEFAULT_CUSTOMER_ID, auth_enabled

CUSTOMER_ID_ARG = "customer_id"


async def inject_customer_id(
    request: MCPToolCallRequest,
    handler: Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]],
) -> MCPToolCallResult:
    """Tool-call interceptor that sets ``customer_id`` from the run config.

    Any value the model supplied is discarded. Falls back to the default customer when the
    config carries none, matching the local CLI default; with auth enabled there is no fallback,
    so a missing id can never silently become another customer's data.

    Args:
        request: The outgoing MCP tool call; ``request.runtime`` holds the LangGraph runtime.
        handler: Executes the (possibly modified) request.

    Returns:
        MCPToolCallResult: The tool result from the MCP server.

    Raises:
        PermissionError: If auth is enabled and the run config carries no customer id.
    """
    config = getattr(request.runtime, "config", None) or {}
    customer_id = config.get("configurable", {}).get(CUSTOMER_ID_ARG)
    if not customer_id:
        if auth_enabled():
            raise PermissionError("no verified customer id in the run config")
        customer_id = DEFAULT_CUSTOMER_ID

    # SECURITY: the trusted id goes last, so it overrides anything the model sent.
    return await handler(request.override(args={**request.args, CUSTOMER_ID_ARG: customer_id}))


def hide_customer_id(tool: BaseTool) -> BaseTool:
    """Remove ``customer_id`` from a tool's model-facing JSON schema.

    Args:
        tool: A tool loaded from MCP, whose ``args_schema`` is a JSON-schema dict.

    Returns:
        BaseTool: The same tool, with the argument no longer advertised to the model.
    """
    # Copy first, so the shared schema is not mutated.
    schema = dict(tool.args_schema)
    schema["properties"] = {k: v for k, v in schema.get("properties", {}).items() if k != CUSTOMER_ID_ARG}
    schema["required"] = [name for name in schema.get("required", []) if name != CUSTOMER_ID_ARG]
    tool.args_schema = schema
    return tool


@asynccontextmanager
async def open_mcp_tools(client: MultiServerMCPClient, server_name: str) -> AsyncGenerator[list[BaseTool]]:
    """Open one MCP session, load its tools bound to it, and close the session on exit.

    Tools from ``client.get_tools()`` open a new session for every call. Tools loaded from an open
    session reuse it, so the connect and ``initialize`` cost is paid once. Enter this from the
    service's lifespan so the session lives on the server's own event loop.

    Args:
        client (MultiServerMCPClient): Client configured with the server connection and interceptors.
        server_name (str): Key of the server in the client configuration.

    Yields:
        list[BaseTool]: Async tools (``customer_id`` hidden) that share the one open session.
    """
    async with client.session(server_name) as session:
        tools = await load_mcp_tools(
            session, tool_interceptors=client.tool_interceptors, server_name=server_name
        )
        yield [hide_customer_id(tool) for tool in tools]
