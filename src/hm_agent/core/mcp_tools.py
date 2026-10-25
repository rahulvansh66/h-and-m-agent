"""Shared helpers for loading MCP tools into an agent.

Role in the system: used by every agent's MCP client (``order_management``, ``fashion``). Three things
are common to all of them:

* the ``customer_id`` argument is hidden from the model's tool schema,
* an interceptor overwrites it on every call with the id from the run config, and
* async-only MCP tools get a sync entry point,

so the LLM can never read another customer's data and the graph can run with plain ``invoke``.
"""

import asyncio
from collections.abc import Awaitable, Callable

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.interceptors import MCPToolCallRequest, MCPToolCallResult

from hm_agent.core.settings import DEFAULT_CUSTOMER_ID

CUSTOMER_ID_ARG = "customer_id"


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


def hide_customer_id(tool: BaseTool) -> BaseTool:
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


def make_sync(tool: BaseTool) -> BaseTool:
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
