"""A2A service for the order management agent (``python -m hm_agent.order_management.a2a_service``).

Role in the system: runs the order ReAct agent in its own process. It still reaches its data
through the MCP order server (agent -> tool over MCP stdio); A2A is only the supervisor -> agent hop.
The MCP session is opened in the app's lifespan and held until shutdown.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from starlette.applications import Starlette

from hm_agent.a2a.cards import build_agent_card
from hm_agent.a2a.server import Runner, build_a2a_app, deferred_runner, graph_runner
from hm_agent.core.config import REPO_ROOT
from hm_agent.core.logging_setup import configure_logging
from hm_agent.core.settings import bind_address
from hm_agent.core.state import ORDER_MANAGEMENT
from hm_agent.order_management.agent import build_order_agent
from hm_agent.order_management.mcp_client import order_tools

# Set while the lifespan runs (MCP session open); None before startup and after shutdown.
_runner: Runner | None = None


def _get_runner() -> Runner:
    """Return the live runner.

    Returns:
        Runner: Runner bound to the open MCP session.

    Raises:
        RuntimeError: If called before startup or after shutdown.
    """
    if _runner is None:
        raise RuntimeError("order management service is not started")
    return _runner


@asynccontextmanager
async def lifespan(_app: Starlette) -> AsyncGenerator[None]:
    """Open the MCP session, build the agent on it, and close the session on shutdown.

    Args:
        _app (Starlette): The app being served (unused).

    Yields:
        None: Control to the server while the session is open.
    """
    global _runner
    async with order_tools() as tools:
        # The ReAct agent's last model call is the customer-facing reply, so its tokens are streamed as written.
        _runner = graph_runner(
            build_order_agent(tools),
            ORDER_MANAGEMENT,
            progress_labels={"tools": "Looked up your order data"},
            stream_tokens=True,
        )
        yield
    _runner = None


def main() -> None:
    """Start the order management agent service on its configured port (or the ``A2A_BIND_*`` override)."""
    load_dotenv(REPO_ROOT / ".env")
    configure_logging(f"a2a-{ORDER_MANAGEMENT}")
    host, port = bind_address(ORDER_MANAGEMENT)  # local defaults; A2A_BIND_HOST / A2A_BIND_PORT override for Runtime
    app = build_a2a_app(
        build_agent_card(ORDER_MANAGEMENT), ORDER_MANAGEMENT, deferred_runner(_get_runner), lifespan=lifespan
    )
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
