"""A2A service for the fashion agent (``python -m hm_agent.fashion.a2a_service``).

Role in the system: runs the fashion ReAct agent in its own process. It reaches its data through the
long-lived fashion MCP server (agent -> tool over streamable HTTP), which must already be running; A2A is only
the supervisor -> agent hop. The MCP session is opened in the app's lifespan and held until shutdown.
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
from hm_agent.core.state import FASHION
from hm_agent.fashion.agent import build_fashion_agent
from hm_agent.fashion.mcp_client import fashion_tools

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
        raise RuntimeError("fashion service is not started")
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
    async with fashion_tools() as tools:
        # The ReAct agent's last model call is the customer-facing reply, so its tokens are streamed as written.
        _runner = graph_runner(
            build_fashion_agent(tools), FASHION, progress_labels={"tools": "Looked through the catalog"}, stream_tokens=True
        )
        yield
    _runner = None


def main() -> None:
    """Start the fashion agent service on its configured port (or the ``A2A_BIND_*`` override)."""
    load_dotenv(REPO_ROOT / ".env")
    configure_logging(f"a2a-{FASHION}")
    host, port = bind_address(FASHION)  # local defaults; A2A_BIND_HOST / A2A_BIND_PORT override for Runtime
    app = build_a2a_app(build_agent_card(FASHION), FASHION, deferred_runner(_get_runner), lifespan=lifespan)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
