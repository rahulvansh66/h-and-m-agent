"""A2A service for the fashion agent (``python -m hm_agent.fashion.a2a_service``).

Role in the system: runs the fashion ReAct agent in its own process. It reaches its data through the
long-lived fashion MCP server (agent -> tool over streamable HTTP), which must already be running; A2A is only
the supervisor -> agent hop.
"""

import uvicorn
from dotenv import load_dotenv

from hm_agent.a2a.cards import build_agent_card
from hm_agent.a2a.server import build_a2a_app, graph_runner
from hm_agent.core.config import REPO_ROOT
from hm_agent.core.logging_setup import configure_logging
from hm_agent.core.settings import bind_address
from hm_agent.core.state import FASHION
from hm_agent.fashion.agent import build_fashion_agent


def main() -> None:
    """Start the fashion agent service on its configured port (or the ``A2A_BIND_*`` override)."""
    load_dotenv(REPO_ROOT / ".env")
    configure_logging(f"a2a-{FASHION}")
    host, port = bind_address(FASHION)  # local defaults; A2A_BIND_HOST / A2A_BIND_PORT override for Runtime
    # The ReAct agent's last model call is the customer-facing reply, so its tokens are streamed as written.
    runner = graph_runner(
        build_fashion_agent(), FASHION, progress_labels={"tools": "Looked through the catalog"}, stream_tokens=True
    )
    app = build_a2a_app(build_agent_card(FASHION), FASHION, runner)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
