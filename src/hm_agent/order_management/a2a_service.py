"""A2A service for the order management agent (``python -m hm_agent.order_management.a2a_service``).

Role in the system: runs the order ReAct agent in its own process. It still reaches its data
through the MCP order server (agent -> tool over MCP stdio); A2A is only the supervisor -> agent hop.
"""

import uvicorn
from dotenv import load_dotenv

from hm_agent.a2a.cards import build_agent_card
from hm_agent.a2a.server import build_a2a_app, graph_runner
from hm_agent.order_management.agent import build_order_agent
from hm_agent.core.config import REPO_ROOT
from hm_agent.core.logging_setup import configure_logging
from hm_agent.core.settings import A2A_HOST, agent_port
from hm_agent.core.state import ORDER_MANAGEMENT


def main() -> None:
    """Start the order management agent service on its configured port."""
    load_dotenv(REPO_ROOT / ".env")
    configure_logging(f"a2a-{ORDER_MANAGEMENT}")
    # The ReAct agent's last model call is the customer-facing reply, so its tokens are streamed as written.
    runner = graph_runner(
        build_order_agent(), ORDER_MANAGEMENT, progress_labels={"tools": "Looked up your order data"}, stream_tokens=True
    )
    app = build_a2a_app(build_agent_card(ORDER_MANAGEMENT), ORDER_MANAGEMENT, runner)
    uvicorn.run(app, host=A2A_HOST, port=agent_port(ORDER_MANAGEMENT), log_level="info")


if __name__ == "__main__":
    main()
