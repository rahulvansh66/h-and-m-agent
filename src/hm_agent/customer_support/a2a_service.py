"""A2A service for the customer support agent (``python -m hm_agent.customer_support.a2a_service``).

Role in the system: runs the condense -> RAG -> respond pipeline in its own process, so the heavy
parts (local embedding model, Qdrant client) load here and not in the supervisor.
"""

import uvicorn
from dotenv import load_dotenv

from hm_agent.a2a.cards import build_agent_card
from hm_agent.a2a.server import build_a2a_app, graph_runner
from hm_agent.customer_support.agent import build_customer_support_agent
from hm_agent.core.config import REPO_ROOT
from hm_agent.core.logging_setup import configure_logging
from hm_agent.core.settings import A2A_HOST, agent_port
from hm_agent.core.state import CUSTOMER_SUPPORT


def main() -> None:
    """Start the customer support agent service on its configured port."""
    load_dotenv(REPO_ROOT / ".env")
    configure_logging(f"a2a-{CUSTOMER_SUPPORT}")
    runner = graph_runner(build_customer_support_agent(), CUSTOMER_SUPPORT)
    uvicorn.run(build_a2a_app(build_agent_card(CUSTOMER_SUPPORT), CUSTOMER_SUPPORT, runner), host=A2A_HOST, port=agent_port(CUSTOMER_SUPPORT), log_level="info")


if __name__ == "__main__":
    main()
