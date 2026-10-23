"""A2A service for the shipment tracking agent (``python -m hm_agent.shipment_tracking.a2a_service``).

Role in the system: runs the slow carrier lookup in its own process. Its Agent Card advertises
push notifications, so callers can send a task and be told when it finishes instead of waiting.
"""

import uvicorn
from dotenv import load_dotenv

from hm_agent.a2a.cards import build_agent_card
from hm_agent.a2a.server import build_a2a_app
from hm_agent.core.config import REPO_ROOT
from hm_agent.core.logging_setup import configure_logging
from hm_agent.core.settings import A2A_HOST, agent_port
from hm_agent.core.state import SHIPMENT_TRACKING
from hm_agent.shipment_tracking.agent import shipment_runner


def main() -> None:
    """Start the shipment tracking agent service on its configured port."""
    load_dotenv(REPO_ROOT / ".env")
    configure_logging(f"a2a-{SHIPMENT_TRACKING}")
    app = build_a2a_app(build_agent_card(SHIPMENT_TRACKING), SHIPMENT_TRACKING, shipment_runner)
    uvicorn.run(app, host=A2A_HOST, port=agent_port(SHIPMENT_TRACKING), log_level="info")


if __name__ == "__main__":
    main()
