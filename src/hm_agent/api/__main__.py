"""Start the API: ``uv run python -m hm_agent.api``.

Role in the system: entry point of the FastAPI layer. Loads ``.env``, configures logging once,
refuses to start without an ``API_KEY`` and serves the app with uvicorn. The A2A agent services
(``scripts/a2a/run_agents.py``) must already be running for chats to be answered.
"""

import logging
import os
import sys

import uvicorn
from dotenv import load_dotenv

from hm_agent.api.app import create_app
from hm_agent.api.deps import API_KEY_ENV
from hm_agent.core.config import REPO_ROOT
from hm_agent.core.logging_setup import configure_logging
from hm_agent.core.settings import API_HOST, API_PORT

logger = logging.getLogger(__name__)


def main() -> None:
    """Load the environment, check the API key is set and run the server."""
    load_dotenv(REPO_ROOT / ".env")
    configure_logging("api")
    if not os.environ.get(API_KEY_ENV):
        logger.error("component=api status=refused reason=%s_not_set", API_KEY_ENV)
        sys.exit(f"{API_KEY_ENV} is not set; add it to .env (see .env.example)")
    uvicorn.run(create_app(), host=API_HOST, port=API_PORT, log_level="info")


if __name__ == "__main__":
    main()
