"""Shared constants: repo paths and values read from the layered YAML config.

Role in the system: single place for values used by several modules.
"""

import os

from hm_agent.core.config import REPO_ROOT, load_config

_CONFIG = load_config()

DEFAULT_CUSTOMER_ID = _CONFIG["defaults"]["customer_id"]
RAG_COLLECTION = _CONFIG["rag"]["collection"]  # Qdrant collection holding the support knowledge base
RAG_TOP_K = _CONFIG["rag"]["top_k"]  # documents fetched per retrieval
RAG_MAX_RETRIES = _CONFIG["rag"]["max_retries"]  # query-rewrite retries when the answer has no citation
RAG_MIN_SCORE = _CONFIG["rag"]["min_score"]  # minimum similarity for a chunk to reach the LLM
RAG_CONDENSE_HISTORY = _CONFIG["rag"]["condense_history_messages"]  # messages used to condense the query
SUPERVISOR_HISTORY = _CONFIG["supervisor"]["history_messages"]  # messages the supervisor sees when routing
CHECKPOINT_SCHEMA = _CONFIG["database"]["checkpoint_schema"]  # Neon schema for LangGraph checkpoints
LOG_LEVEL = _CONFIG["logging"]["level"]
TRACING_TAGS = _CONFIG["tracing"]["tags"]
A2A_HOST = _CONFIG["a2a"]["host"]  # interface the agent services bind to
A2A_TIMEOUT_S = _CONFIG["a2a"]["request_timeout_s"]  # supervisor-side timeout per agent call
A2A_HISTORY = _CONFIG["a2a"]["history_messages"]  # prior messages sent to an agent as context
A2A_PORTS = {name: spec["port"] for name, spec in _CONFIG["a2a"]["agents"].items()}  # agent name -> port


def agent_port(agent: str) -> int:
    """Return the port an agent service listens on.

    Args:
        agent (str): Agent name, e.g. ``order_management``.

    Returns:
        int: Port from the ``a2a.agents`` config section.
    """
    return A2A_PORTS[agent]


def agent_url(agent: str) -> str:
    """Return the base URL of an agent service.

    The env var ``<AGENT>_AGENT_URL`` (e.g. ``ORDER_MANAGEMENT_AGENT_URL``) overrides the
    config-derived local URL, so a deployment can point the supervisor at remote hosts. It is read
    at call time because ``.env`` is loaded after this module is imported.

    Args:
        agent (str): Agent name, e.g. ``order_management``.

    Returns:
        str: Base URL such as ``http://127.0.0.1:8001`` (no trailing slash).
    """
    override = os.environ.get(f"{agent.upper()}_AGENT_URL")
    return (override or f"http://{A2A_HOST}:{A2A_PORTS[agent]}").rstrip("/")


__all__ = [
    "REPO_ROOT",
    "DEFAULT_CUSTOMER_ID",
    "RAG_COLLECTION",
    "RAG_TOP_K",
    "RAG_MAX_RETRIES",
    "RAG_MIN_SCORE",
    "RAG_CONDENSE_HISTORY",
    "SUPERVISOR_HISTORY",
    "CHECKPOINT_SCHEMA",
    "LOG_LEVEL",
    "TRACING_TAGS",
    "A2A_HOST",
    "A2A_TIMEOUT_S",
    "A2A_HISTORY",
    "A2A_PORTS",
    "agent_port",
    "agent_url",
]
