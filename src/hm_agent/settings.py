"""Shared constants: repo paths and values read from the layered YAML config.

Role in the system: single place for values used by several modules.
"""

from hm_agent.config import REPO_ROOT, load_config

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
]
