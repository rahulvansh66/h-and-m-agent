"""Shared constants: repo paths and values read from the layered YAML config.

Role in the system: single place for values used by several modules.
"""

import os

from hm_agent.core.config import REPO_ROOT, load_config

_CONFIG = load_config()

DEFAULT_CUSTOMER_ID = _CONFIG["defaults"]["customer_id"]
RAG_COLLECTION = _CONFIG["rag"]["collection"]  # Qdrant collection holding the support knowledge base
RAG_EMBEDDING = _CONFIG["rag"]["embedding"]  # fastembed model of the knowledge-base collection
RAG_TOP_K = _CONFIG["rag"]["top_k"]  # documents fetched per retrieval
RAG_MAX_RETRIES = _CONFIG["rag"]["max_retries"]  # query-rewrite retries when the answer has no citation
RAG_MIN_SCORE = _CONFIG["rag"]["min_score"]  # minimum similarity for a chunk to reach the LLM
RAG_CONDENSE_HISTORY = _CONFIG["rag"]["condense_history_messages"]  # messages used to condense the query
SUPERVISOR_HISTORY = _CONFIG["supervisor"]["history_messages"]  # messages the supervisor sees when routing
SUPERVISOR_MAX_AGENTS = _CONFIG["supervisor"]["max_agents"]  # agents a multi-intent message may fan out to
SUPERVISOR_STICKY_MAX_WORDS = _CONFIG["supervisor"]["sticky_max_words"]  # longest follow-up kept with the last agent without an LLM call
FASHION_COLLECTION = _CONFIG["fashion"]["collection"]  # Qdrant collection holding the product catalog vectors
FASHION_EMBEDDING = _CONFIG["fashion"]["embedding"]  # fastembed model of the product catalog collection
FASHION_TOP_K = _CONFIG["fashion"]["top_k"]  # products fetched per search
FASHION_MIN_SCORE = _CONFIG["fashion"]["min_score"]  # minimum similarity for a product to be returned
FASHION_MCP_HOST = _CONFIG["fashion"]["mcp_host"]  # interface of the long-lived fashion MCP server
FASHION_MCP_PORT = _CONFIG["fashion"]["mcp_port"]
CHECKPOINT_SCHEMA = _CONFIG["database"]["checkpoint_schema"]  # Neon schema for LangGraph checkpoints
LOG_LEVEL = _CONFIG["logging"]["level"]
TRACING_TAGS = _CONFIG["tracing"]["tags"]
A2A_HOST = _CONFIG["a2a"]["host"]  # interface the agent services bind to
A2A_TIMEOUT_S = _CONFIG["a2a"]["request_timeout_s"]  # supervisor-side timeout per agent call
A2A_HISTORY = _CONFIG["a2a"]["history_messages"]  # prior messages sent to an agent as context
A2A_WEBHOOK_PORT = _CONFIG["a2a"]["webhook_port"]  # port of the supervisor's push-notification receiver
SHIPMENT_CARRIER_DELAY_S = _CONFIG["shipment"]["carrier_delay_s"]  # simulated carrier lookup time
API_HOST = _CONFIG["api"]["host"]  # interface the FastAPI app binds to
API_PORT = _CONFIG["api"]["port"]
API_CORS_ORIGINS = _CONFIG["api"]["cors_origins"]
API_MAX_MESSAGE_CHARS = _CONFIG["api"]["max_message_chars"]
API_RATE_LIMIT_PER_MIN = _CONFIG["api"]["rate_limit_per_min"]
API_POOL_MAX_SIZE = _CONFIG["api"]["pool_max_size"]
RUNTIME_PORT = _CONFIG["runtime"]["port"]  # port of the AgentCore entrypoint
RUNTIME_POOL_MAX_SIZE = _CONFIG["runtime"]["pool_max_size"]
RUNTIME_MAX_MESSAGE_CHARS = _CONFIG["runtime"]["max_message_chars"]
A2A_PORTS = {name: spec["port"] for name, spec in _CONFIG["a2a"]["agents"].items()}  # agent name -> port
AUTH_AUDIENCE = _CONFIG["auth"]["audience"]  # expected token `aud`
AUTH_CUSTOMER_CLAIM = _CONFIG["auth"]["customer_claim"]  # namespaced claim holding the customer id
AUTH_JWKS_CACHE_TTL_S = _CONFIG["auth"]["jwks_cache_ttl_s"]  # in-memory signing-key cache lifetime
AWS_REGION = _CONFIG["aws"]["region"]  # region of the AgentCore Runtimes (used for SigV4 signing)

_TRUE_VALUES = {"1", "true", "yes", "on"}


def _flag(env_var: str, default: bool) -> bool:
    """Read a boolean switch, letting an environment variable override the config default.

    Read at call time because ``.env`` is loaded after this module is imported.

    Args:
        env_var (str): Name of the environment variable that overrides the default.
        default (bool): Value from the config file, used when the variable is unset or empty.

    Returns:
        bool: True when the effective value is one of ``1/true/yes/on`` (case-insensitive).
    """
    raw = os.environ.get(env_var)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in _TRUE_VALUES


def auth_enabled() -> bool:
    """Return whether user-token verification is on (``auth.enabled`` or env ``AUTH_ENABLED``).

    Returns:
        bool: True when tokens must be verified; False keeps the local unauthenticated behaviour.
    """
    return _flag("AUTH_ENABLED", _CONFIG["auth"]["enabled"])


def auth_issuer() -> str:
    """Return the token issuer (``iss``) derived from the ``AUTH0_TENANT`` environment variable.

    Read at call time because ``.env`` is loaded after this module is imported. Accepts a bare
    domain (``tenant.us.auth0.com``) or a full URL; the result always has the ``https://`` scheme
    and a trailing slash, which is how Auth0 writes ``iss`` in its tokens.

    Returns:
        str: Issuer URL such as ``https://tenant.us.auth0.com/``.

    Raises:
        RuntimeError: If ``AUTH0_TENANT`` is unset or empty.
    """
    tenant = os.environ.get("AUTH0_TENANT", "").strip()
    if not tenant:
        raise RuntimeError("AUTH0_TENANT is not set; it is required to verify user tokens")
    if "://" not in tenant:
        tenant = f"https://{tenant}"
    return tenant.rstrip("/") + "/"


def push_receiver_enabled() -> bool:
    """Return whether the supervisor runs the push-notification webhook receiver.

    Off means stream-only: nothing listens on the webhook port, no webhook is registered with agents,
    and a stream that breaks mid-task ends with the "try again" reply instead of a later push.

    Returns:
        bool: ``a2a.push_receiver_enabled``, overridable with env ``PUSH_RECEIVER_ENABLED``.
    """
    return _flag("PUSH_RECEIVER_ENABLED", _CONFIG["a2a"]["push_receiver_enabled"])


def sigv4_enabled() -> bool:
    """Return whether A2A requests are SigV4-signed (``aws.sigv4_enabled`` or env ``SIGV4_ENABLED``).

    Returns:
        bool: True when outgoing agent calls must be signed with IAM credentials.
    """
    return _flag("SIGV4_ENABLED", _CONFIG["aws"]["sigv4_enabled"])


def agent_port(agent: str) -> int:
    """Return the port an agent service listens on.

    Args:
        agent (str): Agent name, e.g. ``order_management``.

    Returns:
        int: Port from the ``a2a.agents`` config section.
    """
    return A2A_PORTS[agent]


def bind_address(agent: str) -> tuple[str, int]:
    """Return the ``(host, port)`` an agent service should listen on.

    Defaults to the local interface and the agent's configured port, so ``scripts/a2a/run_agents.py``
    is unchanged. AgentCore Runtime needs ``0.0.0.0:9000``, which a deployment sets with the
    ``A2A_BIND_HOST`` and ``A2A_BIND_PORT`` environment variables (read at call time, after ``.env`` is loaded).

    Args:
        agent (str): Agent name, e.g. ``order_management``.

    Returns:
        tuple[str, int]: Interface and port to pass to ``uvicorn.run``.

    Raises:
        ValueError: If ``A2A_BIND_PORT`` is set but is not an integer.
    """
    host = os.environ.get("A2A_BIND_HOST", "").strip() or A2A_HOST
    port = os.environ.get("A2A_BIND_PORT", "").strip()
    return host, int(port) if port else A2A_PORTS[agent]


def agent_url(agent: str) -> str:
    """Return the base URL of an agent service.

    The env var ``<AGENT>_AGENT_URL`` (e.g. ``ORDER_MANAGEMENT_AGENT_URL``) overrides the
    config-derived local URL, so a deployment can point the supervisor at remote hosts. It is read
    at call time because ``.env`` is loaded after this module is imported. The same value is the
    address an agent's own Agent Card advertises, so a deployed agent must have it set to its public
    address (the bind address from ``bind_address`` is only where it listens).

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
    "RAG_EMBEDDING",
    "RAG_TOP_K",
    "RAG_MAX_RETRIES",
    "RAG_MIN_SCORE",
    "RAG_CONDENSE_HISTORY",
    "SUPERVISOR_HISTORY",
    "SUPERVISOR_MAX_AGENTS",
    "SUPERVISOR_STICKY_MAX_WORDS",
    "FASHION_COLLECTION",
    "FASHION_EMBEDDING",
    "FASHION_TOP_K",
    "FASHION_MIN_SCORE",
    "FASHION_MCP_HOST",
    "FASHION_MCP_PORT",
    "CHECKPOINT_SCHEMA",
    "LOG_LEVEL",
    "TRACING_TAGS",
    "A2A_HOST",
    "A2A_TIMEOUT_S",
    "A2A_HISTORY",
    "A2A_WEBHOOK_PORT",
    "SHIPMENT_CARRIER_DELAY_S",
    "API_HOST",
    "API_PORT",
    "API_CORS_ORIGINS",
    "API_MAX_MESSAGE_CHARS",
    "API_RATE_LIMIT_PER_MIN",
    "API_POOL_MAX_SIZE",
    "RUNTIME_PORT",
    "RUNTIME_POOL_MAX_SIZE",
    "RUNTIME_MAX_MESSAGE_CHARS",
    "A2A_PORTS",
    "AUTH_AUDIENCE",
    "AUTH_CUSTOMER_CLAIM",
    "AUTH_JWKS_CACHE_TTL_S",
    "AWS_REGION",
    "auth_enabled",
    "auth_issuer",
    "sigv4_enabled",
    "push_receiver_enabled",
    "agent_port",
    "bind_address",
    "agent_url",
]
