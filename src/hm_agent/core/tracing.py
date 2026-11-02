"""Langfuse run config shared by every graph entry point.

Role in the system: one place that builds the ``RunnableConfig`` for a traced run (callback handler,
session / user / tags, release and environment), so the supervisor turn, the per-agent A2A server and
the CLI push path all trace the same way. It also carries the Langfuse trace id: the supervisor creates
it up front, sends it to each agent in the A2A request metadata, and the agent service attaches its spans
to that same trace, so one customer turn is one trace across supervisor, agents and tools.
"""

import logging
import os
from contextvars import ContextVar

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from hm_agent.core.config import load_config
from hm_agent.core.logging_setup import git_commit
from hm_agent.core.settings import PROMPT_LABEL, TRACING_TAGS

logger = logging.getLogger(__name__)

# Key of the trace id inside ``config["configurable"]`` (read by the remote-agent node) and inside the A2A
# request metadata (read by the agent service).
TRACE_ID_CONFIG_KEY = "langfuse_trace_id"
TRACE_ID_METADATA_KEY = "langfuse_trace_id"

# Trace id the agent service received for the request being handled; the runner reads it to join the trace.
incoming_trace_id: ContextVar[str] = ContextVar("incoming_trace_id", default="")

# The Langfuse client reads these once, when it is created: every trace is labelled with the git commit
# (release) and the APP_ENV (environment), so dashboards can be split per release and environment.
os.environ.setdefault("LANGFUSE_RELEASE", git_commit())
os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", str(load_config()["environment"]).lower())


def new_trace_id() -> str:
    """Create a fresh Langfuse trace id (32 lowercase hex characters).

    Returns:
        str: A trace id usable as ``trace_context`` for a ``CallbackHandler`` and for scoring.
    """
    return Langfuse.create_trace_id()


def build_langfuse_config(
    customer_id: str,
    thread_id: str,
    run_name: str,
    *,
    extra_tags: list[str] | None = None,
    trace_id: str | None = None,
    extra_configurable: dict | None = None,
) -> RunnableConfig:
    """Build the run config for one traced graph run.

    A fresh ``CallbackHandler`` per run makes each run its own trace, while the shared ``thread_id``
    (``langfuse_session_id``) groups all turns of a conversation into one Langfuse session.

    Args:
        customer_id (str): Customer the run belongs to (also the Langfuse user id).
        thread_id (str): Conversation thread id (also the Langfuse session id).
        run_name (str): Name of the root observation, e.g. ``hm-supervisor`` or ``a2a-order_management``.
        extra_tags (list[str] | None): Tags added to the configured ``tracing.tags``.
        trace_id (str | None): Existing trace to join (from the supervisor); a new trace is created when None.
        extra_configurable (dict | None): Extra ``configurable`` entries, e.g. the user token.

    Returns:
        RunnableConfig: Config to pass to ``graph.stream`` / ``graph.invoke``. ``configurable`` also holds the
        trace id under ``TRACE_ID_CONFIG_KEY`` when one is set.
    """
    configurable = {"customer_id": customer_id, "thread_id": thread_id, **(extra_configurable or {})}
    handler = CallbackHandler(trace_context={"trace_id": trace_id}) if trace_id else CallbackHandler()
    if trace_id:
        configurable[TRACE_ID_CONFIG_KEY] = trace_id
    return {
        "configurable": configurable,
        # The handler traces every graph node, LLM call and tool call of this run; the usage handler totals
        # tokens per model so the run can be scored for tokens and cost (see hm_agent.evals.usage).
        "callbacks": [handler, UsageMetadataCallbackHandler()],
        "run_name": run_name,
        "metadata": {
            "langfuse_session_id": thread_id,
            "langfuse_user_id": customer_id,
            "langfuse_tags": [*TRACING_TAGS, *(extra_tags or []), f"commit:{git_commit()}", f"prompts:{PROMPT_LABEL}"],
        },
    }
