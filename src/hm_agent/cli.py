"""Command-line entry point for the supervisor multi-agent system.

Run (from the repo root):
    uv run python -m hm_agent "What was the last item I bought?"
    uv run python -m hm_agent --customer-id cust002 "How many purchases have I made?"
    uv run python -m hm_agent                      # interactive chat, many turns in one thread

Omit the query to start a chat loop: every message is sent on the same thread, so the
assistant remembers earlier turns. Type ``exit`` (or Ctrl+C) to quit.

Conversation state is checkpointed to Neon (schema ``checkpoints``); pass ``--thread-id``
with the id printed by a previous run to continue that conversation later.

Needs GROQ_API_KEY and NEON_POSTGRES_CONNECTION_STRING in ``.env``. Runs are traced to
Langfuse (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL); one session per thread.
"""

import argparse
import logging
import sys
import uuid

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.graph.state import CompiledStateGraph

from hm_agent.core.db import neon_checkpointer
from hm_agent.graph import build_graph
from hm_agent.core.logging_setup import configure_logging
from hm_agent.core.settings import DEFAULT_CUSTOMER_ID, REPO_ROOT, TRACING_TAGS

logger = logging.getLogger(__name__)


EXIT_COMMANDS = {"exit", "quit", ":q"}  # typed at the prompt to end a chat session


def _build_config(customer_id: str, thread_id: str) -> RunnableConfig:
    """Build the per-turn run config (thread, customer and Langfuse tracing metadata).

    A fresh ``CallbackHandler`` is created per turn so each turn is its own trace, while the
    shared ``langfuse_session_id`` groups all turns of the thread into one Langfuse session.

    Args:
        customer_id (str): Customer the conversation belongs to, e.g. ``cust001``.
        thread_id (str): Conversation thread id used by the checkpointer and Langfuse.

    Returns:
        RunnableConfig: Config to pass to ``graph.invoke``.
    """
    return {
        "configurable": {"customer_id": customer_id, "thread_id": thread_id},
        # The handler traces every graph node, LLM call and tool call of this turn.
        "callbacks": [CallbackHandler()],
        "run_name": "hm-supervisor",
        "metadata": {
            "langfuse_session_id": thread_id,
            "langfuse_user_id": customer_id,
            "langfuse_tags": TRACING_TAGS,
        },
    }


def _run_turn(graph: CompiledStateGraph, query: str, customer_id: str, thread_id: str) -> str:
    """Send one user message to the graph and return the assistant's reply.

    Args:
        graph (CompiledStateGraph): Compiled supervisor graph with a checkpointer attached.
        query (str): The user's message for this turn.
        customer_id (str): Customer the conversation belongs to.
        thread_id (str): Conversation thread id; earlier turns are loaded from the checkpointer.

    Returns:
        str: Text of the last message in the resulting state.
    """
    result = graph.invoke({"messages": [HumanMessage(query)]}, _build_config(customer_id, thread_id))
    last: AnyMessage = result["messages"][-1]
    return last.text


def main() -> None:
    """CLI entry point: one query when given, otherwise a continuous chat in one thread."""
    load_dotenv(REPO_ROOT / ".env")
    configure_logging("cli")
    # Model replies can contain non-cp1252 characters (e.g. non-breaking hyphens) on Windows consoles.
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="H&M supervisor multi-agent system")
    parser.add_argument("query", nargs="?", default=None, help="User message; omit to start an interactive chat")
    parser.add_argument("--customer-id", default=DEFAULT_CUSTOMER_ID, help="Customer id, e.g. cust001")
    parser.add_argument("--thread-id", default=None, help="Resume an existing conversation thread")
    args = parser.parse_args()

    # Created after load_dotenv so the Langfuse client sees the credentials.
    langfuse = get_client()
    thread_id = args.thread_id or str(uuid.uuid4())
    logger.info("thread_id=%s resumed=%s", thread_id, args.thread_id is not None)
    try:
        # One checkpointer connection and compiled graph serve every turn of the session.
        with neon_checkpointer() as checkpointer:
            graph = build_graph(checkpointer)
            if args.query is not None:
                print(_run_turn(graph, args.query, args.customer_id, thread_id))
            else:
                print(f"Chat started (thread_id: {thread_id}). Type 'exit' to quit.")
                while True:
                    try:
                        query = input("\nyou> ").strip()
                    except (EOFError, KeyboardInterrupt):
                        break
                    if query.lower() in EXIT_COMMANDS:
                        break
                    if query:
                        print(f"\nassistant> {_run_turn(graph, query, args.customer_id, thread_id)}")
    finally:
        # Short-lived script: send queued trace events before the process exits.
        langfuse.flush()
    print(f"thread_id: {thread_id}")
