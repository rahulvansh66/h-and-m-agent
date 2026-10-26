"""Command-line entry point for the supervisor multi-agent system.

Run (from the repo root):
    uv run python -m hm_agent "What was the last item I bought?"
    uv run python -m hm_agent --customer-id cust002 "How many purchases have I made?"
    uv run python -m hm_agent                      # interactive chat, many turns in one thread

Omit the query to start a chat loop: every message is sent on the same thread, so the
assistant remembers earlier turns. Type ``exit`` (or Ctrl+C) to quit.

Conversation state is checkpointed to Neon (schema ``checkpoints``); pass ``--thread-id``
with the id printed by a previous run to continue that conversation later.

Replies stream: agent progress lines and (for agents that stream tokens) the answer text are printed
as they arrive. If an agent's stream breaks, its A2A push notification webhook is the fallback: the CLI
starts a small receiver, the first reply is an acknowledgement, and the real answer is printed (and
saved to the thread) when the agent calls the webhook. One-shot runs wait for it before exiting.

Needs GROQ_API_KEY and NEON_POSTGRES_CONNECTION_STRING in ``.env``. Runs are traced to
Langfuse (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL); one session per thread.
"""

import argparse
import logging
import sys
import threading
import uuid
from collections.abc import Callable

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langfuse import get_client
from langgraph.graph.state import CompiledStateGraph

from hm_agent.a2a.push_receiver import PendingTask, PushReceiver, set_receiver
from hm_agent.core.db import neon_checkpointer
from hm_agent.graph import build_graph
from hm_agent.turns import stream_turn
from hm_agent.core.logging_setup import configure_logging
from hm_agent.core.settings import A2A_WEBHOOK_PORT, DEFAULT_CUSTOMER_ID, REPO_ROOT, push_receiver_enabled

logger = logging.getLogger(__name__)


EXIT_COMMANDS = {"exit", "quit", ":q"}  # typed at the prompt to end a chat session
PUSH_WAIT_S = 120  # how long a one-shot run waits for a pushed result before exiting

# Held while a turn runs and while a pushed result is saved. Both write checkpoints to the same
# thread; without the lock a result that arrives mid-turn is overwritten by the turn's checkpoint.
_THREAD_LOCK = threading.Lock()


def _run_turn(graph: CompiledStateGraph, query: str, customer_id: str, thread_id: str) -> None:
    """Send one user message to the graph and print the answer as it streams in.

    Progress lines from the agent are printed as they arrive. If the agent streams its reply, the
    pieces are printed as they come; otherwise the full reply is printed once the turn ends.

    Args:
        graph (CompiledStateGraph): Compiled supervisor graph with a checkpointer attached.
        query (str): The user's message for this turn.
        customer_id (str): Customer the conversation belongs to.
        thread_id (str): Conversation thread id; earlier turns are loaded from the checkpointer.
    """
    streamed = False  # True once reply pieces were printed, so the final reply is not printed again
    with _THREAD_LOCK:
        for event in stream_turn(graph, query, customer_id, thread_id):
            if event["type"] == "progress":
                print(f"  ... {event['text']}", flush=True)
            elif event["type"] == "token":
                print(event["text"] if streamed else f"\nassistant> {event['text']}", end="", flush=True)
                streamed = True
            elif event["type"] == "reply":
                print("" if streamed else f"\nassistant> {event['text']}", flush=True)


def _make_deliver(graph: CompiledStateGraph, interactive: bool) -> Callable[[PendingTask, str], None]:
    """Create the callback that shows a pushed result and saves it into the conversation thread.

    Saving the message means the next turn (and the supervisor's routing) sees the answer as if
    the agent had replied normally.

    Args:
        graph (CompiledStateGraph): Compiled supervisor graph with a checkpointer attached.
        interactive (bool): True in chat mode, where the prompt is shown again after the message.

    Returns:
        Callable[[PendingTask, str], None]: Function ``(task, text) -> None`` run on a worker thread.
    """

    def deliver(task: PendingTask, text: str) -> None:
        """Print the result at once, then add it to the thread once no turn is running.

        Args:
            task (PendingTask): The finished background task.
            text (str): Result text to show the customer.
        """
        print(f"\nassistant ({task.agent}, update)> {text}", flush=True)
        if interactive:
            print("\nyou> ", end="", flush=True)
        config: RunnableConfig = {"configurable": {"thread_id": task.thread_id, "customer_id": task.customer_id}}
        # Wait for a running turn to finish so its checkpoint does not overwrite this message.
        with _THREAD_LOCK:
            # as_node makes the message look like it came from the agent's node, so no other node runs.
            graph.update_state(config, {"messages": [AIMessage(text, name=task.agent)]}, as_node=task.agent)

    return deliver


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
    receiver: PushReceiver | None = None
    logger.info("thread_id=%s resumed=%s", thread_id, args.thread_id is not None)
    try:
        # One checkpointer connection and compiled graph serve every turn of the session.
        with neon_checkpointer() as checkpointer:
            graph = build_graph(checkpointer)
            interactive = args.query is None
            # Receives push notifications from agents with slow tasks and delivers their results.
            if push_receiver_enabled():
                receiver = PushReceiver(A2A_WEBHOOK_PORT, _make_deliver(graph, interactive))
                receiver.start()
                set_receiver(receiver)
            if not interactive:
                _run_turn(graph, args.query, args.customer_id, thread_id)
                if receiver is not None and not receiver.wait_idle(PUSH_WAIT_S):
                    logger.warning("component=cli status=push_timeout wait_s=%d", PUSH_WAIT_S)
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
                        _run_turn(graph, query, args.customer_id, thread_id)
    finally:
        if receiver is not None:
            set_receiver(None)
            receiver.stop()
        # Short-lived script: send queued trace events before the process exits.
        langfuse.flush()
    print(f"thread_id: {thread_id}")
