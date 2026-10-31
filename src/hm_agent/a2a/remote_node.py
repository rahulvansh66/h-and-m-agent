"""Supervisor-graph node that delegates the turn to a remote A2A agent.

Role in the system: replaces the in-process agent subgraphs. The node sends the latest customer
message (plus a short history window) to the agent service and appends the reply as an AI message
named after the agent, which is the label the supervisor uses to keep follow-ups with that agent.
The call is streamed: progress lines and answer pieces are forwarded to the graph's ``custom`` stream
as they arrive (so a caller using ``graph.stream`` can show them live) and the full reply is still
returned as one message. For agents that also support push notifications the webhook is registered
as a fallback: if the stream breaks before the task ends, the node returns an acknowledgement and the
real answer is added to the thread later by the push receiver. When the supervisor split the message
between several agents, the node instead returns its answer as an ``AgentResult`` (and streams no answer
pieces) so the ``synthesize`` node can merge them into one reply.
"""

import logging
import time
from collections.abc import Callable

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from hm_agent.a2a.client import stream_remote_agent
from hm_agent.a2a.push_receiver import COMPLETED_STATE, PendingTask, get_receiver
from hm_agent.auth import USER_TOKEN_CONFIG_KEY
from hm_agent.core.settings import A2A_HISTORY, A2A_TIMEOUT_S, agent_url
from hm_agent.core.state import AgentResult, AgentState

logger = logging.getLogger(__name__)

UNAVAILABLE_REPLY = "Sorry, I can't answer that right now. Please try again in a moment."
PENDING_REPLY = "I'm checking with the carrier now. This can take a little while; I'll post the result here as soon as it arrives."


def make_remote_agent_node(agent: str) -> Callable[[AgentState, RunnableConfig], dict]:
    """Create a graph node that calls the A2A agent ``agent`` over a streaming connection.

    Args:
        agent (str): Agent name; also the AI message name and the key of its URL setting.

    Returns:
        Callable[[AgentState, RunnableConfig], dict]: Node reading ``messages`` and writing one AI reply.
    """

    def node(state: AgentState, config: RunnableConfig) -> dict:
        """Call the remote agent for the latest customer message and stream its output.

        Custom stream events (``{"type": "started" | "progress" | "token", "agent", ...}``) are
        written for each event the agent sends. If the agent is unreachable or its task fails, a
        generic apology is returned instead of raising, so one broken agent does not crash the
        conversation. If the stream breaks while a push-registered task is still running, the task
        is handed to the push receiver and an acknowledgement is returned.

        Reads: ``messages`` and, on a split turn, ``query`` (this agent's sub-question, sent instead
        of the customer's message). Writes: ``messages`` (one AI reply), or on a split turn ``results``
        (one ``AgentResult``, no token events) for ``synthesize`` to merge.

        Args:
            state (AgentState): Current supervisor state, or the payload ``route_to_agent`` sent.
            config (RunnableConfig): Run config carrying ``customer_id``, ``thread_id`` and, when auth is on, the user token.

        Returns:
            dict: ``{"messages": [AIMessage]}`` named after the agent, or ``{"results": [AgentResult]}`` on a split turn.
        """
        configurable = config.get("configurable", {})
        customer_id, thread_id = configurable.get("customer_id", ""), configurable.get("thread_id", "")
        user_token = configurable.get(USER_TOKEN_CONFIG_KEY, "")  # lets the agent verify the user; never logged
        messages = state["messages"]
        sub_query = state.get("query")  # set only when the message was split between several agents
        latest = sub_query or messages[-1].text
        # Prior turns only; the latest message travels separately as the text part.
        history = [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant", "text": m.text, "name": m.name}
            for m in messages[:-1][-A2A_HISTORY:]
        ]
        writer = get_stream_writer()
        receiver = get_receiver()
        url = agent_url(agent)
        task_id, push_registered, terminal, pieces = "", False, "", []
        start = time.perf_counter()
        try:
            events = stream_remote_agent(
                url,
                latest,
                history,
                customer_id,
                thread_id,
                A2A_TIMEOUT_S,
                push=receiver.target if receiver else None,
                user_token=user_token,
            )
            for event in events:
                if event.kind == "started":
                    task_id, push_registered = event.task_id, event.push_registered
                    writer({"type": "started", "agent": agent, "task_id": task_id})
                elif event.kind == "progress":
                    writer({"type": "progress", "agent": agent, "text": event.text})
                elif event.kind == "answer":
                    pieces.append(event.text)
                    if not sub_query:  # a split turn streams only the merged answer
                        writer({"type": "token", "agent": agent, "text": event.text})
                else:
                    terminal = event.state
            if terminal != COMPLETED_STATE:
                raise RuntimeError(f"task {task_id} ended in state {terminal or 'unknown'}")
            reply, status = "".join(pieces), "complete"
        except Exception:
            if task_id and push_registered and not terminal and receiver is not None:
                # Stream lost but the task may still be running: the webhook will report its result.
                logger.warning("node=%s status=stream_lost thread_id=%s task_id=%s fallback=push", agent, thread_id, task_id)
                receiver.register(PendingTask(task_id, agent, url, customer_id, thread_id))
                reply, status = PENDING_REPLY, "pending"
            else:
                logger.exception("node=%s status=remote_failed thread_id=%s", agent, thread_id)
                reply, status = UNAVAILABLE_REPLY, "failed"
        if task_id and push_registered and terminal and receiver is not None:
            # The result came through the stream; ignore the webhook notifications that follow.
            receiver.mark_streamed(task_id)
        logger.info("node=%s status=%s thread_id=%s duration_s=%.2f", agent, status, thread_id, time.perf_counter() - start)
        if sub_query:
            # Split turn: hand the answer to ``synthesize`` instead of adding it to the conversation.
            return {"results": [AgentResult(agent=agent, query=sub_query, status=status, text=reply)]}
        return {"messages": [AIMessage(reply, name=agent)]}

    return node
