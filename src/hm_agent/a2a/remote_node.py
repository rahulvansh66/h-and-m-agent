"""Supervisor-graph node that delegates the turn to a remote A2A agent.

Role in the system: replaces the in-process agent subgraphs. The node sends the latest customer
message (plus a short history window) to the agent service and appends the reply as an AI message
named after the agent, which is the label the supervisor uses to keep follow-ups with that agent.
"""

import logging
import time
from collections.abc import Callable

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from hm_agent.a2a.client import call_remote_agent
from hm_agent.core.settings import A2A_HISTORY, A2A_TIMEOUT_S, agent_url
from hm_agent.core.state import AgentState

logger = logging.getLogger(__name__)

UNAVAILABLE_REPLY = "Sorry, I can't answer that right now. Please try again in a moment."


def make_remote_agent_node(agent: str) -> Callable[[AgentState, RunnableConfig], dict]:
    """Create a graph node that calls the A2A agent ``agent``.

    Args:
        agent (str): Agent name; also the AI message name and the key of its URL setting.

    Returns:
        Callable[[AgentState, RunnableConfig], dict]: Node reading ``messages`` and writing one AI message.
    """

    def node(state: AgentState, config: RunnableConfig) -> dict:
        """Call the remote agent for the latest customer message.

        If the agent is unreachable or its task fails, a generic apology is returned instead of
        raising, so one broken agent does not crash the conversation.

        Reads: ``messages``. Writes: ``messages`` (one AI reply).

        Args:
            state (AgentState): Current supervisor state.
            config (RunnableConfig): Run config carrying ``customer_id`` and ``thread_id``.

        Returns:
            dict: ``{"messages": [AIMessage]}`` named after the agent.
        """
        configurable = config.get("configurable", {})
        customer_id, thread_id = configurable.get("customer_id", ""), configurable.get("thread_id", "")
        messages = state["messages"]
        latest = messages[-1].text
        # Prior turns only; the latest message travels separately as the text part.
        history = [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant", "text": m.text, "name": m.name}
            for m in messages[:-1][-A2A_HISTORY:]
        ]
        start = time.perf_counter()
        try:
            reply = call_remote_agent(agent_url(agent), latest, history, customer_id, thread_id, A2A_TIMEOUT_S)
            logger.info("node=%s status=complete thread_id=%s duration_s=%.2f", agent, thread_id, time.perf_counter() - start)
        except Exception:
            logger.exception("node=%s status=remote_failed thread_id=%s duration_s=%.2f", agent, thread_id, time.perf_counter() - start)
            reply = UNAVAILABLE_REPLY
        return {"messages": [AIMessage(reply, name=agent)]}

    return node
