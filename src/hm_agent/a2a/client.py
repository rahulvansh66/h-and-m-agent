"""A2A client: sends one customer message to a remote agent and returns its text reply.

Role in the system: used by the supervisor graph (via ``remote_node``) to reach an agent service.
It resolves the Agent Card, sends a blocking ``SendMessage`` with the latest text, the recent
history and the customer/thread ids, and extracts the reply from the task's artifact.
The optional wire log (``A2A_WIRE_LOG``) lives in ``hm_agent.a2a.wire_log``.
"""

import asyncio
import logging
import time

import httpx
from a2a.client import ClientConfig, ClientFactory
from a2a.helpers.proto_helpers import get_artifact_text, new_data_part, new_text_part
from a2a.types.a2a_pb2 import Message, Role, SendMessageRequest, TaskState

from hm_agent.a2a.wire_log import wire_log_hooks

logger = logging.getLogger(__name__)


async def _send(url: str, text: str, history: list[dict], customer_id: str, thread_id: str, timeout_s: float) -> str:
    """Send one message to an agent and return the reply text (async implementation).

    Args:
        url (str): Base URL of the agent, e.g. ``http://127.0.0.1:8001``.
        text (str): Latest customer message.
        history (list[dict]): Prior messages as ``{"role", "text", "name"}`` dicts, oldest first.
        customer_id (str): Customer the conversation belongs to.
        thread_id (str): Conversation thread id; used as the A2A ``contextId``.
        timeout_s (float): HTTP timeout in seconds for each call.

    Returns:
        str: Text of the task's reply artifact.

    Raises:
        RuntimeError: If the task ends in a state other than completed.
        httpx.HTTPError: If the agent cannot be reached or times out.
    """
    async with httpx.AsyncClient(timeout=timeout_s, event_hooks=wire_log_hooks()) as http:
        factory = ClientFactory(ClientConfig(streaming=False, httpx_client=http))
        client = await factory.create_from_url(url)
        request = SendMessageRequest(
            message=Message(
                message_id=f"msg-{time.time_ns()}",
                context_id=thread_id,
                role=Role.ROLE_USER,
                parts=[new_text_part(text), new_data_part({"history": history})],
            ),
            metadata={"customer_id": customer_id, "thread_id": thread_id},
        )
        task = None
        async for response in client.send_message(request):
            if response.HasField("task"):
                task = response.task
        await client.close()
    if task is None:
        raise RuntimeError("agent returned no task")
    state = TaskState.Name(task.status.state)
    if task.status.state != TaskState.TASK_STATE_COMPLETED:
        raise RuntimeError(f"task {task.id} ended in state {state}")
    logger.info("a2a_call status=complete url=%s task_id=%s context_id=%s state=%s", url, task.id, task.context_id, state)
    return "\n".join(get_artifact_text(a) for a in task.artifacts)


def call_remote_agent(
    url: str, text: str, history: list[dict], customer_id: str, thread_id: str, timeout_s: float
) -> str:
    """Send one message to a remote A2A agent and return its reply (blocking).

    Args:
        url (str): Base URL of the agent, e.g. ``http://127.0.0.1:8001``.
        text (str): Latest customer message.
        history (list[dict]): Prior messages as ``{"role", "text", "name"}`` dicts, oldest first.
        customer_id (str): Customer the conversation belongs to.
        thread_id (str): Conversation thread id; used as the A2A ``contextId``.
        timeout_s (float): HTTP timeout in seconds for each call.

    Returns:
        str: The agent's reply text.

    Raises:
        RuntimeError: If the task does not complete.
        httpx.HTTPError: If the agent cannot be reached or times out.
    """
    return asyncio.run(_send(url, text, history, customer_id, thread_id, timeout_s))
