"""A2A client: sends one customer message to a remote agent and returns its reply or a pending task.

Role in the system: used by the supervisor graph (via ``remote_node``) to reach an agent service.
It resolves the Agent Card and sends ``SendMessage`` with the latest text, the recent history and
the customer/thread ids. If the card advertises push notifications and the caller offered a
webhook, the task is started in the background (``return_immediately``) and the result arrives
later at the webhook; otherwise the call blocks and the reply is read from the task's artifact.
``stream_remote_agent`` is the streaming variant: it keeps the connection open and yields the agent's
progress lines and answer pieces as they arrive (and still registers the webhook for agents that
support push, as a fallback if the connection drops).
The optional wire log (``A2A_WIRE_LOG``) lives in ``hm_agent.a2a.wire_log``.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Literal

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.helpers.proto_helpers import get_artifact_text, get_message_text, new_data_part, new_text_part
from a2a.types.a2a_pb2 import (
    GetTaskRequest,
    Message,
    Role,
    SendMessageConfiguration,
    SendMessageRequest,
    Task,
    TaskPushNotificationConfig,
    TaskState,
)

from hm_agent.a2a.wire_log import wire_log_hooks

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PushTarget:
    """Webhook the agent should call when a background task changes state.

    Attributes:
        url: Address the agent POSTs the task updates to.
        token: Shared secret the agent sends back in ``X-A2A-Notification-Token``.
    """

    url: str
    token: str


@dataclass(frozen=True)
class RemoteResult:
    """Outcome of one call to a remote agent.

    Attributes:
        task_id: Id of the A2A task the agent created.
        reply: Reply text when the call blocked until completion; None when the task is pending.
        pending: True when the task runs in the background and the result will be pushed.
    """

    task_id: str
    reply: str | None = None
    pending: bool = False


def task_reply(task: Task) -> str:
    """Join the text of all artifacts of a task.

    A streamed reply is one artifact whose parts are the appended chunks, in order, so the parts of
    one artifact are joined without a separator (they are pieces of one text) and separate
    artifacts with a newline.

    Args:
        task (Task): A2A task, normally in the completed state.

    Returns:
        str: The reply text (empty if there are no artifacts).
    """
    return "\n".join(get_artifact_text(a, delimiter="") for a in task.artifacts)


async def _send(
    url: str,
    text: str,
    history: list[dict],
    customer_id: str,
    thread_id: str,
    timeout_s: float,
    push: PushTarget | None,
) -> RemoteResult:
    """Send one message to an agent (async implementation).

    Args:
        url (str): Base URL of the agent, e.g. ``http://127.0.0.1:8001``.
        text (str): Latest customer message.
        history (list[dict]): Prior messages as ``{"role", "text", "name"}`` dicts, oldest first.
        customer_id (str): Customer the conversation belongs to.
        thread_id (str): Conversation thread id; used as the A2A ``contextId``.
        timeout_s (float): HTTP timeout in seconds for each call.
        push (PushTarget | None): Webhook to register. Used only if the agent's card advertises
            push notifications; otherwise the call blocks.

    Returns:
        RemoteResult: The reply (blocking call) or the pending task id (push call).

    Raises:
        RuntimeError: If the agent returns no task, or a blocking task ends in a state other than completed.
        httpx.HTTPError: If the agent cannot be reached or times out.
    """
    async with httpx.AsyncClient(timeout=timeout_s, event_hooks=wire_log_hooks()) as http:
        card = await A2ACardResolver(http, url).get_agent_card()
        client = ClientFactory(ClientConfig(streaming=False, httpx_client=http)).create(card)
        use_push = push is not None and card.capabilities.push_notifications
        configuration = SendMessageConfiguration()
        if use_push:
            # Reply with the task right away; the agent keeps working and POSTs updates to the webhook.
            configuration = SendMessageConfiguration(
                return_immediately=True,
                task_push_notification_config=TaskPushNotificationConfig(url=push.url, token=push.token),
            )
        request = SendMessageRequest(
            message=Message(
                message_id=f"msg-{time.time_ns()}",
                context_id=thread_id,
                role=Role.ROLE_USER,
                parts=[new_text_part(text), new_data_part({"history": history})],
            ),
            configuration=configuration,
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
    if use_push:
        logger.info("a2a_call status=pending url=%s task_id=%s context_id=%s state=%s", url, task.id, task.context_id, state)
        return RemoteResult(task_id=task.id, pending=True)
    if task.status.state != TaskState.TASK_STATE_COMPLETED:
        raise RuntimeError(f"task {task.id} ended in state {state}")
    logger.info("a2a_call status=complete url=%s task_id=%s context_id=%s state=%s", url, task.id, task.context_id, state)
    return RemoteResult(task_id=task.id, reply=task_reply(task))


def call_remote_agent(
    url: str,
    text: str,
    history: list[dict],
    customer_id: str,
    thread_id: str,
    timeout_s: float,
    push: PushTarget | None = None,
) -> RemoteResult:
    """Send one message to a remote A2A agent (blocking until the agent answers the request).

    Args:
        url (str): Base URL of the agent, e.g. ``http://127.0.0.1:8001``.
        text (str): Latest customer message.
        history (list[dict]): Prior messages as ``{"role", "text", "name"}`` dicts, oldest first.
        customer_id (str): Customer the conversation belongs to.
        thread_id (str): Conversation thread id; used as the A2A ``contextId``.
        timeout_s (float): HTTP timeout in seconds for each call.
        push (PushTarget | None): Webhook for push notifications; ignored by agents that do not
            advertise the capability.

    Returns:
        RemoteResult: ``reply`` set for a normal call, ``pending=True`` when the result will be pushed.

    Raises:
        RuntimeError: If the task does not complete (blocking call) or no task comes back.
        httpx.HTTPError: If the agent cannot be reached or times out.
    """
    return asyncio.run(_send(url, text, history, customer_id, thread_id, timeout_s, push))


async def fetch_task(url: str, task_id: str, timeout_s: float) -> Task:
    """Fetch the current state of a task from an agent (``GetTask``).

    Used after a push notification: the notification says the task changed, this reads the result.

    Args:
        url (str): Base URL of the agent that owns the task.
        task_id (str): Id of the task.
        timeout_s (float): HTTP timeout in seconds.

    Returns:
        Task: The task with its status and artifacts.

    Raises:
        httpx.HTTPError: If the agent cannot be reached or times out.
    """
    async with httpx.AsyncClient(timeout=timeout_s, event_hooks=wire_log_hooks()) as http:
        card = await A2ACardResolver(http, url).get_agent_card()
        client = ClientFactory(ClientConfig(streaming=False, httpx_client=http)).create(card)
        task = await client.get_task(GetTaskRequest(id=task_id))
        await client.close()
    return task


# Task states after which a task sends no further events.
_TERMINAL_STATES = frozenset(
    {
        TaskState.TASK_STATE_COMPLETED,
        TaskState.TASK_STATE_FAILED,
        TaskState.TASK_STATE_CANCELED,
        TaskState.TASK_STATE_REJECTED,
    }
)

EventKind = Literal["started", "progress", "answer", "state"]


@dataclass(frozen=True)
class RemoteEvent:
    """One event of a streamed remote call, in the order the agent produced it.

    Attributes:
        kind: ``started`` (task created), ``progress`` (status line), ``answer`` (piece of the reply)
            or ``state`` (the task reached a terminal state).
        text: Progress line or answer piece; empty for the other kinds.
        task_id: Id of the A2A task (set on ``started``).
        state: Terminal task state name such as ``TASK_STATE_COMPLETED`` (set on ``state``).
        push_registered: On ``started``, True when a push webhook was registered for the task.
    """

    kind: EventKind
    text: str = ""
    task_id: str = ""
    state: str = ""
    push_registered: bool = False


async def _stream(
    url: str,
    text: str,
    history: list[dict],
    customer_id: str,
    thread_id: str,
    timeout_s: float,
    push: PushTarget | None,
) -> AsyncIterator[RemoteEvent]:
    """Send one message with streaming and yield the agent's events (async implementation).

    Args:
        url (str): Base URL of the agent, e.g. ``http://127.0.0.1:8003``.
        text (str): Latest customer message.
        history (list[dict]): Prior messages as ``{"role", "text", "name"}`` dicts, oldest first.
        customer_id (str): Customer the conversation belongs to.
        thread_id (str): Conversation thread id; used as the A2A ``contextId``.
        timeout_s (float): HTTP timeout in seconds (also the longest silence tolerated on the stream).
        push (PushTarget | None): Webhook to register as a fallback; used only if the card advertises push.

    Yields:
        RemoteEvent: ``started`` first, then ``progress`` / ``answer`` events and a final ``state``.

    Raises:
        httpx.HTTPError: If the agent cannot be reached or the stream breaks.
    """
    async with httpx.AsyncClient(timeout=timeout_s, event_hooks=wire_log_hooks()) as http:
        card = await A2ACardResolver(http, url).get_agent_card()
        client = ClientFactory(ClientConfig(streaming=card.capabilities.streaming, httpx_client=http)).create(card)
        use_push = push is not None and card.capabilities.push_notifications
        # Unlike the push-only call there is no ``return_immediately``: the stream stays open while the
        # webhook is registered as a second, independent route for the same updates.
        configuration = SendMessageConfiguration()
        if use_push:
            configuration = SendMessageConfiguration(
                task_push_notification_config=TaskPushNotificationConfig(url=push.url, token=push.token)
            )
        request = SendMessageRequest(
            message=Message(
                message_id=f"msg-{time.time_ns()}",
                context_id=thread_id,
                role=Role.ROLE_USER,
                parts=[new_text_part(text), new_data_part({"history": history})],
            ),
            configuration=configuration,
            metadata={"customer_id": customer_id, "thread_id": thread_id},
        )
        async for response in client.send_message(request):
            if response.HasField("task"):
                yield RemoteEvent("started", task_id=response.task.id, push_registered=use_push)
            elif response.HasField("status_update"):
                status = response.status_update.status
                if status.state == TaskState.TASK_STATE_WORKING and status.message.parts:
                    yield RemoteEvent("progress", text=get_message_text(status.message))
                elif status.state in _TERMINAL_STATES:
                    yield RemoteEvent("state", state=TaskState.Name(status.state))
            elif response.HasField("artifact_update"):
                yield RemoteEvent("answer", text=get_artifact_text(response.artifact_update.artifact, delimiter=""))
        await client.close()


def stream_remote_agent(
    url: str,
    text: str,
    history: list[dict],
    customer_id: str,
    thread_id: str,
    timeout_s: float,
    push: PushTarget | None = None,
) -> Iterator[RemoteEvent]:
    """Send one message to a remote A2A agent and yield its events as they arrive (blocking iterator).

    The async stream runs on a private event loop so this can be used from a synchronous graph node.

    Args:
        url (str): Base URL of the agent, e.g. ``http://127.0.0.1:8003``.
        text (str): Latest customer message.
        history (list[dict]): Prior messages as ``{"role", "text", "name"}`` dicts, oldest first.
        customer_id (str): Customer the conversation belongs to.
        thread_id (str): Conversation thread id; used as the A2A ``contextId``.
        timeout_s (float): HTTP timeout in seconds.
        push (PushTarget | None): Webhook for push notifications, used as a fallback if the stream
            breaks; ignored by agents that do not advertise the capability.

    Yields:
        RemoteEvent: Events in the order the agent produced them.

    Raises:
        httpx.HTTPError: If the agent cannot be reached or the stream breaks mid-way.
    """
    loop = asyncio.new_event_loop()
    events = _stream(url, text, history, customer_id, thread_id, timeout_s, push)
    try:
        while True:
            try:
                yield loop.run_until_complete(events.__anext__())
            except StopAsyncIteration:
                return
    finally:
        loop.run_until_complete(events.aclose())
        loop.close()
