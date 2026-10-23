"""Webhook receiver for A2A push notifications, run inside the supervisor process.

Role in the system: when an agent advertises push notifications, ``remote_node`` starts its task in
the background and registers the task here. The agent then POSTs every status/artifact update of
that task to this receiver. The receiver authenticates the call, waits for a terminal state, reads
the final result with ``GetTask`` and hands it to a delivery callback (the CLI prints it and saves it
into the conversation thread).

The module also keeps one process-wide receiver (``set_receiver`` / ``get_receiver``) so the graph
node can find it without the receiver being threaded through the graph config.
"""

import asyncio
import hmac
import json
import logging
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from hm_agent.a2a.client import PushTarget, fetch_task, task_reply
from hm_agent.a2a.wire_log import log_push_notification
from hm_agent.core.settings import A2A_HOST, A2A_TIMEOUT_S

logger = logging.getLogger(__name__)

NOTIFY_PATH = "/a2a/notify"
TOKEN_HEADER = "X-A2A-Notification-Token"
STARTUP_TIMEOUT_S = 10
# Idle connections are kept open longer than the gap between progress updates. With uvicorn's 5 s default a
# pooled sender connection can be closed just as the next update is posted, and that update is lost
# (the SDK sender does not retry).
KEEP_ALIVE_S = 75
FAILED_REPLY = "Sorry, the shipment lookup did not finish. Please try again in a moment."

COMPLETED_STATE = "TASK_STATE_COMPLETED"
# Task states after which no further updates come (names as sent in the JSON payload).
_TERMINAL_STATES = {
    COMPLETED_STATE,
    "TASK_STATE_FAILED",
    "TASK_STATE_CANCELED",
    "TASK_STATE_REJECTED",
}


@dataclass(frozen=True)
class PendingTask:
    """A background task the supervisor is waiting on.

    Attributes:
        task_id: A2A task id assigned by the agent.
        agent: Agent name; also the name of the AI message that will carry the result.
        url: Base URL of the agent, used to read the final result.
        customer_id: Customer the conversation belongs to.
        thread_id: Conversation thread the result must be added to.
    """

    task_id: str
    agent: str
    url: str
    customer_id: str
    thread_id: str


# Called with the finished task and the text to show; runs on a worker thread.
DeliverFn = Callable[[PendingTask, str], None]


def _event_task_state(payload: dict) -> tuple[str, str | None]:
    """Extract the task id and, if present, the task state from a pushed ``StreamResponse`` JSON.

    Args:
        payload (dict): Decoded JSON body of the notification.

    Returns:
        tuple[str, str | None]: ``(task_id, state_name)``; the state is None for artifact updates
        and unknown shapes, and the id is an empty string when it cannot be found.
    """
    if "task" in payload:
        task = payload["task"]
        return task.get("id", ""), task.get("status", {}).get("state")
    if "statusUpdate" in payload:
        update = payload["statusUpdate"]
        return update.get("taskId", ""), update.get("status", {}).get("state")
    if "artifactUpdate" in payload:
        return payload["artifactUpdate"].get("taskId", ""), None
    return "", None


class PushReceiver:
    """HTTP receiver for push notifications plus the registry of tasks being waited on."""

    def __init__(self, port: int, deliver: DeliverFn) -> None:
        """Create the receiver (not started yet).

        Args:
            port (int): Local port to listen on.
            deliver (DeliverFn): Callback invoked once per finished task with the text to show.
        """
        self._port = port
        self._deliver = deliver
        self._token = secrets.token_urlsafe(24)  # new secret per process; sent to agents per task
        self._pending: dict[str, PendingTask] = {}
        self._finished_early: dict[str, str] = {}  # terminal state seen before register() ran
        self._claimed: set[str] = set()  # task ids already being delivered; guards duplicate notifications
        self._changed = threading.Condition()
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def target(self) -> PushTarget:
        """Webhook address and shared secret to hand to agents.

        Returns:
            PushTarget: URL of ``/a2a/notify`` on this receiver plus its token.
        """
        return PushTarget(url=f"http://{A2A_HOST}:{self._port}{NOTIFY_PATH}", token=self._token)

    def start(self) -> None:
        """Start the receiver on a background thread and wait until it accepts connections.

        Raises:
            RuntimeError: If the server did not come up within ``STARTUP_TIMEOUT_S`` seconds.
        """
        app = Starlette(routes=[Route(NOTIFY_PATH, self._handle, methods=["POST"])])
        self._server = uvicorn.Server(uvicorn.Config(app, host=A2A_HOST, port=self._port, log_level="warning", timeout_keep_alive=KEEP_ALIVE_S))
        self._thread = threading.Thread(target=self._server.run, name="a2a-push-receiver", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while not self._server.started:
            if time.monotonic() > deadline or not self._thread.is_alive():
                raise RuntimeError(f"push receiver did not start on port {self._port}")
            time.sleep(0.05)
        logger.info("component=push_receiver status=started url=%s", self.target.url)

    def stop(self) -> None:
        """Ask the server to exit and wait briefly for its thread."""
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)

    def register(self, task: PendingTask) -> None:
        """Remember a background task so its result can be delivered when it finishes.

        Args:
            task (PendingTask): Task started with a push webhook.
        """
        with self._changed:
            self._pending[task.task_id] = task
            early = self._finished_early.pop(task.task_id, None)
        logger.info("component=push_receiver status=registered task_id=%s agent=%s", task.task_id, task.agent)
        if early is not None:
            # The agent finished before the start call returned; deliver now.
            threading.Thread(target=self._finish, args=(task.task_id, early), daemon=True).start()

    def mark_streamed(self, task_id: str) -> None:
        """Record that a task's result already reached the caller through its stream.

        Notifications for it are then ignored, and a terminal one that arrived before this call is
        forgotten, so nothing is delivered twice or kept forever.

        Args:
            task_id (str): Id of a task whose stream ended in a terminal state.
        """
        with self._changed:
            self._claimed.add(task_id)
            self._finished_early.pop(task_id, None)
        logger.info("component=push_receiver status=streamed_ignore task_id=%s", task_id)

    def wait_idle(self, timeout_s: float) -> bool:
        """Block until no task is pending.

        Used by one-shot CLI runs so the process does not exit before a pushed result arrives.

        Args:
            timeout_s (float): Maximum seconds to wait.

        Returns:
            bool: True if every pending task was delivered, False on timeout.
        """
        with self._changed:
            return self._changed.wait_for(lambda: not self._pending, timeout=timeout_s)

    async def _handle(self, request: Request) -> Response:
        """Handle one notification POST from an agent.

        Rejects calls without the shared token, ignores intermediate updates and, on a terminal
        state, reads the result and delivers it.

        Args:
            request (Request): Incoming notification.

        Returns:
            Response: 401 for a bad token, 400 for a non-JSON body, otherwise 200.
        """
        if not hmac.compare_digest(request.headers.get(TOKEN_HEADER, ""), self._token):
            logger.warning("component=push_receiver status=rejected reason=bad_token")
            return JSONResponse({"error": "invalid token"}, status_code=401)
        body = await request.body()
        log_push_notification(str(request.url), body.decode("utf-8", "replace"))  # no-op unless A2A_WIRE_LOG is set
        try:
            payload = json.loads(body)
        except ValueError:
            return JSONResponse({"error": "invalid json"}, status_code=400)
        task_id, state = _event_task_state(payload)
        logger.info("component=push_receiver status=received task_id=%s state=%s", task_id, state)
        if state in _TERMINAL_STATES:
            with self._changed:
                known = task_id in self._pending
                if not known and task_id not in self._claimed:
                    self._finished_early[task_id] = state
            if known:
                await asyncio.to_thread(self._finish, task_id, state)
        return Response(status_code=200)

    def _finish(self, task_id: str, state: str) -> None:
        """Read the result of a finished task and deliver it (runs on a worker thread).

        Args:
            task_id (str): Id of the finished task.
            state (str): Terminal state name, e.g. ``TASK_STATE_COMPLETED``.
        """
        with self._changed:
            task = self._pending.get(task_id)
            if task is None or task_id in self._claimed:
                return  # already delivered or in progress (a task can be reported more than once)
            self._claimed.add(task_id)
        try:
            text = FAILED_REPLY
            if state == COMPLETED_STATE:
                # The notification only says "done"; the answer is read from the task itself.
                text = task_reply(asyncio.run(fetch_task(task.url, task_id, A2A_TIMEOUT_S))) or FAILED_REPLY
            self._deliver(task, text)
            logger.info("component=push_receiver status=delivered task_id=%s state=%s", task_id, state)
        except Exception:
            logger.exception("component=push_receiver status=deliver_failed task_id=%s", task_id)
        finally:
            with self._changed:
                self._pending.pop(task_id, None)
                self._changed.notify_all()


_receiver: PushReceiver | None = None


def set_receiver(receiver: PushReceiver | None) -> None:
    """Install (or clear) the process-wide receiver used by ``remote_node``.

    Args:
        receiver (PushReceiver | None): Started receiver, or None to disable push for this process.
    """
    global _receiver
    _receiver = receiver


def get_receiver() -> PushReceiver | None:
    """Return the process-wide receiver.

    Returns:
        PushReceiver | None: The installed receiver, or None when push notifications are not set up.
    """
    return _receiver
