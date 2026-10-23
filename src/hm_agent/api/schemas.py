"""Request and response models of the API.

Role in the system: the public contract of ``hm_agent.api``. FastAPI validates incoming JSON against
these models and builds the OpenAPI document from them.
"""

from typing import Literal

from pydantic import BaseModel, Field

from hm_agent.core.settings import API_MAX_MESSAGE_CHARS


class ChatRequest(BaseModel):
    """Body of ``POST /v1/chat`` and ``POST /v1/chat/stream``.

    Attributes:
        message: The customer's message.
        thread_id: Conversation to continue; a new thread is started when omitted.
    """

    message: str = Field(min_length=1, max_length=API_MAX_MESSAGE_CHARS)
    thread_id: str | None = Field(default=None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class ChatResponse(BaseModel):
    """Answer of ``POST /v1/chat``.

    Attributes:
        thread_id: Conversation the message belongs to (send it back to continue).
        agent: Agent that produced the reply, e.g. ``order_management``.
        reply: Full reply text.
    """

    thread_id: str
    agent: str
    reply: str


class ThreadMessage(BaseModel):
    """One message of a stored conversation.

    Attributes:
        role: ``user`` for the customer, ``assistant`` for any agent.
        agent: Name of the agent that wrote an assistant message, otherwise None.
        text: Message text.
    """

    role: Literal["user", "assistant"]
    agent: str | None = None
    text: str


class HistoryResponse(BaseModel):
    """Answer of ``GET /v1/threads/{thread_id}/messages``.

    Attributes:
        thread_id: The conversation id.
        messages: Messages in order, oldest first.
    """

    thread_id: str
    messages: list[ThreadMessage]


class AgentStatus(BaseModel):
    """Reachability of one A2A agent.

    Attributes:
        agent: Agent name.
        ready: True when its Agent Card answered.
    """

    agent: str
    ready: bool


class ReadyResponse(BaseModel):
    """Answer of ``GET /readyz``.

    Attributes:
        ready: True when every agent is reachable.
        agents: Per-agent status.
    """

    ready: bool
    agents: list[AgentStatus]
