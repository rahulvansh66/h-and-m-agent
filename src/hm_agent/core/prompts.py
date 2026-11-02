"""Langfuse-managed prompts with an in-code fallback.

Role in the system: every LLM prompt (supervisor routing, synthesis, RAG, agents) is fetched here by
name and label, so prompts are versioned and edited in Langfuse without a deploy. The constants that
stay in each module are the fallback: if Langfuse is unreachable, unconfigured, or the prompt does not
exist yet, the app runs on the in-code text and logs ``source=fallback`` instead of failing.

Chat prompts are fetched as a ``ManagedPrompt`` and used as ``prompt.chain(model)``: the prompt template
carries ``langfuse_prompt`` metadata, which the Langfuse callback handler uses to link every generation
to the exact prompt version that produced it.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langfuse import get_client

from hm_agent.core.settings import PROMPT_CACHE_TTL_S, PROMPT_LABEL

logger = logging.getLogger(__name__)

PromptKind = Literal["text", "chat"]
ChatMessages = list[dict[str, str]]  # [{"role": "system", "content": "... {{var}} ..."}, ...]

# Seconds to wait for Langfuse before using the fallback; keeps a Langfuse outage from stalling a turn.
FETCH_TIMEOUT_S = 3

# Set to a truthy value (tests do this) to skip Langfuse entirely and always use the in-code fallback.
DISABLE_ENV_VAR = "LANGFUSE_PROMPTS_DISABLED"


@dataclass(frozen=True)
class ManagedPrompt:
    """A prompt resolved from Langfuse (or its fallback), ready to compile or chain to a model.

    Attributes:
        name (str): Langfuse prompt name, e.g. ``hm/supervisor-router``.
        kind (PromptKind): ``"text"`` (one string) or ``"chat"`` (list of role/content messages).
        version (int): Langfuse version number; ``0`` for the in-code fallback.
        source (str): ``"langfuse"`` or ``"fallback"``.
        client (Any): The Langfuse prompt client (``TextPromptClient`` / ``ChatPromptClient``).
    """

    name: str
    kind: PromptKind
    version: int
    source: str
    client: Any

    def compile(self, **variables: str) -> str | ChatMessages:
        """Substitute ``{{variable}}`` placeholders.

        Args:
            **variables (str): Value for each ``{{name}}`` placeholder in the prompt.

        Returns:
            str | ChatMessages: The text for a text prompt, or the role/content messages for a chat prompt.
        """
        return self.client.compile(**variables)

    def template(self) -> ChatPromptTemplate:
        """Return the chat prompt as a LangChain template that links generations to this prompt version.

        Returns:
            ChatPromptTemplate: Template whose ``{variable}`` slots are filled at invoke time.

        Raises:
            TypeError: If this is a text prompt (use ``compile`` for those).
        """
        if self.kind != "chat":
            raise TypeError(f"prompt {self.name} is a text prompt; use compile()")
        return ChatPromptTemplate(
            self.client.get_langchain_prompt(), metadata={"langfuse_prompt": self.client}
        )

    def chain(self, model: Runnable) -> Runnable:
        """Pipe this chat prompt into a model so each call is linked to the prompt version in Langfuse.

        Args:
            model (Runnable): Chat model, or ``model.with_structured_output(...)``.

        Returns:
            Runnable: ``template | model``; invoke it with a dict of the prompt's variables.
        """
        return self.template() | model


def _client_enabled() -> bool:
    """Return whether Langfuse credentials exist and prompt fetching is not disabled.

    Returns:
        bool: False when ``LANGFUSE_PROMPTS_DISABLED`` is set or the Langfuse keys are missing.
    """
    if os.environ.get(DISABLE_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def get_managed_prompt(name: str, default: str | ChatMessages, *, kind: PromptKind) -> ManagedPrompt:
    """Fetch a prompt by name and label from Langfuse, falling back to the in-code text.

    Args:
        name (str): Langfuse prompt name, e.g. ``hm/support-condense``.
        default (str | ChatMessages): In-code prompt used when Langfuse is unavailable. Must use
            ``{{variable}}`` placeholders and match ``kind`` (a string, or a list of role/content dicts).
        kind (PromptKind): ``"text"`` or ``"chat"``; a prompt's type cannot change in Langfuse.

    Returns:
        ManagedPrompt: The resolved prompt. Never raises on Langfuse errors.
    """
    langfuse = get_client() if _client_enabled() else None
    if langfuse is not None:
        try:
            client = langfuse.get_prompt(
                name,
                label=PROMPT_LABEL,
                type=kind,
                fallback=default,
                cache_ttl_seconds=PROMPT_CACHE_TTL_S,
                max_retries=1,
                fetch_timeout_seconds=FETCH_TIMEOUT_S,
            )
            # The SDK hands back the fallback (version 0, is_fallback) on a fetch error instead of raising.
            source = "fallback" if getattr(client, "is_fallback", False) else "langfuse"
            logger.debug("prompt=%s label=%s version=%s source=%s", name, PROMPT_LABEL, client.version, source)
            return ManagedPrompt(name, kind, client.version, source, client)
        except Exception:  # noqa: BLE001 - a prompt-store problem must never fail a customer turn
            logger.exception("prompt=%s status=fetch_failed using=fallback", name)
    return _fallback(name, default, kind)


def _fallback(name: str, default: str | ChatMessages, kind: PromptKind) -> ManagedPrompt:
    """Wrap the in-code default as a ``ManagedPrompt`` without calling Langfuse.

    Args:
        name (str): Prompt name (kept for logging and metadata).
        default (str | ChatMessages): In-code prompt text or messages.
        kind (PromptKind): ``"text"`` or ``"chat"``.

    Returns:
        ManagedPrompt: A version-0 prompt with ``source="fallback"``.
    """
    from langfuse.model import ChatPromptClient, TextPromptClient
    from langfuse.api import Prompt_Chat, Prompt_Text

    if kind == "chat":
        raw = Prompt_Chat(name=name, version=0, config={}, labels=[], tags=[], prompt=default)
        client: Any = ChatPromptClient(raw, is_fallback=True)
    else:
        raw = Prompt_Text(name=name, version=0, config={}, labels=[], tags=[], prompt=default)
        client = TextPromptClient(raw, is_fallback=True)
    return ManagedPrompt(name, kind, 0, "fallback", client)
