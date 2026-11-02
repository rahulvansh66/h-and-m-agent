"""Registry of every Langfuse-managed prompt and its in-code default.

Role in the system: single list used by ``scripts/langfuse/sync_prompts.py`` to push the code defaults to
Langfuse as version 1 (label ``production``), and by tests to check that every prompt has a fallback. Agent
modules never import this file (they import ``hm_agent.core.prompts``), so there is no import cycle.
"""

from dataclasses import dataclass

from hm_agent.core.prompts import ChatMessages, PromptKind
from hm_agent.customer_support.agent import CONDENSE_CHAT_PROMPT, CONDENSE_PROMPT_NAME
from hm_agent.customer_support.rag.pipeline import (
    ANSWER_CHAT_PROMPT,
    ANSWER_PROMPT_NAME,
    REWRITE_CHAT_PROMPT,
    REWRITE_PROMPT_NAME,
)
from hm_agent.fashion.agent import FASHION_PROMPT, FASHION_PROMPT_NAME
from hm_agent.order_management.agent import ORDER_MANAGEMENT_PROMPT, ORDER_MANAGEMENT_PROMPT_NAME
from hm_agent.supervisor.agent import SUPERVISOR_CHAT_PROMPT, SUPERVISOR_PROMPT_NAME
from hm_agent.supervisor.synthesis import SYNTHESIS_CHAT_PROMPT, SYNTHESIS_PROMPT_NAME


@dataclass(frozen=True)
class PromptSpec:
    """One managed prompt.

    Attributes:
        name (str): Langfuse prompt name.
        kind (PromptKind): ``"text"`` or ``"chat"``.
        default (str | ChatMessages): The in-code text or messages, with ``{{variable}}`` placeholders.
    """

    name: str
    kind: PromptKind
    default: str | ChatMessages


PROMPTS: list[PromptSpec] = [
    PromptSpec(SUPERVISOR_PROMPT_NAME, "chat", SUPERVISOR_CHAT_PROMPT),
    PromptSpec(SYNTHESIS_PROMPT_NAME, "chat", SYNTHESIS_CHAT_PROMPT),
    PromptSpec(CONDENSE_PROMPT_NAME, "chat", CONDENSE_CHAT_PROMPT),
    PromptSpec(ANSWER_PROMPT_NAME, "chat", ANSWER_CHAT_PROMPT),
    PromptSpec(REWRITE_PROMPT_NAME, "chat", REWRITE_CHAT_PROMPT),
    PromptSpec(ORDER_MANAGEMENT_PROMPT_NAME, "text", ORDER_MANAGEMENT_PROMPT),
    PromptSpec(FASHION_PROMPT_NAME, "text", FASHION_PROMPT),
]
