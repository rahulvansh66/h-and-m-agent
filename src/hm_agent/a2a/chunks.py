"""Streaming contract between an agent runner and the A2A executor.

Role in the system: a runner no longer returns one string; it yields ``Chunk`` objects as it works.
The executor (``hm_agent.a2a.server``) turns each one into an A2A event: ``progress`` becomes a
``working`` status update carrying a message, ``answer`` becomes an artifact chunk of the reply.
"""

from dataclasses import dataclass
from typing import Literal

ChunkKind = Literal["progress", "answer"]


@dataclass(frozen=True)
class Chunk:
    """One piece of output from a runner.

    Attributes:
        kind: ``"progress"`` for a status line ("Contacting carrier"), ``"answer"`` for a piece of the reply.
        text: The text of this piece. Answer pieces are concatenated in order to form the full reply.
    """

    kind: ChunkKind
    text: str


def progress(text: str) -> Chunk:
    """Build a progress chunk.

    Args:
        text (str): Status line to show while the agent works.

    Returns:
        Chunk: Chunk of kind ``progress``.
    """
    return Chunk("progress", text)


def answer(text: str) -> Chunk:
    """Build an answer chunk.

    Args:
        text (str): A piece of the final reply.

    Returns:
        Chunk: Chunk of kind ``answer``.
    """
    return Chunk("answer", text)
