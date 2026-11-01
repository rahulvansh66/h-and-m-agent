"""Stream coordinator for split turns: shows the agents' answers one section at a time, as they arrive.

Role in the system: when the supervisor sends one message to several agents in parallel, their answers
must not interleave and the customer should not wait for the slowest agent before seeing text. The branch
nodes (``hm_agent.a2a.remote_node``) hand every answer piece to the turn's ``SplitStream``. The first
branch to produce a piece holds the baton and its pieces stream live behind a short lead-in; the pieces of
the other branches are buffered. When the baton holder finishes, the baton passes to the next branch: its
buffer is flushed at once and, if it is still running, it continues live. ``synthesize`` closes the turn by
streaming the notes for parts that failed, and builds the history message from ``emitted_text``, so what is
saved is exactly what the customer saw.

The registry is in-process: the branches and ``synthesize`` of one turn run in the same graph run, which is
how the runtime executes them. Entries are dropped by ``synthesize`` and, as a safety net, after
``STALE_AFTER_S`` if a turn crashed before getting there.
"""

import logging
import threading
import time
from collections.abc import Callable

from hm_agent.core.state import MULTI_AGENT

logger = logging.getLogger(__name__)

STALE_AFTER_S = 600  # a registered turn older than this is assumed dead and dropped
SECTION_SEPARATOR = "\n\n"

# One short lead-in per agent, shown before its section so the customer can tell the parts apart.
LEAD_INS = {
    "order_management": "About your orders:",
    "shipment_tracking": "About your delivery:",
    "customer_support": "About the problem with your item:",
    "fashion": "About outfit and product suggestions:",
}

Writer = Callable[[dict], None]


class _Section:
    """Pieces of one branch's answer and where it is in the show."""

    def __init__(self) -> None:
        """Create an empty section."""
        self.pending: list[str] = []  # pieces received but not shown yet
        self.finished = False
        self.complete = False
        self.shown = False  # lead-in emitted: the section is, or was, on screen


class SplitStream:
    """Per-turn coordinator that serialises the streaming of parallel branch answers.

    All public methods are thread-safe; branch nodes call them from different worker threads.
    """

    def __init__(self) -> None:
        """Create a coordinator with nothing streamed yet."""
        self._lock = threading.RLock()
        self._created = time.monotonic()
        self._sections: dict[str, _Section] = {}
        self._waiting: list[str] = []  # agents with buffered pieces, in the order they first produced one
        self._baton: str | None = None
        self._emitted: list[str] = []
        self._first_token_s: float | None = None
        self.baton_passes = 0

    @property
    def created_at(self) -> float:
        """float: Monotonic creation time, used to drop stale registry entries."""
        return self._created

    @property
    def streamed(self) -> set[str]:
        """set[str]: Agents whose section was (at least partly) shown to the customer."""
        with self._lock:
            return {agent for agent, section in self._sections.items() if section.shown}

    def emitted_text(self) -> str:
        """Return everything shown to the customer so far.

        Returns:
            str: The concatenated token texts; this is what ``synthesize`` saves to the history.
        """
        with self._lock:
            return "".join(self._emitted)

    def piece(self, agent: str, text: str, writer: Writer) -> None:
        """Take one answer piece from a running branch.

        The first branch to call this holds the baton and its piece is streamed at once; for the others
        the piece is buffered until the baton reaches them.

        Args:
            agent: Branch that produced the piece.
            text: The piece of answer text.
            writer: Stream writer of the calling node; used to emit when this call has something to show.
        """
        with self._lock:
            section = self._sections.setdefault(agent, _Section())
            section.pending.append(text)
            # Queue the agent once, in arrival order, unless it is already on screen.
            if agent not in self._waiting and not section.shown:
                self._waiting.append(agent)
            if self._baton is None:
                self._advance(writer)  # nobody is streaming: the first producer takes the baton
            elif self._baton == agent:
                self._flush(agent, writer)  # we hold the baton: show the piece live
            # otherwise the piece stays buffered until the baton reaches this agent

    def finish(self, agent: str, complete: bool, writer: Writer) -> None:
        """Mark a branch as done and pass the baton on if it held it.

        A branch that did not complete loses any pieces that were not shown yet; what was already on
        screen stays, and ``synthesize`` adds the failure note.

        Args:
            agent: Branch that finished.
            complete: True if it ended with a complete answer, False if it failed or went pending.
            writer: Stream writer of the calling node.
        """
        with self._lock:
            section = self._sections.setdefault(agent, _Section())
            section.finished, section.complete = True, complete
            if not complete and self._baton != agent:
                # A failed, never-shown answer is dropped so partial text is not presented as a section.
                section.pending.clear()
                if agent in self._waiting:
                    self._waiting.remove(agent)
            if self._baton == agent:
                self._baton = None  # release the baton, then hand it to the next waiting branch
                self._advance(writer)
            elif self._baton is None:
                self._advance(writer)  # a buffered branch may have been waiting for a free baton

    def emit_section(self, agent: str, text: str, writer: Writer) -> None:
        """Show a whole section that was not streamed live (used by ``synthesize``).

        Args:
            agent: Agent the text belongs to.
            text: The complete answer.
            writer: Stream writer of the calling node.
        """
        with self._lock:
            self._sections.setdefault(agent, _Section()).shown = True
            self._emit(f"{LEAD_INS.get(agent, '')}\n{text}".lstrip("\n"), writer, new_block=True)

    def emit_note(self, text: str, writer: Writer) -> None:
        """Show a standalone block such as a failure note or a sources list.

        Args:
            text: Block text.
            writer: Stream writer of the calling node.
        """
        with self._lock:
            self._emit(text, writer, new_block=True)

    def emit_raw(self, text: str, writer: Writer) -> None:
        """Show a piece of continuing text with no separator (used by the streaming composer).

        Args:
            text: Piece of text.
            writer: Stream writer of the calling node.
        """
        with self._lock:
            self._emit(text, writer, new_block=False)

    def log_summary(self, thread_id: str) -> None:
        """Log how the turn streamed (first-token latency and baton hand-overs).

        Args:
            thread_id: Conversation thread, for correlation.
        """
        first = f"{self._first_token_s:.2f}" if self._first_token_s is not None else "none"
        logger.info("node=split_stream thread_id=%s first_token_s=%s baton_passes=%d", thread_id, first, self.baton_passes)

    # --- internals (callers hold the lock) ---

    def _emit(self, text: str, writer: Writer, new_block: bool) -> None:
        """Write one token event and record it.

        Args:
            text: Text to show.
            writer: Stream writer to use.
            new_block: True to put a blank line before it when something was already shown.
        """
        if not text:
            return
        if new_block and self._emitted:
            text = SECTION_SEPARATOR + text  # blank line between blocks, but never before the first one
        if self._first_token_s is None:  # time-to-first-token, reported by ``log_summary``
            self._first_token_s = time.monotonic() - self._created
        self._emitted.append(text)
        writer({"type": "token", "agent": MULTI_AGENT, "text": text})

    def _flush(self, agent: str, writer: Writer) -> None:
        """Show the buffered pieces of the baton holder.

        Args:
            agent: The baton holder.
            writer: Stream writer to use.
        """
        section = self._sections[agent]
        if section.pending:
            self._emit("".join(section.pending), writer, new_block=False)
            section.pending.clear()

    def _advance(self, writer: Writer) -> None:
        """Give the baton to the next waiting branch, repeating while branches are already finished.

        Args:
            writer: Stream writer to use.
        """
        while self._baton is None and self._waiting:
            agent = self._waiting.pop(0)
            section = self._sections[agent]
            if not section.pending and section.finished:
                continue  # finished with nothing to show (e.g. failed): skip it, try the next waiting branch
            if self._emitted:
                self.baton_passes += 1  # only counts real hand-overs, not the very first start
            section.shown = True
            self._emit(LEAD_INS.get(agent, "") + "\n", writer, new_block=True)
            self._flush(agent, writer)
            if not section.finished:
                self._baton = agent  # still running: it continues live


# Process-wide registry turn_id -> coordinator. Branches run on different threads, so access is locked.
_TURNS: dict[str, SplitStream] = {}
_TURNS_LOCK = threading.Lock()


def get_split_stream(turn_id: str) -> SplitStream:
    """Return the coordinator of a turn, creating it for the first branch that asks.

    Args:
        turn_id: Identifier the supervisor stored in the state for this turn.

    Returns:
        SplitStream: The shared coordinator. Stale entries of crashed turns are dropped on the way.
    """
    with _TURNS_LOCK:
        # Opportunistic cleanup: sweep abandoned turns here instead of running a background reaper thread.
        now = time.monotonic()
        for key in [k for k, v in _TURNS.items() if now - v.created_at > STALE_AFTER_S]:
            del _TURNS[key]
            logger.warning("node=split_stream status=stale_dropped turn_id=%s", key)
        return _TURNS.setdefault(turn_id, SplitStream())


def pop_split_stream(turn_id: str) -> SplitStream:
    """Remove and return a turn's coordinator; a fresh, empty one if no branch created it.

    Args:
        turn_id: Identifier of the turn.

    Returns:
        SplitStream: The coordinator that streamed the turn (or an empty one).
    """
    with _TURNS_LOCK:
        return _TURNS.pop(turn_id, None) or SplitStream()


def active_turns() -> int:
    """Return how many turns are registered (used by tests to check nothing leaks).

    Returns:
        int: Number of registered coordinators.
    """
    with _TURNS_LOCK:
        return len(_TURNS)
