"""Tests for the split-turn stream coordinator: baton hand-over, no interleaving, history == shown text.

Role in the system: checks ``hm_agent.supervisor.split_stream`` with plain threads and events (no sleeps
decide the outcome), so the ordering guarantees of progressive streaming are proven, not assumed.
"""

import threading

import pytest

from hm_agent.supervisor import split_stream
from hm_agent.supervisor.split_stream import LEAD_INS, SplitStream

FASHION, SHIPMENT = "fashion", "shipment_tracking"


@pytest.fixture(autouse=True)
def clean_registry():
    """Start and end each test with no registered turns, so counts are not affected by other tests."""
    split_stream._TURNS.clear()
    yield
    split_stream._TURNS.clear()


class Recorder:
    """Thread-safe stand-in for the stream writer that keeps the token texts in order."""

    def __init__(self) -> None:
        """Create an empty recorder."""
        self.events: list[dict] = []

    def __call__(self, event: dict) -> None:
        """Record one event.

        Args:
            event (dict): Event written by the coordinator.
        """
        self.events.append(event)

    @property
    def text(self) -> str:
        """str: All token texts joined, as the customer would read them."""
        return "".join(e["text"] for e in self.events)


def test_first_piece_takes_the_baton_and_streams_live() -> None:
    """The first branch to answer streams at once behind its lead-in; the other is held back."""
    stream, out = SplitStream(), Recorder()
    stream.piece(FASHION, "Wear jeans.", out)
    stream.piece(SHIPMENT, "On its way.", out)
    assert out.text == f"{LEAD_INS[FASHION]}\nWear jeans."
    assert stream.streamed == {FASHION}
    assert all(e["agent"] == "multi_agent" and e["type"] == "token" for e in out.events)


def test_baton_passes_to_a_branch_that_already_finished() -> None:
    """When the live branch ends, a buffered finished branch is shown right after it, in one block."""
    stream, out = SplitStream(), Recorder()
    stream.piece(FASHION, "Jeans.", out)
    stream.piece(SHIPMENT, "On its way.", out)
    stream.finish(SHIPMENT, True, out)  # finished while not holding the baton: still held back
    assert "On its way." not in out.text
    stream.finish(FASHION, True, out)
    assert out.text == f"{LEAD_INS[FASHION]}\nJeans.\n\n{LEAD_INS[SHIPMENT]}\nOn its way."
    assert stream.baton_passes == 1


def test_baton_passes_to_a_running_branch_which_then_continues_live() -> None:
    """A still-running branch gets its buffer flushed at hand-over and then streams its next pieces live."""
    stream, out = SplitStream(), Recorder()
    stream.piece(FASHION, "Jeans.", out)
    stream.piece(SHIPMENT, "On ", out)
    stream.finish(FASHION, True, out)
    assert out.text.endswith(f"{LEAD_INS[SHIPMENT]}\nOn ")
    stream.piece(SHIPMENT, "its way.", out)
    assert out.text.endswith("On its way.")
    stream.finish(SHIPMENT, True, out)
    assert stream.emitted_text() == out.text


def test_failed_branch_without_the_baton_is_never_shown() -> None:
    """Buffered pieces of a branch that then fails are dropped; synthesize adds the note for it."""
    stream, out = SplitStream(), Recorder()
    stream.piece(FASHION, "Jeans.", out)
    stream.piece(SHIPMENT, "partial", out)
    stream.finish(SHIPMENT, False, out)
    stream.finish(FASHION, True, out)
    assert "partial" not in out.text and LEAD_INS[SHIPMENT] not in out.text
    assert stream.streamed == {FASHION}


def test_live_branch_that_fails_keeps_what_was_shown_and_frees_the_baton() -> None:
    """A branch that dies after streaming leaves its text on screen and the next branch still gets its turn."""
    stream, out = SplitStream(), Recorder()
    stream.piece(FASHION, "Jea", out)
    stream.piece(SHIPMENT, "On its way.", out)
    stream.finish(FASHION, False, out)
    stream.finish(SHIPMENT, True, out)
    assert out.text == f"{LEAD_INS[FASHION]}\nJea\n\n{LEAD_INS[SHIPMENT]}\nOn its way."


def test_late_branch_after_everything_finished_streams_immediately() -> None:
    """If nobody holds the baton, the next branch to answer streams as soon as it produces a piece."""
    stream, out = SplitStream(), Recorder()
    stream.piece(FASHION, "Jeans.", out)
    stream.finish(FASHION, True, out)
    stream.piece(SHIPMENT, "On its way.", out)
    assert out.text.endswith("On its way.")


def test_emit_helpers_add_blank_lines_between_blocks() -> None:
    """Sections and notes are separated by a blank line; raw pieces are not."""
    stream, out = SplitStream(), Recorder()
    stream.emit_section(FASHION, "Jeans.", out)
    stream.emit_note("Note.", out)
    stream.emit_raw(" more", out)
    assert out.text == f"{LEAD_INS[FASHION]}\nJeans.\n\nNote. more"
    assert stream.streamed == {FASHION}


def test_parallel_branches_never_interleave() -> None:
    """Two threads producing pieces at once still yield two whole sections, one after the other."""
    stream, out = SplitStream(), Recorder()
    first_piece_in, second_done = threading.Event(), threading.Event()

    def live() -> None:
        stream.piece(FASHION, "A1 ", out)
        first_piece_in.set()
        second_done.wait(5)  # the other branch finishes while this one still holds the baton
        stream.piece(FASHION, "A2 ", out)
        stream.finish(FASHION, True, out)

    def held() -> None:
        first_piece_in.wait(5)
        for piece in ("B1 ", "B2 ", "B3"):
            stream.piece(SHIPMENT, piece, out)
        stream.finish(SHIPMENT, True, out)
        second_done.set()

    threads = [threading.Thread(target=live), threading.Thread(target=held)]
    [t.start() for t in threads]
    [t.join(10) for t in threads]
    assert out.text == f"{LEAD_INS[FASHION]}\nA1 A2 \n\n{LEAD_INS[SHIPMENT]}\nB1 B2 B3"


@pytest.mark.parametrize("repeat", range(20))
def test_racing_branches_always_produce_whole_sections(repeat: int) -> None:
    """Many racing runs: whatever the thread timing, each agent's text appears as one contiguous run."""
    stream, out = SplitStream(), Recorder()
    start = threading.Barrier(2)

    def branch(agent: str, tag: str) -> None:
        start.wait(5)
        for i in range(5):
            stream.piece(agent, f"{tag}{i} ", out)
        stream.finish(agent, True, out)

    threads = [threading.Thread(target=branch, args=(FASHION, "f")), threading.Thread(target=branch, args=(SHIPMENT, "s"))]
    [t.start() for t in threads]
    [t.join(10) for t in threads]
    body = out.text
    assert "f0 f1 f2 f3 f4 " in body and "s0 s1 s2 s3 s4 " in body
    assert stream.emitted_text() == body


def test_registry_creates_once_pops_and_does_not_leak(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same turn id gets the same coordinator, popping removes it, and stale turns are dropped."""
    first = split_stream.get_split_stream("turn-a")
    assert split_stream.get_split_stream("turn-a") is first
    assert split_stream.pop_split_stream("turn-a") is first
    assert split_stream.active_turns() == 0
    split_stream.get_split_stream("turn-old")
    monkeypatch.setattr(split_stream, "STALE_AFTER_S", -1)
    split_stream.get_split_stream("turn-new")
    assert split_stream.active_turns() == 1  # only the new one survived the purge
    split_stream.pop_split_stream("turn-new")
    assert split_stream.pop_split_stream("never-created").emitted_text() == ""
