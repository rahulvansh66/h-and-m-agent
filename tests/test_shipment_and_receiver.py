"""Tests for the shipment agent's streamed lookup and the push receiver's streamed-task bookkeeping.

Role in the system: covers the pieces of the streaming + push combination that need no server:
progress chunks from the slow carrier lookup, order id extraction, and ignoring webhook
notifications for tasks whose result already arrived over the stream.
"""

from hm_agent.a2a.push_receiver import PushReceiver
from hm_agent.shipment_tracking.agent import ASK_FOR_ORDER_REPLY, find_order_id, shipment_runner
from hm_agent.shipment_tracking.tools import PROGRESS_STEPS, stream_carrier_status


def test_carrier_lookup_streams_progress_then_answer() -> None:
    """One progress line per step, then a single answer chunk naming the order."""
    chunks = list(stream_carrier_status("ORD-1007", delay_s=0.04))
    assert [c.kind for c in chunks] == ["progress"] * len(PROGRESS_STEPS) + ["answer"]
    assert "ORD-1007" in chunks[0].text and "ORD-1007" in chunks[-1].text


def test_runner_asks_for_order_id_when_missing() -> None:
    """Without an order id anywhere in the conversation the agent asks for one."""
    chunks = list(shipment_runner("where is my parcel?", [], "cust001", "t1"))
    assert [(c.kind, c.text) for c in chunks] == [("answer", ASK_FOR_ORDER_REPLY)]


def test_order_id_is_found_in_history() -> None:
    """A follow-up without an id reuses the most recent id from earlier messages."""
    history = [{"role": "user", "text": "track ord-1007", "name": None}]
    assert find_order_id("and now?", history) == "ORD-1007"


def test_mark_streamed_discards_early_terminal_notification() -> None:
    """A terminal notification that beat the stream's end is forgotten, and later ones are ignored."""
    receiver = PushReceiver(port=0, deliver=lambda task, text: None)
    receiver._finished_early["task-1"] = "TASK_STATE_COMPLETED"
    receiver.mark_streamed("task-1")
    assert "task-1" not in receiver._finished_early
    assert "task-1" in receiver._claimed
