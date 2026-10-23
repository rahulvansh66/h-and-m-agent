"""Shipment tracking agent: finds the order id in the conversation and checks the carrier.

Role in the system: the A2A agent whose tasks are slow, so its Agent Card advertises push
notifications. It is deliberately not an LLM agent: the order id is extracted with a regex and the
only step is the slow carrier lookup, streamed as progress lines. That keeps the demo free of Groq
calls and makes streaming and push notifications the only moving parts.
"""

import logging
import re
from collections.abc import Iterator

from hm_agent.a2a.chunks import Chunk, answer
from hm_agent.shipment_tracking.tools import stream_carrier_status

logger = logging.getLogger(__name__)

ORDER_ID_PATTERN = re.compile(r"\bORD-\d+\b", re.IGNORECASE)
ASK_FOR_ORDER_REPLY = "Which order should I track? Please send the order id, for example ORD-1007."


def find_order_id(text: str, history: list[dict]) -> str | None:
    """Find the order id the customer is asking about.

    The latest message wins; otherwise the most recent earlier message that mentions one, so a
    follow-up like "where is it now?" still works.

    Args:
        text (str): Latest customer message.
        history (list[dict]): Prior messages as ``{"role", "text", "name"}`` dicts, oldest first.

    Returns:
        str | None: Order id in upper case (e.g. ``ORD-1007``), or None if none was mentioned.
    """
    for candidate in [text, *(m["text"] for m in reversed(history))]:
        match = ORDER_ID_PATTERN.search(candidate)
        if match:
            return match.group(0).upper()
    return None


def shipment_runner(text: str, history: list[dict], customer_id: str, thread_id: str) -> Iterator[Chunk]:
    """Answer a shipment question: ask for an order id, or check the carrier and report.

    Args:
        text (str): Latest customer message.
        history (list[dict]): Prior messages as ``{"role", "text", "name"}`` dicts, oldest first.
        customer_id (str): Customer the conversation belongs to (only logged).
        thread_id (str): Conversation thread id (only logged).

    Yields:
        Chunk: Progress lines while the carrier is checked, then the status text (or the request
        for an order id, when none was given) as the answer.
    """
    order_id = find_order_id(text, history)
    if order_id is None:
        logger.info("agent=shipment_tracking step=no_order_id thread_id=%s customer_id=%s", thread_id, customer_id)
        yield answer(ASK_FOR_ORDER_REPLY)
        return
    yield from stream_carrier_status(order_id)
