"""Dummy carrier / warehouse lookup used by the shipment tracking agent.

Role in the system: stands in for a slow external system (a carrier API or a warehouse
management system). It waits ``SHIPMENT_CARRIER_DELAY_S`` seconds and then returns a status that
is derived from the order id, so the same order always gets the same answer. The wait is what makes
the agent a good fit for A2A push notifications: the customer does not hold a connection open.
"""

import hashlib
import logging
import time
from collections.abc import Iterator

from hm_agent.a2a.chunks import Chunk, answer, progress
from hm_agent.core.settings import SHIPMENT_CARRIER_DELAY_S

logger = logging.getLogger(__name__)

# Progress lines of a streamed lookup; the carrier wait is split evenly across them.
PROGRESS_STEPS = (
    "Contacting the carrier about {order_id}",
    "Waiting for the carrier system to respond",
    "Reading the latest scan events",
    "Preparing your delivery update",
)

# (carrier, last scan location, status line template); the order id picks one deterministically.
_SCENARIOS: tuple[tuple[str, str, str], ...] = (
    ("DHL Express", "Hamburg sorting hub", "In transit, expected delivery in 2 days"),
    ("PostNord", "Stockholm distribution centre", "Out for delivery, expected today"),
    ("UPS", "Rotterdam depot", "Held at the depot, delivery attempt scheduled for tomorrow"),
    ("DB Schenker", "H&M central warehouse, Hamburg", "Packed and handed over to the carrier, pickup pending"),
)


def _build_status(order_id: str) -> str:
    """Build the carrier status text for an order id (deterministic, no waiting).

    Args:
        order_id (str): Order id such as ``ORD-1007``.

    Returns:
        str: Human-readable status with carrier, tracking number, last scan and delivery outlook.
    """
    digest = hashlib.sha256(order_id.upper().encode()).hexdigest()
    carrier, location, outlook = _SCENARIOS[int(digest[:8], 16) % len(_SCENARIOS)]
    tracking = f"{carrier.split()[0].upper()}{int(digest[8:18], 16) % 10**10:010d}"
    logger.info("tool=check_carrier_status status=complete order_id=%s carrier=%s", order_id, carrier)
    return (
        f"Order {order_id.upper()} is with {carrier} (tracking {tracking}). "
        f"Last scan: {location}. {outlook}."
    )


def check_carrier_status(order_id: str, delay_s: float = SHIPMENT_CARRIER_DELAY_S) -> str:
    """Look up the shipping status of an order with the (simulated) carrier.

    Blocks for ``delay_s`` seconds to imitate a slow external system, then builds a status text
    from a scenario chosen by hashing the order id.

    Args:
        order_id (str): Order id such as ``ORD-1007``.
        delay_s (float): Seconds to wait before answering. Defaults to the configured carrier delay.

    Returns:
        str: Human-readable status with carrier, tracking number, last scan and delivery outlook.
    """
    logger.info("tool=check_carrier_status status=start order_id=%s delay_s=%.1f", order_id, delay_s)
    time.sleep(delay_s)
    return _build_status(order_id)


def stream_carrier_status(order_id: str, delay_s: float = SHIPMENT_CARRIER_DELAY_S) -> Iterator[Chunk]:
    """Look up the shipping status like ``check_carrier_status``, reporting progress while waiting.

    The wait is split evenly across ``PROGRESS_STEPS``; each step yields a progress line before
    sleeping, so a streaming caller sees the lookup advance instead of silence.

    Args:
        order_id (str): Order id such as ``ORD-1007``.
        delay_s (float): Total seconds to wait before answering. Defaults to the configured carrier delay.

    Yields:
        Chunk: One progress chunk per step, then the status text as the answer chunk.
    """
    logger.info("tool=check_carrier_status status=start order_id=%s delay_s=%.1f", order_id, delay_s)
    for step in PROGRESS_STEPS:
        yield progress(step.format(order_id=order_id.upper()))
        time.sleep(delay_s / len(PROGRESS_STEPS))
    yield answer(_build_status(order_id))
