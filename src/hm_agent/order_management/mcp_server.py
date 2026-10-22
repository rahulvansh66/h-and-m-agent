"""MCP server exposing read-only SQL tools on the Neon ``order_management`` schema.

Role in the system: tool provider for the order_management agent. The agent talks to this
server over MCP (stdio transport, spawned on demand by the client), so no always-on process
is needed. Each tool takes a ``customer_id``; the MCP client overwrites it from the run config
(see ``hm_agent.order_management.mcp_client``), so the LLM never chooses which customer's data is read.

Run standalone (for inspection with an MCP client):
    uv run python -m hm_agent.order_management.mcp_server
"""

import logging

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from hm_agent.core.db import fetch_all, fetch_one
from hm_agent.core.logging_setup import configure_logging
from hm_agent.core.settings import REPO_ROOT

logger = logging.getLogger(__name__)

MAX_LIST_LIMIT = 10  # keeps list_orders replies short

mcp = FastMCP("order-management")


@mcp.tool()
def get_last_bought_product(customer_id: str) -> str:
    """Get the product name of the last item a customer bought.

    Args:
        customer_id: Id of the customer, e.g. ``cust001``.

    Returns:
        str: The product name, or a message saying no purchases were found.
    """
    row = fetch_one(
        """
        SELECT i.product_name
        FROM order_management.orders o
        JOIN order_management.order_items i USING (order_id)
        WHERE o.customer_id = %s
        ORDER BY o.order_date DESC, o.order_id DESC, i.line_no DESC
        LIMIT 1
        """,
        (customer_id,),
    )
    logger.info("tool=get_last_bought_product found=%s", row is not None)
    return row[0] if row else "No purchases found for this customer."


@mcp.tool()
def get_purchase_count(customer_id: str) -> str:
    """Get the total number of items a customer has purchased.

    Args:
        customer_id: Id of the customer, e.g. ``cust001``.

    Returns:
        str: The purchase count as text.
    """
    row = fetch_one(
        """
        SELECT COALESCE(SUM(i.quantity), 0)
        FROM order_management.orders o
        JOIN order_management.order_items i USING (order_id)
        WHERE o.customer_id = %s
        """,
        (customer_id,),
    )
    count = int(row[0]) if row else 0
    logger.info("tool=get_purchase_count count=%d", count)
    return f"{count} items purchased."


@mcp.tool()
def get_order_details(customer_id: str, order_id: str) -> str:
    """Get the date, sales channel and items of one of a customer's orders.

    The query filters on both ids, so an order belonging to another customer is reported as
    not found. Price, status and delivery columns are left out on purpose: price is a
    normalized value that must not be quoted, and status/delivery are not recorded per order.

    Args:
        customer_id: Id of the customer, e.g. ``cust001``.
        order_id: Id of the order, e.g. ``ORD-1007``.

    Returns:
        str: Order date, channel and product names, or a message saying the order was not found.
    """
    rows = fetch_all(
        """
        SELECT o.order_date, o.sales_channel, i.product_name
        FROM order_management.orders o
        JOIN order_management.order_items i USING (order_id)
        WHERE o.customer_id = %s AND o.order_id = %s
        ORDER BY i.line_no
        """,
        (customer_id, order_id),
    )
    logger.info("tool=get_order_details items=%d", len(rows))
    if not rows:
        return "No order with that id was found for this customer."
    order_date, channel, _ = rows[0]
    items = ", ".join(row[2] for row in rows)
    return f"Order {order_id} placed on {order_date} ({channel}), {len(rows)} item(s): {items}."


@mcp.tool()
def list_orders(customer_id: str, limit: int = 5) -> str:
    """List a customer's most recent orders with date, sales channel and item count.

    Args:
        customer_id: Id of the customer, e.g. ``cust001``.
        limit: Maximum number of orders to return (capped at ``MAX_LIST_LIMIT``).

    Returns:
        str: One line per order, newest first, or a message saying no orders were found.
    """
    limit = max(1, min(limit, MAX_LIST_LIMIT))
    rows = fetch_all(
        """
        SELECT o.order_id, o.order_date, o.sales_channel, COUNT(i.line_no)
        FROM order_management.orders o
        JOIN order_management.order_items i USING (order_id)
        WHERE o.customer_id = %s
        GROUP BY o.order_id, o.order_date, o.sales_channel
        ORDER BY o.order_date DESC, o.order_id DESC
        LIMIT %s
        """,
        (customer_id, limit),
    )
    logger.info("tool=list_orders orders=%d", len(rows))
    if not rows:
        return "No orders found for this customer."
    return "\n".join(f"{oid}: {date} ({channel}), {n} item(s)" for oid, date, channel, n in rows)


def main() -> None:
    """Start the server on the stdio transport.

    Logs go to stderr (the default handler stream) because stdout carries the MCP protocol.
    """
    # The server runs as a subprocess; load_dotenv does not override variables the parent set.
    load_dotenv(REPO_ROOT / ".env")
    configure_logging("mcp-order-management")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
