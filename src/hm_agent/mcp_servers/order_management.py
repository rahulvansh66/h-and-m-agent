"""MCP server exposing read-only SQL tools on the Neon ``order_management`` schema.

Role in the system: tool provider for the order_management agent. The agent talks to this
server over MCP (stdio transport, spawned on demand by the client), so no always-on process
is needed. Each tool takes a ``customer_id``; the MCP client overwrites it from the run config
(see ``hm_agent.mcp_client``), so the LLM never chooses which customer's data is read.

Run standalone (for inspection with an MCP client):
    uv run python -m hm_agent.mcp_servers.order_management
"""

import logging

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from hm_agent.db import fetch_one
from hm_agent.logging_setup import configure_logging
from hm_agent.settings import REPO_ROOT

logger = logging.getLogger(__name__)

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
