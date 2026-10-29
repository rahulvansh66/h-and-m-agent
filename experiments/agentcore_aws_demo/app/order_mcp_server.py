"""Order MCP server: three read-only tools over the Neon Postgres `order_management` schema."""

import logging
import os
import sys

import psycopg
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="[order-mcp] %(message)s")
logger = logging.getLogger(__name__)

# Defaults are the local setup; on AgentCore Runtime Terraform sets HOST=0.0.0.0, PORT=8000, STATELESS_HTTP=true.
mcp = FastMCP(
    "order-management",
    host=os.environ.get("HOST", "127.0.0.1"),
    port=int(os.environ.get("PORT", "8001")),
    stateless_http=os.environ.get("STATELESS_HTTP", "").lower() == "true",
)


def fetch_all(sql: str, params: tuple) -> list[tuple]:
    """Run one read-only query on Neon and return all rows.

    A new connection per call keeps the server stateless (Runtime may start a fresh microVM for any request).

    Args:
        sql: SQL text with %s placeholders.
        params: Values for the placeholders (never string-formatted into the SQL).

    Returns:
        list[tuple]: The result rows.

    Raises:
        KeyError: If NEON_POSTGRES_CONNECTION_STRING is not set.
    """
    with psycopg.connect(os.environ["NEON_POSTGRES_CONNECTION_STRING"], connect_timeout=15) as conn:
        return conn.execute(sql, params).fetchall()


@mcp.tool()
def list_orders(customer_id: str) -> str:
    """List a customer's most recent orders (id, date, channel, number of items), newest first.

    Args:
        customer_id: Customer id such as cust001.
    """
    rows = fetch_all(
        """
        SELECT o.order_id, o.order_date, o.sales_channel, COUNT(i.line_no)
        FROM order_management.orders o JOIN order_management.order_items i USING (order_id)
        WHERE o.customer_id = %s
        GROUP BY o.order_id, o.order_date, o.sales_channel
        ORDER BY o.order_date DESC, o.order_id DESC LIMIT 5
        """,
        (customer_id,),
    )
    logger.info("tool=list_orders orders=%d", len(rows))
    if not rows:
        return "No orders found for this customer."
    return "\n".join(f"{oid}: {date} ({channel}), {n} item(s)" for oid, date, channel, n in rows)


@mcp.tool()
def get_order_details(customer_id: str, order_id: str) -> str:
    """Get the date, channel and items of one order. Only finds orders that belong to the given customer.

    Args:
        customer_id: Customer id such as cust001.
        order_id: Order id such as ORD-1007.
    """
    rows = fetch_all(
        """
        SELECT o.order_date, o.sales_channel, i.product_name
        FROM order_management.orders o JOIN order_management.order_items i USING (order_id)
        WHERE o.customer_id = %s AND o.order_id = %s
        ORDER BY i.line_no
        """,
        (customer_id, order_id.upper()),
    )
    logger.info("tool=get_order_details items=%d", len(rows))
    if not rows:
        return "No order with that id was found for this customer."
    order_date, channel, _ = rows[0]
    return f"Order {order_id.upper()} placed on {order_date} ({channel}), {len(rows)} item(s): {', '.join(r[2] for r in rows)}."


@mcp.tool()
def get_last_bought_product(customer_id: str) -> str:
    """Get the name of the last product a customer bought.

    Args:
        customer_id: Customer id such as cust001.
    """
    rows = fetch_all(
        """
        SELECT i.product_name
        FROM order_management.orders o JOIN order_management.order_items i USING (order_id)
        WHERE o.customer_id = %s
        ORDER BY o.order_date DESC, o.order_id DESC, i.line_no DESC LIMIT 1
        """,
        (customer_id,),
    )
    return rows[0][0] if rows else "No purchases found for this customer."


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
