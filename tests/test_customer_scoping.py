"""Tests that a customer can only reach their own data (authorization, not just authentication).

Role in the system: covers every data query path of the order_management, fashion and shipment_tracking
agents. The tool interceptor test needs no database; the rest read the seeded test database through the
``db`` fixture (never the main one).

Query paths and their coverage:

* order_management ``get_last_bought_product``, ``get_purchase_count``, ``list_orders``,
  ``get_order_details``: filter on ``customer_id``  -> ``test_order_tools_return_only_own_data``
* fashion ``get_purchase_history``, ``get_style_profile``, ``get_styling_guide`` (via ``_resolve_segment``):
  filter on ``customer_id``                          -> ``test_fashion_tools_return_only_own_data``
* fashion ``search_products``: catalog search, reads no customer data (nothing to scope)
* shipment_tracking carrier lookup: gated by ``order_belongs_to_customer`` -> ``test_shipment_*``
* the ``customer_id`` argument itself is overwritten from the run config -> ``test_interceptor_*``
"""

from types import SimpleNamespace

import pytest

from hm_agent.core import mcp_tools
from hm_agent.core.db import fetch_all
from hm_agent.fashion import mcp_server as fashion_server
from hm_agent.order_management import mcp_server as order_server
from hm_agent.shipment_tracking import agent as shipment_agent
from hm_agent.shipment_tracking.tools import order_belongs_to_customer


@pytest.fixture
def seeded(db, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Point the data layer at the test database and pick two customers who both have orders.

    Args:
        db (str): DSN of the test database (from the ``db`` fixture).
        monkeypatch (pytest.MonkeyPatch): Used to redirect the data layer to the test DSN.

    Returns:
        dict[str, str]: ``{"a": customer, "b": other customer, "a_order": order of a, "b_order": order of b}``.
    """
    monkeypatch.setenv("NEON_POSTGRES_CONNECTION_STRING", db)
    rows = fetch_all(
        "SELECT customer_id, MIN(order_id) FROM order_management.orders GROUP BY customer_id ORDER BY customer_id", ()
    )
    if len(rows) < 2:
        pytest.skip("test database needs orders for at least two customers")
    (a, a_order), (b, b_order) = rows[0], rows[1]
    return {"a": a, "b": b, "a_order": a_order, "b_order": b_order}


def test_order_tools_return_only_own_data(seeded) -> None:
    """Customer A cannot read customer B's order, and listings contain only their own orders."""
    a, b = seeded["a"], seeded["b"]
    assert "No order" in order_server.get_order_details(a, seeded["b_order"])
    assert seeded["a_order"] in order_server.get_order_details(a, seeded["a_order"])
    assert seeded["b_order"] not in order_server.list_orders(a, limit=10)
    assert seeded["a_order"] not in order_server.list_orders(b, limit=10)
    # Aggregates are per customer too: A's count must not include B's items.
    total = fetch_all("SELECT COALESCE(SUM(quantity), 0) FROM order_management.order_items", ())[0][0]
    assert order_server.get_purchase_count(a) != f"{total} items purchased."
    assert order_server.get_last_bought_product(a) != order_server.get_last_bought_product("no-such-customer")


def test_fashion_tools_return_only_own_data(seeded) -> None:
    """Purchase history and profile come only from the requested customer's rows."""
    a, b = seeded["a"], seeded["b"]
    owned = lambda cust: {  # noqa: E731  product ids a customer really bought
        r[0] for r in fetch_all("SELECT product_id FROM product_recommendation.purchase_history WHERE customer_id = %s", (cust,))
    }
    only_b = owned(b) - owned(a)
    history_a = fashion_server.get_purchase_history(a, limit=10)
    assert all(pid not in history_a for pid in only_b)
    assert "No purchases" in fashion_server.get_purchase_history("no-such-customer")
    assert "No style profile" in fashion_server.get_style_profile("no-such-customer")
    assert fashion_server._resolve_segment("no-such-customer") is None


def test_shipment_order_gate(seeded) -> None:
    """Only the owner passes the ownership gate; other customers, unknown orders and empty ids do not."""
    a, b = seeded["a"], seeded["b"]
    assert order_belongs_to_customer(seeded["a_order"], a)
    assert order_belongs_to_customer(seeded["a_order"].lower(), a)
    assert not order_belongs_to_customer(seeded["b_order"], a)
    assert not order_belongs_to_customer(seeded["a_order"], b)
    assert not order_belongs_to_customer("ORD-0000", a)
    assert not order_belongs_to_customer(seeded["a_order"], "")


def test_shipment_runner_refuses_other_customers_order(seeded, monkeypatch: pytest.MonkeyPatch) -> None:
    """Asking about someone else's order gets the generic reply and never reaches the carrier lookup."""
    called: list[str] = []
    monkeypatch.setattr(shipment_agent, "stream_carrier_status", lambda order_id: called.append(order_id) or iter(()))
    chunks = list(shipment_agent.shipment_runner(f"where is {seeded['b_order']}?", [], seeded["a"], "t1"))
    assert [c.text for c in chunks] == [shipment_agent.ORDER_NOT_FOUND_REPLY]
    assert called == []
    list(shipment_agent.shipment_runner(f"where is {seeded['a_order']}?", [], seeded["a"], "t1"))
    assert called == [seeded["a_order"]]


class FakeRequest:
    """Minimal stand-in for ``MCPToolCallRequest``: carries args and a runtime config."""

    def __init__(self, args: dict, configurable: dict) -> None:
        """Create the request.

        Args:
            args (dict): Tool arguments, as the model supplied them.
            configurable (dict): The run config's ``configurable`` section.
        """
        self.args = args
        self.runtime = SimpleNamespace(config={"configurable": configurable})

    def override(self, args: dict) -> "FakeRequest":
        """Return a copy with replaced arguments.

        Args:
            args (dict): New tool arguments.

        Returns:
            FakeRequest: The modified request.
        """
        return FakeRequest(args, self.runtime.config["configurable"])


async def echo_args(request: FakeRequest) -> dict:
    """Handler that returns the arguments it would send to the MCP server.

    Args:
        request (FakeRequest): The (already modified) request.

    Returns:
        dict: The arguments.
    """
    return request.args


@pytest.mark.asyncio
async def test_interceptor_overwrites_model_supplied_customer_id() -> None:
    """A customer_id chosen by the model is replaced by the one from the run config."""
    request = FakeRequest({"customer_id": "cust999", "limit": 3}, {"customer_id": "cust001"})
    assert await mcp_tools.inject_customer_id(request, echo_args) == {"customer_id": "cust001", "limit": 3}


@pytest.mark.asyncio
async def test_interceptor_has_no_default_fallback_when_auth_is_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """With auth on, a missing customer id is an error, never the default customer's data."""
    monkeypatch.setattr(mcp_tools, "auth_enabled", lambda: True)
    with pytest.raises(PermissionError):
        await mcp_tools.inject_customer_id(FakeRequest({}, {}), echo_args)


@pytest.mark.asyncio
async def test_interceptor_falls_back_to_default_customer_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    """With auth off (local default) the CLI default customer is used, as before."""
    monkeypatch.setattr(mcp_tools, "auth_enabled", lambda: False)
    result = await mcp_tools.inject_customer_id(FakeRequest({}, {}), echo_args)
    assert result["customer_id"] == mcp_tools.DEFAULT_CUSTOMER_ID
