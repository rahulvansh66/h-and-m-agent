"""MCP server for the fashion agent: customer taste, purchases, semantic product search and styling guides.

Role in the system: tool provider for the fashion agent. Unlike the order server (stdio, spawned per call)
this one is long-lived and served over streamable HTTP, so the local embedding model and the Qdrant client
load once instead of on every ``search_products`` call. Each tool takes a ``customer_id``; the MCP client
overwrites it from the run config (see ``hm_agent.fashion.mcp_client``), so the LLM never chooses whose data is read.

Run standalone (``scripts/a2a/run_agents.py`` starts it with the other services):
    uv run python -m hm_agent.fashion.mcp_server
"""

import logging
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from qdrant_client import models

from hm_agent.core.db import fetch_all, fetch_one
from hm_agent.core.logging_setup import configure_logging
from hm_agent.core.settings import (
    FASHION_COLLECTION,
    FASHION_EMBEDDING,
    FASHION_MCP_HOST,
    FASHION_MCP_PORT,
    FASHION_MIN_SCORE,
    FASHION_TOP_K,
    REPO_ROOT,
)
from hm_agent.core.vector_store import get_client, get_embedder

logger = logging.getLogger(__name__)

MAX_LIST_LIMIT = 10  # keeps purchase-history replies short
GUIDES_DIR = Path(__file__).parent / "guides"
# Not outfit pieces: always filtered out of searches unless the customer explicitly asks for them.
NON_OUTFIT_GROUPS = ["Swimwear", "Underwear"]
# index_group values that say something about the customer's segment; Divided, Sport, ... are not counted.
SEGMENT_GROUPS = {"Ladieswear": "women", "Menswear": "men"}

mcp = FastMCP("fashion", host=FASHION_MCP_HOST, port=FASHION_MCP_PORT)


def _read_guide(segment: str) -> str:
    """Read one styling guide file.

    Args:
        segment: ``women`` or ``men``.

    Returns:
        str: The guide text.
    """
    return (GUIDES_DIR / f"{segment}.md").read_text(encoding="utf-8").strip()


def _resolve_segment(customer_id: str) -> str | None:
    """Derive the styling segment from the customer's purchases.

    Counts purchases per ``index_group`` (Ladieswear vs Menswear only). The segment is the larger side;
    a tie or no signal gives ``None``.

    Args:
        customer_id: Id of the customer, e.g. ``cust001``.

    Returns:
        str | None: ``women``, ``men`` or ``None`` when there is no reliable signal.
    """
    rows = fetch_all(
        """
        SELECT c.index_group, COUNT(*)
        FROM product_recommendation.purchase_history p
        JOIN product_recommendation.product_catalog c USING (product_id)
        WHERE p.customer_id = %s AND c.index_group = ANY(%s)
        GROUP BY c.index_group
        """,
        (customer_id, list(SEGMENT_GROUPS)),
    )
    counts = {SEGMENT_GROUPS[group]: n for group, n in rows}
    women, men = counts.get("women", 0), counts.get("men", 0)
    if women == men:
        return None
    return "women" if women > men else "men"


@mcp.resource("styling://guide/women")
def women_guide() -> str:
    """Styling rules of thumb for women's outfits."""
    return _read_guide("women")


@mcp.resource("styling://guide/men")
def men_guide() -> str:
    """Styling rules of thumb for men's outfits."""
    return _read_guide("men")


@mcp.tool()
def get_style_profile(customer_id: str) -> str:
    """Get the customer's taste profile: age, favourite product group and number of purchases.

    Args:
        customer_id: Id of the customer, e.g. ``cust001``.

    Returns:
        str: A one-line profile, or a message saying none was found.
    """
    row = fetch_one(
        """
        SELECT age, top_product_group, purchase_count
        FROM personalization.personalization
        WHERE customer_id = %s
        """,
        (customer_id,),
    )
    logger.info("tool=get_style_profile found=%s", row is not None)
    if not row:
        return "No style profile found for this customer."
    age, top_group, purchase_count = row
    return f"Age {age}; most bought product group: {top_group}; {purchase_count} purchases."


@mcp.tool()
def get_purchase_history(customer_id: str, limit: int = 5) -> str:
    """List the products a customer bought most recently, with their catalog details.

    Use it to resolve "the jacket I bought last time" and to avoid recommending items already owned.

    Args:
        customer_id: Id of the customer, e.g. ``cust001``.
        limit: Maximum number of purchases to return (capped at ``MAX_LIST_LIMIT``).

    Returns:
        str: One line per purchase, newest first, or a message saying none were found.
    """
    limit = max(1, min(limit, MAX_LIST_LIMIT))
    rows = fetch_all(
        """
        SELECT c.product_id, c.product_name, c.product_type, c.product_group, c.colour, p.purchase_date
        FROM product_recommendation.purchase_history p
        JOIN product_recommendation.product_catalog c USING (product_id)
        WHERE p.customer_id = %s
        ORDER BY p.purchase_date DESC, p.id DESC
        LIMIT %s
        """,
        (customer_id, limit),
    )
    logger.info("tool=get_purchase_history purchases=%d", len(rows))
    if not rows:
        return "No purchases found for this customer."
    return "\n".join(
        f"{pid}: {name} ({ptype}, {group}, {colour}), bought {date}" for pid, name, ptype, group, colour, date in rows
    )


@mcp.tool()
def get_styling_guide(customer_id: str) -> str:
    """Get the short styling guide that fits the customer, to follow when pairing items into outfits.

    The segment is derived from the customer's purchases, never chosen by the caller.

    Args:
        customer_id: Id of the customer, e.g. ``cust001``.

    Returns:
        str: The women's or men's guide, or neutral advice when the segment cannot be told.
    """
    segment = _resolve_segment(customer_id)
    logger.info("tool=get_styling_guide segment=%s", segment)
    if segment is None:
        return "No segment known for this customer: give neutral advice and do not assume a gender."
    return _read_guide(segment)


@mcp.tool()
def search_products(
    customer_id: str,
    query: str,
    exclude_product_groups: list[str] | None = None,
    exclude_product_ids: list[str] | None = None,
    allow_swimwear_underwear: bool = False,
) -> str:
    """Search the product catalog by meaning (style, occasion, item type) and return real products.

    The query is embedded and matched against the catalog vectors; payload filters then enforce hard
    rules. Swimwear and underwear are excluded unless explicitly allowed. Results below the relevance
    threshold are dropped, and product details come from the database, not the vector store.

    Args:
        customer_id: Id of the customer (unused by the search itself; set by the system).
        query: Natural-language description, e.g. ``bottom to pair with a slim black jacket, casual``.
        exclude_product_groups: Product groups to leave out, e.g. ``["Garment Upper body"]`` for "not another top".
        exclude_product_ids: Product ids to leave out, e.g. items the customer already owns.
        allow_swimwear_underwear: True only if the customer explicitly asked for swimwear or underwear.

    Returns:
        str: One line per matching product, best first, or a message saying there is no good match.
    """
    excluded_groups = list(exclude_product_groups or [])
    if not allow_swimwear_underwear:
        excluded_groups += [g for g in NON_OUTFIT_GROUPS if g not in excluded_groups]
    must_not = []
    if excluded_groups:
        must_not.append(models.FieldCondition(key="product_group", match=models.MatchAny(any=excluded_groups)))
    if exclude_product_ids:
        must_not.append(models.FieldCondition(key="product_id", match=models.MatchAny(any=exclude_product_ids)))

    vector = next(iter(get_embedder(FASHION_EMBEDDING).query_embed(query))).tolist()
    hits = get_client().query_points(
        collection_name=FASHION_COLLECTION,
        query=vector,
        query_filter=models.Filter(must_not=must_not),
        limit=FASHION_TOP_K,
        with_payload=True,
    ).points
    hits = [h for h in hits if h.score >= FASHION_MIN_SCORE]
    if not hits:
        logger.warning("tool=search_products status=no_match collection=%s", FASHION_COLLECTION)
        return "No good match found in the catalog."

    ids = [h.payload["product_id"] for h in hits]
    rows = fetch_all(
        """
        SELECT product_id, product_name, product_type, product_group, colour, department, description
        FROM product_recommendation.product_catalog
        WHERE product_id = ANY(%s)
        """,
        (ids,),
    )
    by_id = {row[0]: row for row in rows}
    logger.info("tool=search_products matches=%d top_score=%.3f", len(hits), hits[0].score)
    lines = []
    for hit in hits:
        row = by_id.get(hit.payload["product_id"])
        if row is None:  # vector store is ahead of the database: skip rather than invent details
            continue
        pid, name, ptype, group, colour, department, description = row
        line = f"{pid}: {name} ({ptype}, {group}, {colour}, {department}), match {hit.score:.2f}"
        lines.append(f"{line}. {description}" if description else line)
    return "\n".join(lines) or "No good match found in the catalog."


def main() -> None:
    """Start the server on the streamable HTTP transport and warm the embedding model first."""
    load_dotenv(REPO_ROOT / ".env")
    configure_logging("mcp-fashion")
    get_embedder(FASHION_EMBEDDING)  # load once at startup so the first search is not slow
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
