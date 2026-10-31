"""Build the Qdrant collection used by the fashion agent from the Neon ``product_catalog`` table.

Run (from the repo root):
    uv run python scripts/fashion/ingest_catalog.py

Each product is embedded once, from one text string that carries what a customer's words can match
(name, type, colour, department, description). Hard-filter fields (product_group, index_group, ...) go in the
payload instead, so they are applied as exact filters at query time. The collection is recreated on every
run, so the script is idempotent.
"""

import logging

from dotenv import load_dotenv
from qdrant_client import models

from hm_agent.core.db import fetch_all
from hm_agent.core.logging_setup import configure_logging
from hm_agent.core.settings import FASHION_COLLECTION, FASHION_EMBEDDING, REPO_ROOT
from hm_agent.core.vector_store import get_client, get_embedder

logger = logging.getLogger(__name__)

# Payload fields the search filters on (see hm_agent.fashion.mcp_server.search_products).
FILTER_FIELDS = ("product_group", "product_id", "index_group")


def build_text(name: str, product_type: str, colour: str, department: str, description: str | None) -> str:
    """Build the text that is embedded for one product.

    Args:
        name: Product name.
        product_type: Fine-grained type, e.g. ``Blazer``.
        colour: Colour name.
        department: Department, a soft hint for style or occasion (e.g. ``Suit``).
        description: Free-text description; may be empty.

    Returns:
        str: ``"{name}. {type} in {colour}. {department}. {description}"`` without empty parts.
    """
    parts = [f"{name}.", f"{product_type} in {colour}.", f"{department}.", f"{description}" if description else ""]
    return " ".join(p for p in parts if p)


def main() -> None:
    """Read the catalog from Neon, embed it locally and (re)create the Qdrant collection."""
    load_dotenv(REPO_ROOT / ".env")
    configure_logging("ingest-fashion")
    rows = fetch_all(
        """
        SELECT product_id, product_name, product_type, product_group, colour, department, index_group, description
        FROM product_recommendation.product_catalog
        ORDER BY product_id
        """,
        (),
    )
    texts = [build_text(r[1], r[2], r[4], r[5], r[7]) for r in rows]
    vectors = [v.tolist() for v in get_embedder(FASHION_EMBEDDING).embed(texts)]

    client = get_client()
    if client.collection_exists(FASHION_COLLECTION):
        client.delete_collection(FASHION_COLLECTION)
    client.create_collection(
        FASHION_COLLECTION,
        vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
    )
    # Qdrant Cloud refuses to filter on a payload field that has no index.
    for field in FILTER_FIELDS:
        client.create_payload_index(FASHION_COLLECTION, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD)
    client.upsert(
        FASHION_COLLECTION,
        points=[
            models.PointStruct(
                id=i,
                vector=vector,
                payload={
                    "product_id": r[0],
                    "product_type": r[2],
                    "product_group": r[3],
                    "colour": r[4],
                    "index_group": r[6],
                },
            )
            for i, (r, vector) in enumerate(zip(rows, vectors), start=1)
        ],
    )
    logger.warning("step=ingest status=complete collection=%s points=%d", FASHION_COLLECTION, len(rows))


if __name__ == "__main__":
    main()
