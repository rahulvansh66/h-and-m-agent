"""Shared Qdrant client and local embedding models.

Role in the system: connection plumbing for every agent that does vector search (customer support RAG,
fashion product search, catalog ingestion). Only the connection and model loading are shared; each agent
keeps its own search logic, collection, ``top_k`` and ``min_score``, and names its own embedding model in
its config section, so the models can be tuned independently.
"""

import os
from functools import cache

from fastembed import TextEmbedding
from qdrant_client import QdrantClient


@cache
def get_client() -> QdrantClient:
    """Build and cache the Qdrant client from ``QDRANT_ENDPOINT`` / ``QDRANT_API_KEY``.

    Returns:
        QdrantClient: Client connected to the configured Qdrant cluster.

    Raises:
        KeyError: If ``QDRANT_ENDPOINT`` or ``QDRANT_API_KEY`` is not set.
    """
    return QdrantClient(url=os.environ["QDRANT_ENDPOINT"], api_key=os.environ["QDRANT_API_KEY"])


@cache
def get_embedder(model_name: str) -> TextEmbedding:
    """Load and cache a local fastembed model by name (one cached instance per model).

    The model must be the one used to ingest the collection being searched, otherwise queries and
    documents live in different vector spaces.

    Args:
        model_name (str): fastembed model name, e.g. ``BAAI/bge-small-en-v1.5``; each agent passes the
            ``embedding`` value from its own config section.

    Returns:
        TextEmbedding: Local fastembed model (downloaded on first use).
    """
    return TextEmbedding(model_name)
