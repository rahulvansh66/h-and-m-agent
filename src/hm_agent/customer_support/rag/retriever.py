"""Qdrant retriever: embeds a query locally and returns the closest knowledge-base chunks.

Role in the system: the retrieval step of the agentic RAG pipeline (``hm_agent.customer_support.rag.pipeline``).
The only module that talks to Qdrant.
"""

import logging

from hm_agent.core.settings import RAG_COLLECTION, RAG_EMBEDDING, RAG_TOP_K
from hm_agent.core.vector_store import get_client, get_embedder

logger = logging.getLogger(__name__)


def retrieve(query: str, top_k: int = RAG_TOP_K) -> list[dict]:
    """Return the ``top_k`` knowledge-base chunks most similar to the query.

    Args:
        query: Natural-language search query.
        top_k: Number of chunks to return.

    Returns:
        list[dict]: Chunks ordered by similarity, each with ``text``, ``doc_title``,
        ``source_file``, ``page`` and ``score``. Empty if nothing matched.
    """
    vector = next(iter(get_embedder(RAG_EMBEDDING).query_embed(query))).tolist()
    hits = get_client().query_points(
        collection_name=RAG_COLLECTION, query=vector, limit=top_k, with_payload=True
    ).points
    docs = [
        {
            "text": h.payload.get("text", ""),
            "doc_title": h.payload.get("doc_title", ""),
            "source_file": h.payload.get("source_file", ""),
            "page": h.payload.get("page", ""),
            "score": h.score,
        }
        for h in hits
    ]
    if not docs:
        logger.warning("step=retrieve status=empty collection=%s", RAG_COLLECTION)
    else:
        logger.info("step=retrieve docs=%d top_score=%.3f", len(docs), docs[0]["score"])
    return docs
