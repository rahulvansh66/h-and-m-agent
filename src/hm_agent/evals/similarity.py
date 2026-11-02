"""Embedding similarity between texts, using the app's own local fastembed model (no API cost).

Role in the system: the free, fast answer-quality signals. ``answer_relevancy_sim`` (question vs reply) is the
cheap relevancy proxy used both online (every turn) and offline; ``answer_similarity`` (reply vs golden answer)
is the offline closeness-to-golden check. They are cosine similarities of sentence embeddings: good for spotting
regressions and off-topic or wrong answers, not a replacement for the LLM-judged RAGAS metrics, and the absolute
values depend on the embedding model (compare runs, do not read 0.7 as 70%).
"""

import math

from hm_agent.core.settings import RAG_EMBEDDING
from hm_agent.core.vector_store import get_embedder


def cosine_similarity(a: str, b: str) -> float:
    """Return the cosine similarity of two texts' embeddings.

    Args:
        a (str): First text.
        b (str): Second text.

    Returns:
        float: Cosine similarity in [-1, 1]; 0.0 if either text is empty.
    """
    if not a.strip() or not b.strip():
        return 0.0
    va, vb = (list(map(float, v)) for v in get_embedder(RAG_EMBEDDING).embed([a, b]))
    norm = math.sqrt(sum(x * x for x in va)) * math.sqrt(sum(x * x for x in vb))
    return sum(x * y for x, y in zip(va, vb)) / norm if norm else 0.0
