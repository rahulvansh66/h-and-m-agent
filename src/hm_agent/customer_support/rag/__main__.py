"""Command-line entry point for the agentic RAG pipeline.

Run (from the repo root):
    uv run python -m hm_agent.customer_support.rag "What happens if my package can't be delivered?"

Needs GROQ_API_KEY, QDRANT_ENDPOINT and QDRANT_API_KEY in ``.env``.
"""

import argparse
import sys

from dotenv import load_dotenv
from langfuse import get_client

from hm_agent.core.logging_setup import configure_logging
from hm_agent.customer_support.rag.pipeline import build_rag_graph
from hm_agent.core.settings import DEFAULT_CUSTOMER_ID, REPO_ROOT
from hm_agent.core.tracing import build_langfuse_config


def main() -> None:
    """CLI entry point: answer one question from the knowledge base with citations."""
    # Model replies can contain non-cp1252 characters (e.g. non-breaking hyphens) on Windows consoles.
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(REPO_ROOT / ".env")
    configure_logging("rag")

    parser = argparse.ArgumentParser(description="H&M agentic RAG")
    parser.add_argument("query", help="Question to answer from the help articles")
    args = parser.parse_args()

    # Traced like a real turn, so the RAG prompts and retrieval show up in Langfuse when run standalone.
    config = build_langfuse_config(DEFAULT_CUSTOMER_ID, "rag-cli", "rag-cli", extra_tags=["rag-cli"])
    result = build_rag_graph().invoke({"query": args.query}, config)
    print(result["answer"])
    get_client().flush()


if __name__ == "__main__":
    main()
