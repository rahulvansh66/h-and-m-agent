"""Upload the golden set (data/evals/golden.jsonl) to a Langfuse dataset (idempotent).

Role in the system: setup step for offline evaluation. The JSONL file in git is the source of truth; this script
creates the dataset ``evals.dataset`` (default ``hm-golden-v1``) if needed and upserts each item by its id, so
re-running updates items in place instead of duplicating them. To change the golden set in a way that should
not rewrite history, bump the dataset name in the config (``hm-golden-v2``) and run this again.

Usage:
    uv run python scripts/langfuse/sync_dataset.py [--dry-run]
"""

import argparse
import logging

from dotenv import load_dotenv
from langfuse import get_client

from hm_agent.core.config import REPO_ROOT
from hm_agent.core.settings import EVAL_DATASET
from hm_agent.evals.golden import load_golden

logger = logging.getLogger("sync_dataset")


def main() -> None:
    """Validate the golden file and upsert its items into the Langfuse dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="validate and list the items without writing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv(REPO_ROOT / ".env")

    items = load_golden()  # raises on a malformed file before anything is written
    if args.dry_run:
        for item in items:
            logger.info("item=%s suite=%s dry_run=True", item.id, item.metadata["suite"])
        return

    langfuse = get_client()
    langfuse.create_dataset(
        name=EVAL_DATASET,
        description="H&M assistant golden set: routing, tools, RAG and guardrail cases",
        metadata={"source": "data/evals/golden.jsonl"},
    )
    for item in items:
        langfuse.create_dataset_item(dataset_name=EVAL_DATASET, id=item.id, **item.as_dataset_item())
        logger.info("item=%s suite=%s status=upserted", item.id, item.metadata["suite"])
    langfuse.flush()
    logger.info("dataset=%s items=%d", EVAL_DATASET, len(items))


if __name__ == "__main__":
    main()
