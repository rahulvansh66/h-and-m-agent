"""Golden dataset: load, validate and convert ``data/evals/golden.jsonl`` for Langfuse.

Role in the system: the 10-item golden set is the offline evaluation's source of truth and lives in git;
``scripts/langfuse/sync_dataset.py`` uploads it to a Langfuse dataset, and the runner can also feed it to an
experiment directly. Each line is ``{"id", "input": {question, customer_id}, "expected_output": {...},
"metadata": {suite, tags, ...}}``.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from hm_agent.core.config import REPO_ROOT

GOLDEN_PATH = REPO_ROOT / "data" / "evals" / "golden.jsonl"

# Suites decide how the offline task runs the item (see ``hm_agent.evals.tasks``).
SUITES = {"supervisor", "rag", "agent_tools", "guardrail"}


@dataclass(frozen=True)
class GoldenItem:
    """One golden example.

    Attributes:
        id (str): Stable id (also the Langfuse dataset item id, so re-syncing updates in place).
        question (str): The customer message.
        customer_id (str): Customer the message is sent as.
        expected (dict): Expected output: ``expected_agents``, ``reference``, ``expected_tools``,
            ``expected_tool_args``, ``expected_guardrail_action``, ``must_not_contain``, ``expected_chunks`` (gold chunk ids
            for retrieval metrics, ``source_file|p<page>|<kind>|<chunk_index>``).
        metadata (dict): ``suite``, optional ``agent`` and ``tags``.
    """

    id: str
    question: str
    customer_id: str
    expected: dict
    metadata: dict

    def as_dataset_item(self) -> dict:
        """Return the item in Langfuse dataset-item shape.

        Returns:
            dict: ``{"input", "expected_output", "metadata"}``; ``metadata`` also carries the item id.
        """
        return {
            "input": {"question": self.question, "customer_id": self.customer_id},
            "expected_output": self.expected,
            "metadata": {**self.metadata, "id": self.id},
        }


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldenItem]:
    """Read and validate the golden set.

    Args:
        path (Path): JSONL file to read.

    Returns:
        list[GoldenItem]: The items, in file order.

    Raises:
        ValueError: If an id repeats, a suite is unknown, or a required field is missing.
    """
    items: list[GoldenItem] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        try:
            item = GoldenItem(
                id=raw["id"],
                question=raw["input"]["question"],
                customer_id=raw["input"]["customer_id"],
                expected=raw["expected_output"],
                metadata=raw["metadata"],
            )
        except KeyError as exc:
            raise ValueError(f"{path.name}:{line_no} missing field {exc}") from exc
        if item.metadata.get("suite") not in SUITES:
            raise ValueError(f"{path.name}:{line_no} unknown suite {item.metadata.get('suite')!r}")
        if any(existing.id == item.id for existing in items):
            raise ValueError(f"{path.name}:{line_no} duplicate id {item.id!r}")
        items.append(item)
    return items
