"""Push the in-code prompt defaults to Langfuse (idempotent).

Role in the system: one-off / on-demand setup step. For each prompt in ``hm_agent.prompt_registry`` it creates
the prompt in Langfuse with label ``production`` if it does not exist, and adds a new version only when the
code default differs from the current ``production`` version. After the first sync, prompts are edited in the
Langfuse UI (new version, move the ``production`` label) and the app picks them up without a deploy.

Usage:
    uv run python scripts/langfuse/sync_prompts.py [--dry-run]
"""

import argparse
import logging

from dotenv import load_dotenv
from langfuse import get_client

from hm_agent.core.config import REPO_ROOT
from hm_agent.prompt_registry import PROMPTS, PromptSpec

logger = logging.getLogger("sync_prompts")

PRODUCTION = "production"


def _normalise(prompt: object) -> object:
    """Reduce a chat prompt to comparable ``(role, content)`` pairs; leave text prompts as they are.

    Args:
        prompt (object): A text string, or a list of messages (dicts or SDK message objects).

    Returns:
        object: The string unchanged, or a list of ``(role, content)`` tuples.
    """
    if isinstance(prompt, str):
        return prompt
    pairs = []
    for message in prompt:
        get = message.get if isinstance(message, dict) else lambda key, m=message: getattr(m, key, None)
        pairs.append((get("role"), get("content")))
    return pairs


def sync_one(spec: PromptSpec, dry_run: bool) -> str:
    """Create or update one prompt in Langfuse.

    Args:
        spec (PromptSpec): The prompt to sync.
        dry_run (bool): When True, only report what would change and write nothing.

    Returns:
        str: ``"created"``, ``"updated"`` or ``"unchanged"``.
    """
    langfuse = get_client()
    try:
        current = langfuse.get_prompt(spec.name, label=PRODUCTION, type=spec.kind, cache_ttl_seconds=0)
        if _normalise(current.prompt) == _normalise(spec.default):
            return "unchanged"
        outcome = "updated"
    except Exception:  # noqa: BLE001 - the SDK raises when the prompt or label does not exist yet
        outcome = "created"
    if not dry_run:
        langfuse.create_prompt(
            name=spec.name,
            type=spec.kind,
            prompt=spec.default,
            labels=[PRODUCTION],
            commit_message="Synced from in-code default",
        )
    return outcome


def main() -> None:
    """Sync every registered prompt and log one line per prompt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="show what would change without writing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv(REPO_ROOT / ".env")

    for spec in PROMPTS:
        logger.info("prompt=%s kind=%s status=%s dry_run=%s", spec.name, spec.kind, sync_one(spec, args.dry_run), args.dry_run)
    get_client().flush()


if __name__ == "__main__":
    main()
