"""Create the Amazon Bedrock guardrail used by the H&M assistant (run once, by hand).

Role in the system: one-off setup. The guardrail has no running cost; it is billed per screened text unit when
the app calls ``ApplyGuardrail``. Run it only after local testing passes, then put the printed id in ``.env``
as ``BEDROCK_GUARDRAIL_ID`` and set ``guardrails.enabled: true`` (or env ``GUARDRAILS_ENABLED=true``).

Policies (kept to the minimum to keep the per-call cost low):
    * Content filters (input and output): HATE, INSULTS, SEXUAL, VIOLENCE, MISCONDUCT at HIGH strength.
    * Prompt attack filter (input only): prompt injection / jailbreak attempts.
    * Sensitive information: EMAIL, PHONE and ADDRESS are masked (ANONYMIZE); card numbers are BLOCKED.

Usage:
    uv run python scripts/aws/create_guardrail.py [--name hm-assistant-guardrail] [--region us-east-1]
"""

import argparse
import logging

import boto3
from dotenv import load_dotenv

from hm_agent.core.config import REPO_ROOT
from hm_agent.core.settings import GUARDRAIL_BLOCKED_REPLY, GUARDRAIL_REGION

logger = logging.getLogger("create_guardrail")

CONTENT_FILTERS = ["HATE", "INSULTS", "SEXUAL", "VIOLENCE", "MISCONDUCT"]
MASKED_PII = ["EMAIL", "PHONE", "ADDRESS"]
BLOCKED_PII = ["CREDIT_DEBIT_CARD_NUMBER"]


def build_request(name: str) -> dict:
    """Build the ``create_guardrail`` request.

    Args:
        name (str): Guardrail name.

    Returns:
        dict: Keyword arguments for ``bedrock.create_guardrail``.
    """
    filters = [{"type": t, "inputStrength": "HIGH", "outputStrength": "HIGH"} for t in CONTENT_FILTERS]
    # The prompt-attack filter only applies to input; its output strength must be NONE.
    filters.append({"type": "PROMPT_ATTACK", "inputStrength": "HIGH", "outputStrength": "NONE"})
    pii = [{"type": t, "action": "ANONYMIZE"} for t in MASKED_PII] + [{"type": t, "action": "BLOCK"} for t in BLOCKED_PII]
    return {
        "name": name,
        "description": "H&M assistant: toxicity, prompt-injection and PII protection",
        "contentPolicyConfig": {"filtersConfig": filters},
        "sensitiveInformationPolicyConfig": {"piiEntitiesConfig": pii},
        "blockedInputMessaging": GUARDRAIL_BLOCKED_REPLY,
        "blockedOutputsMessaging": GUARDRAIL_BLOCKED_REPLY,
    }


def main() -> None:
    """Create the guardrail and a numbered version, then print how to enable it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="hm-assistant-guardrail")
    parser.add_argument("--region", default=GUARDRAIL_REGION)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv(REPO_ROOT / ".env")

    client = boto3.client("bedrock", region_name=args.region)
    created = client.create_guardrail(**build_request(args.name))
    guardrail_id = created["guardrailId"]
    version = client.create_guardrail_version(guardrailIdentifier=guardrail_id, description="initial")["version"]
    logger.info("guardrail_id=%s version=%s region=%s", guardrail_id, version, args.region)
    logger.info("Add to .env:  BEDROCK_GUARDRAIL_ID=%s  BEDROCK_GUARDRAIL_VERSION=%s  GUARDRAILS_ENABLED=true", guardrail_id, version)


if __name__ == "__main__":
    main()
