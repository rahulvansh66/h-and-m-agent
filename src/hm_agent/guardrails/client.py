"""Amazon Bedrock Guardrails client: screens text for toxicity, prompt attacks and PII.

Role in the system: used by ``hm_agent.turns`` to screen the customer message before the graph runs (input)
and the final reply before it is returned (output), and by the offline evaluation to score the guardrail.
It calls the model-independent ``ApplyGuardrail`` API, so no Bedrock model is invoked: the cost is per
screened text unit. The result reports what happened (``NONE`` / ``MASKED`` / ``BLOCKED``) and which
categories fired; the text itself is never logged, only category names and counts.
"""

import logging
import time
from dataclasses import dataclass, field
from functools import cache
from typing import Any, Literal

import boto3

from hm_agent.core.settings import GUARDRAIL_REGION, guardrail_identity

logger = logging.getLogger(__name__)

Source = Literal["INPUT", "OUTPUT"]
Action = Literal["NONE", "MASKED", "BLOCKED"]

# Content-filter types that count as toxic / harmful language; PROMPT_ATTACK is reported separately.
TOXICITY_FILTERS = {"HATE", "INSULTS", "SEXUAL", "VIOLENCE", "MISCONDUCT"}
PROMPT_ATTACK_FILTER = "PROMPT_ATTACK"


@dataclass(frozen=True)
class GuardrailResult:
    """Outcome of screening one piece of text.

    Attributes:
        action (Action): ``NONE`` (clean), ``MASKED`` (PII replaced, text still usable) or ``BLOCKED``.
        text (str): Text to use from here on: the masked text for ``MASKED``, otherwise the original.
        toxicity (bool): A toxicity filter (hate, insults, sexual, violence, misconduct) fired.
        prompt_attack (bool): The prompt-attack filter fired (input only).
        pii_entities (dict[str, int]): PII entity type -> number found, e.g. ``{"EMAIL": 1}``.
        latency_s (float): Duration of the ApplyGuardrail call.
        error (bool): The call failed; ``action`` then follows the fail-open / fail-closed setting.
    """

    action: Action = "NONE"
    text: str = ""
    toxicity: bool = False
    prompt_attack: bool = False
    pii_entities: dict[str, int] = field(default_factory=dict)
    latency_s: float = 0.0
    error: bool = False

    @property
    def pii_count(self) -> int:
        """Return the total number of PII entities found.

        Returns:
            int: Sum over ``pii_entities``.
        """
        return sum(self.pii_entities.values())


@cache
def _client() -> Any:
    """Build the ``bedrock-runtime`` client once (uses the default AWS credential chain).

    Returns:
        Any: A boto3 ``bedrock-runtime`` client.
    """
    return boto3.client("bedrock-runtime", region_name=GUARDRAIL_REGION)


def _parse(response: dict, original: str, latency_s: float) -> GuardrailResult:
    """Turn an ``ApplyGuardrail`` response into a ``GuardrailResult``.

    Args:
        response (dict): Raw API response (``action``, ``outputs``, ``assessments``).
        original (str): The text that was screened.
        latency_s (float): Duration of the call.

    Returns:
        GuardrailResult: ``BLOCKED`` when any filter or PII entity was blocked, ``MASKED`` when PII was only
        anonymised, ``NONE`` when the guardrail did not intervene.
    """
    toxicity = prompt_attack = blocked = False
    pii: dict[str, int] = {}
    for assessment in response.get("assessments", []):
        for flt in assessment.get("contentPolicy", {}).get("filters", []):
            if flt.get("detected", True) is False:
                continue
            if flt.get("type") in TOXICITY_FILTERS:
                toxicity = True
            elif flt.get("type") == PROMPT_ATTACK_FILTER:
                prompt_attack = True
            blocked = blocked or flt.get("action") == "BLOCKED"
        for entity in assessment.get("sensitiveInformationPolicy", {}).get("piiEntities", []):
            pii[entity["type"]] = pii.get(entity["type"], 0) + 1
            blocked = blocked or entity.get("action") == "BLOCKED"
        for regex in assessment.get("sensitiveInformationPolicy", {}).get("regexes", []):
            pii[regex["name"]] = pii.get(regex["name"], 0) + 1
            blocked = blocked or regex.get("action") == "BLOCKED"

    if response.get("action") != "GUARDRAIL_INTERVENED":
        return GuardrailResult("NONE", original, latency_s=latency_s)
    if blocked:
        return GuardrailResult("BLOCKED", original, toxicity, prompt_attack, pii, latency_s)
    masked = "".join(o.get("text", "") for o in response.get("outputs", [])) or original
    return GuardrailResult("MASKED", masked, toxicity, prompt_attack, pii, latency_s)


def screen(text: str, source: Source, *, fail_closed: bool) -> GuardrailResult:
    """Screen text with the configured guardrail.

    Args:
        text (str): Customer message (``INPUT``) or final reply (``OUTPUT``). Never logged.
        source (Source): Which side is being screened; prompt-attack detection applies to ``INPUT`` only.
        fail_closed (bool): On an API error, return ``BLOCKED`` (True) or let the text through as ``NONE`` (False).

    Returns:
        GuardrailResult: The outcome. Never raises on AWS errors; ``error`` is set instead.
    """
    start = time.perf_counter()
    try:
        guardrail_id, version = guardrail_identity()
        response = _client().apply_guardrail(
            guardrailIdentifier=guardrail_id,
            guardrailVersion=version,
            source=source,
            content=[{"text": {"text": text}}],
        )
    except Exception:  # noqa: BLE001 - AWS / config errors must degrade per the fail-open / fail-closed setting
        latency_s = time.perf_counter() - start
        logger.exception("guardrail source=%s status=error fail_closed=%s", source, fail_closed)
        return GuardrailResult("BLOCKED" if fail_closed else "NONE", text, latency_s=latency_s, error=True)
    result = _parse(response, text, time.perf_counter() - start)
    logger.info(
        "guardrail source=%s action=%s toxicity=%s prompt_attack=%s pii_entities=%d latency_s=%.2f",
        source, result.action, result.toxicity, result.prompt_attack, result.pii_count, result.latency_s,
    )
    return result
