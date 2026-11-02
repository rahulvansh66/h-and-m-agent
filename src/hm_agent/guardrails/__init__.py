"""Amazon Bedrock Guardrails integration: toxicity, PII masking and prompt-injection screening."""

from hm_agent.guardrails.client import GuardrailResult, screen

__all__ = ["GuardrailResult", "screen"]
