"""Tiny Groq chat helper (OpenAI-compatible API over httpx): one function, no SDK, nothing logged but aggregates."""

import logging
import os
import time

import httpx

logger = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-20b"   # the project's "small" model (config/base.yaml); Terraform passes GROQ_MODEL on AWS


def groq_enabled() -> bool:
    """Tell whether an LLM key is configured; callers fall back to keyword rules when it is not.

    Returns:
        bool: True if GROQ_API_KEY is set and non-empty.
    """
    return bool(os.environ.get("GROQ_API_KEY", "").strip())


async def chat(messages: list[dict], tools: list[dict] | None = None, json_mode: bool = False) -> dict:
    """Send one chat request to Groq and return the assistant message.

    Free-tier friendly: temperature 0 and low reasoning effort keep the small model fast and cheap.
    Only model name, token counts and latency are logged, never the prompt or the reply (they can hold customer data).

    Args:
        messages: OpenAI-style messages (`{"role": ..., "content": ...}`).
        tools: Optional function specs; if given, the model may answer with `tool_calls` instead of text.
        json_mode: Force the reply to be a JSON object (the prompt must mention the word JSON).

    Returns:
        dict: The assistant message: `content` (text) and, when the model chose a tool, `tool_calls`.

    Raises:
        httpx.HTTPError: On network errors or a non-2xx status (for example 429 when the free-tier limit is hit).
    """
    model = os.environ.get("GROQ_MODEL") or DEFAULT_MODEL
    payload: dict = {"model": model, "messages": messages, "temperature": 0, "reasoning_effort": "low"}
    if tools:
        payload.update(tools=tools, tool_choice="auto")
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=30) as http:
        response = await http.post(GROQ_URL, json=payload, headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"})
        response.raise_for_status()
    data = response.json()
    usage = data.get("usage", {})
    logger.info("llm model=%s prompt_tokens=%s completion_tokens=%s duration_s=%.2f",
                model, usage.get("prompt_tokens"), usage.get("completion_tokens"), time.perf_counter() - start)
    return data["choices"][0]["message"]
