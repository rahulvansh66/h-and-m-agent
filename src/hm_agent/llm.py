"""LLM factory: builds Groq chat models from the layered YAML config.

Role in the system: every agent/node gets its model here, so model names stay in config.
"""

from langchain_groq import ChatGroq

from hm_agent.config import load_config


def load_model(size: str = "small") -> ChatGroq:
    """Build the Groq chat model of the given size from the ``llm`` config section.

    Args:
        size: Key under ``llm`` in the config (``"small"`` or ``"big"``).

    Returns:
        ChatGroq: A deterministic (temperature 0) chat model.
    """
    model_name = load_config()["llm"][size]
    return ChatGroq(model=model_name, temperature=0)
