"""Token usage and notional cost from LangChain usage metadata.

Role in the system: the supervisor turn, each agent run and the offline evaluation attach a
``UsageMetadataCallbackHandler`` to the run config, then turn what it collected into total tokens and a
notional USD cost using the Groq list prices in ``evals.pricing``. The Groq free tier is not billed; the cost
is what the same traffic would cost at list price, so cost regressions are visible before paying for them.
"""

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.runnables import RunnableConfig

from hm_agent.core.settings import EVAL_PRICING

TOKENS_PER_MILLION = 1_000_000


def usage_handler(config: RunnableConfig) -> UsageMetadataCallbackHandler | None:
    """Find the usage handler attached to a run config.

    Args:
        config (RunnableConfig): Config built by ``build_langfuse_config``.

    Returns:
        UsageMetadataCallbackHandler | None: The handler, or None when the config has none.
    """
    for handler in config.get("callbacks", []) or []:
        if isinstance(handler, UsageMetadataCallbackHandler):
            return handler
    return None


def summarise_usage(usage_by_model: dict[str, dict]) -> dict[str, float]:
    """Total the tokens and notional cost over every model used in a run.

    Args:
        usage_by_model (dict[str, dict]): ``handler.usage_metadata``: model name -> ``input_tokens`` /
            ``output_tokens`` / ``total_tokens``.

    Returns:
        dict[str, float]: ``input_tokens``, ``output_tokens``, ``total_tokens`` and ``cost_usd``. Models missing
        from ``evals.pricing`` add tokens but no cost.
    """
    totals = {"input_tokens": 0.0, "output_tokens": 0.0, "total_tokens": 0.0, "cost_usd": 0.0}
    for model, usage in usage_by_model.items():
        tokens_in, tokens_out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        totals["input_tokens"] += tokens_in
        totals["output_tokens"] += tokens_out
        totals["total_tokens"] += usage.get("total_tokens", tokens_in + tokens_out)
        price = EVAL_PRICING.get(model)
        if price:
            totals["cost_usd"] += (tokens_in * price["input"] + tokens_out * price["output"]) / TOKENS_PER_MILLION
    return totals
