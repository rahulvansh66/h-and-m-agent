"""DeepEval metrics for the golden set: tone, policy adherence, tool correctness, hallucination, safety.

Role in the system: the judge metrics of the offline evaluation that RAGAS does not cover (``--metrics deepeval``).
Names are prefixed ``deepeval_`` so they never collide with RAGAS or deterministic metrics.

    deepeval_helpfulness_tone   G-Eval: helpful, polite, concise, brand voice, no invented facts
    deepeval_policy_adherence   G-Eval: reply agrees with the reference return / replacement guidance
    deepeval_tool_correctness   did the agent call the expected tools
    deepeval_no_hallucination  1.0 = the reply does not contradict the retrieved contexts
    deepeval_prompt_alignment   does the reply follow the rules of the (versioned) RAG answer prompt
    deepeval_non_toxic, deepeval_unbiased  1.0 = clean (this DeepEval version scores higher-is-better); guardrail items and the fashion agent only

Judge calls are limited to the items each metric applies to, and reasons are switched off, to stay within the
Groq free tier. A metric that errors yields a ``None`` score with the error name instead of failing the run.
"""

import logging
import re
from typing import Any

from langfuse import Evaluation

from hm_agent.core.prompts import get_managed_prompt
from hm_agent.customer_support.rag.pipeline import ANSWER_CHAT_PROMPT, ANSWER_PROMPT_NAME
from hm_agent.evals.judges import deepeval_judge
from hm_agent.evals.text import strip_sources

logger = logging.getLogger(__name__)

HELPFULNESS_CRITERIA = (
    "The reply is helpful and addresses the customer's message, is polite and concise, sounds like a friendly H&M "
    "customer service assistant, and does not invent facts such as order details, prices or policies."
)
POLICY_CRITERIA = (
    "The reply gives the same return / replacement guidance as the expected output, without contradicting it or "
    "adding steps that are not in it."
)


def rag_rules() -> list[str]:
    """Read the rules of the RAG answer prompt, so alignment is checked against the version in use.

    Returns:
        list[str]: The ``- ...`` bullet lines under ``# RULES`` of the current ``production`` / configured-label prompt.
    """
    prompt = get_managed_prompt(ANSWER_PROMPT_NAME, ANSWER_CHAT_PROMPT, kind="chat")
    system = prompt.compile(context="", query="")[0]["content"]
    rules = re.search(r"# RULES\n(.*?)(?:\n#|\Z)", system, re.DOTALL)
    return [line[2:].strip() for line in (rules.group(1) if rules else "").splitlines() if line.startswith("- ")]


def _measure(name: str, metric: Any, case: Any) -> Evaluation:
    """Run one DeepEval metric, turning errors into a ``None`` evaluation.

    Args:
        name (str): Evaluation name to report.
        metric (Any): Configured DeepEval metric.
        case (Any): ``LLMTestCase`` to score.

    Returns:
        Evaluation: The metric score in [0, 1], or ``None`` with the failure in the comment.
    """
    try:
        metric.measure(case)
    except Exception as exc:  # noqa: BLE001 - one flaky judge call must not kill the whole evaluation
        logger.exception("deepeval metric=%s status=failed", name)
        return Evaluation(name=name, value=None, comment=f"error={type(exc).__name__}")
    return Evaluation(name=name, value=float(metric.score))


def deepeval_evaluator(*, input: Any, output: dict, expected_output: Any = None, metadata: Any = None, **_kwargs: Any) -> list[Evaluation]:
    """Langfuse item evaluator: the DeepEval metrics that apply to the item.

    Args:
        input (Any): Item input ``{"question", "customer_id"}``.
        output (dict): Task output from ``run_golden_task``.
        expected_output (Any): Item expected output (``reference``, ``expected_tools``).
        metadata (Any): Item metadata (``suite``, ``agent``).
        **_kwargs (Any): Ignored extras.

    Returns:
        list[Evaluation]: DeepEval evaluations; empty for blocked, failed or routing-only items.
    """
    expected, metadata = expected_output or {}, metadata or {}
    suite, answer = metadata.get("suite", ""), strip_sources(output["answer"])
    if suite == "supervisor" or output["blocked"] or output["error"] or not answer:
        return []

    from deepeval.metrics import (
        BiasMetric,
        GEval,
        HallucinationMetric,
        PromptAlignmentMetric,
        ToolCorrectnessMetric,
        ToxicityMetric,
    )
    from deepeval.test_case import LLMTestCase, SingleTurnParams, ToolCall

    judge = deepeval_judge()
    common = {"model": judge, "async_mode": False, "verbose_mode": False}
    reference, contexts = expected.get("reference"), output.get("contexts") or []
    case = LLMTestCase(
        input=input["question"],
        actual_output=answer,
        expected_output=reference,
        context=contexts or None,
        retrieval_context=contexts or None,
        tools_called=[ToolCall(name=c["name"], input_parameters=c["args"]) for c in output["tool_calls"]],
        expected_tools=[ToolCall(name=n) for n in expected.get("expected_tools", [])],
    )
    io = [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT]
    results = [
        _measure(
            "deepeval_helpfulness_tone",
            GEval(name="helpfulness_tone", criteria=HELPFULNESS_CRITERIA, evaluation_params=io, **common),
            case,
        )
    ]
    if suite == "rag" and reference:
        results.append(
            _measure(
                "deepeval_policy_adherence",
                GEval(name="policy_adherence", criteria=POLICY_CRITERIA, evaluation_params=[*io, SingleTurnParams.EXPECTED_OUTPUT], **common),
                case,
            )
        )
    if expected.get("expected_tools"):
        results.append(_measure("deepeval_tool_correctness", ToolCorrectnessMetric(include_reason=False, **common), case))
    if suite == "rag" and contexts:
        results.append(_measure("deepeval_no_hallucination", HallucinationMetric(include_reason=False, **common), case))
        results.append(
            _measure("deepeval_prompt_alignment", PromptAlignmentMetric(prompt_instructions=rag_rules(), include_reason=False, **common), case)
        )
    if suite == "guardrail" or metadata.get("agent") == "fashion":
        results.append(_measure("deepeval_non_toxic", ToxicityMetric(include_reason=False, **common), case))
        results.append(_measure("deepeval_unbiased", BiasMetric(include_reason=False, **common), case))
    return results
