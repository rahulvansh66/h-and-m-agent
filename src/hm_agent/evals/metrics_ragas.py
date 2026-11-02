"""RAGAS metrics for the golden set: retrieval quality and answer faithfulness / relevance / correctness.

Role in the system: the RAG-focused judge metrics of the offline evaluation (``--metrics ragas``).

    faithfulness        are the answer's claims supported by the retrieved contexts (hallucination guard)
    response_relevancy  does the answer address the question (embedding similarity of generated questions)
    context_precision   are the relevant retrieved chunks ranked first (needs a reference answer)
    context_recall      does the retrieved context cover the reference answer
    factual_correctness does the answer agree with the reference answer

Each metric costs several judge calls, so they only run on items that have what they need (contexts and / or a
reference). A metric that errors (for example a rate-limit that outlasts the retries) yields a ``None`` score
with the error name as a comment instead of failing the run.
"""

import logging
import math
from typing import Any

from langfuse import Evaluation

from hm_agent.evals.judges import allow_ragas_import, ragas_embeddings, ragas_llm, run_sync
from hm_agent.evals.online import CANNED_REPLIES
from hm_agent.evals.text import strip_sources

logger = logging.getLogger(__name__)


def _safe_score(name: str, metric: Any, sample: Any) -> Evaluation:
    """Score one RAGAS metric, turning errors and NaN into a ``None`` evaluation.

    Args:
        name (str): Evaluation name to report.
        metric (Any): Configured RAGAS metric.
        sample (Any): ``SingleTurnSample`` to score.

    Returns:
        Evaluation: The score in [0, 1], or ``None`` with the failure in the comment.
    """
    try:
        score = run_sync(metric.single_turn_ascore(sample))
    except Exception as exc:  # noqa: BLE001 - one flaky judge call must not kill the whole evaluation
        logger.exception("ragas metric=%s status=failed", name)
        return Evaluation(name=name, value=None, comment=f"error={type(exc).__name__}")
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return Evaluation(name=name, value=None, comment="no score")
    return Evaluation(name=name, value=float(score))


def ragas_evaluator(*, input: Any, output: dict, expected_output: Any = None, metadata: Any = None, **_kwargs: Any) -> list[Evaluation]:
    """Langfuse item evaluator: the RAGAS metrics that apply to the item.

    Args:
        input (Any): Item input ``{"question", "customer_id"}``.
        output (dict): Task output from ``run_golden_task``.
        expected_output (Any): Item expected output (``reference`` is the ground-truth answer).
        metadata (Any): Item metadata (``suite``).
        **_kwargs (Any): Ignored extras.

    Returns:
        list[Evaluation]: RAGAS evaluations; empty for items without an answer or without anything to grade.
    """
    expected, metadata = expected_output or {}, metadata or {}
    answer, reference = strip_sources(output["answer"]), expected.get("reference")
    if output["blocked"] or not answer or output["error"]:
        return []
    contexts = output.get("contexts") or []
    is_rag = metadata.get("suite") == "rag" and contexts
    if metadata.get("suite") == "supervisor" or output["answer"] in CANNED_REPLIES:
        return []  # routing-only items and replies written by code have no model answer to grade

    allow_ragas_import()
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import (
        FactualCorrectness,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    llm = ragas_llm()
    sample = SingleTurnSample(
        user_input=input["question"], response=answer, retrieved_contexts=contexts or None, reference=reference
    )
    results: list[Evaluation] = []
    if is_rag:
        results.append(_safe_score("faithfulness", Faithfulness(llm=llm), sample))
    # Relevancy needs only the question and the answer, so it runs for every answered item, agents included.
    # strictness=1: Groq accepts one completion per request, and the default of 3 asks for n=3.
    results.append(_safe_score("response_relevancy", ResponseRelevancy(llm=llm, embeddings=ragas_embeddings(), strictness=1), sample))
    if is_rag:
        if reference:
            results.append(_safe_score("context_precision", LLMContextPrecisionWithReference(llm=llm), sample))
            results.append(_safe_score("context_recall", LLMContextRecall(llm=llm), sample))
    if reference:
        results.append(_safe_score("factual_correctness", FactualCorrectness(llm=llm), sample))
    return results
