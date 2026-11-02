"""Offline evaluation runner: golden set -> Langfuse experiment with RAGAS, DeepEval and deterministic metrics.

Role in the system: the command that answers "did this prompt / model / code change make the system better or
worse?". It runs every golden item through the real components (``hm_agent.evals.tasks``), scores it with the
selected metric groups, stores the run in Langfuse (dataset run, per-item scores, run-level aggregates, linked
traces) and, unless ``--no-gate``, exits non-zero when a run-level number breaches ``evals.thresholds``.

Run (from the repo root, with the ``evals`` dependency group installed):
    uv run --group evals python -m hm_agent.evals --check                 # validate the golden file, no LLM calls
    uv run --group evals python -m hm_agent.evals --metrics det --limit 3 # cheapest: no judge calls
    uv run --group evals python -m hm_agent.evals --run-name prompts-v2   # full run (about 80 Groq judge calls)

The judge is the Groq model of ``evals.judge_model``: a full run is meant to be triggered by hand, not in CI.
Guardrail items call Bedrock only when guardrails are enabled (``GUARDRAILS_ENABLED=true``).
"""

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langfuse import Evaluation, get_client

from hm_agent.core.config import REPO_ROOT, load_config
from hm_agent.core.logging_setup import configure_logging, git_commit
from hm_agent.core.prompts import get_managed_prompt
from hm_agent.core.settings import EVAL_DATASET, EVAL_THRESHOLDS, guardrails_enabled
from hm_agent.evals.golden import load_golden
from hm_agent.evals.metrics_det import aggregate, check_thresholds, deterministic_evaluator
from hm_agent.evals.tasks import run_golden_task

logger = logging.getLogger("hm_agent.evals")

METRIC_GROUPS = ("det", "ragas", "deepeval")


def build_evaluators(groups: set[str]) -> list[Any]:
    """Collect the item evaluators for the selected metric groups.

    Args:
        groups (set[str]): Any of ``det``, ``ragas``, ``deepeval``.

    Returns:
        list[Any]: Evaluator callables. RAGAS / DeepEval are imported only when selected, so a ``det`` run
        works without the ``evals`` dependency group.
    """
    evaluators: list[Any] = []
    if "det" in groups:
        evaluators.append(deterministic_evaluator)
    if "ragas" in groups:
        from hm_agent.evals.metrics_ragas import ragas_evaluator

        evaluators.append(ragas_evaluator)
    if "deepeval" in groups:
        from hm_agent.evals.metrics_deepeval import deepeval_evaluator

        evaluators.append(deepeval_evaluator)
    return evaluators


def run_aggregates(*, item_results: list[Any], **_kwargs: Any) -> list[Evaluation]:
    """Langfuse run evaluator: mean of every metric, latency percentiles, throughput and total cost.

    Args:
        item_results (list[Any]): Per-item results with their evaluations.
        **_kwargs (Any): Ignored extras.

    Returns:
        list[Evaluation]: One run-level evaluation per number from ``aggregate``, named ``run_<metric>``.
    """
    return [Evaluation(name=f"run_{name}", value=value) for name, value in aggregate(item_results).items()]


def write_report(path: Path, run_name: str, result: Any, summary: dict[str, float], breaches: list[str]) -> None:
    """Save the run as JSON: one entry per item (question, answer, every score) plus the run-level numbers.

    Args:
        path (Path): File to write.
        run_name (str): Langfuse run name.
        result (Any): The experiment result.
        summary (dict[str, float]): Output of ``aggregate``.
        breaches (list[str]): Regression-gate messages.
    """
    items = []
    for item_result in result.item_results:
        item = item_result.item
        metadata = (item.get("metadata") if isinstance(item, dict) else getattr(item, "metadata", None)) or {}
        payload = (item.get("input") if isinstance(item, dict) else getattr(item, "input", None)) or {}
        output = item_result.output or {}
        items.append(
            {
                "id": metadata.get("id"),
                "suite": metadata.get("suite"),
                "question": payload.get("question"),
                "answer": output.get("answer"),
                "agents": output.get("agents"),
                "error": output.get("error"),
                "scores": {e.name: e.value for e in item_result.evaluations},
                "trace_id": item_result.trace_id,
            }
        )
    path.write_text(
        json.dumps({"run": run_name, "metadata": run_metadata(), "summary": summary, "breaches": breaches, "items": items}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_metadata() -> dict:
    """Describe what is being evaluated, so runs can be compared by prompt version, model and release.

    Returns:
        dict: Prompt versions, model names, git commit, guardrail state and judge model.
    """
    from hm_agent.prompt_registry import PROMPTS

    config = load_config()
    return {
        "commit": git_commit(),
        "prompt_versions": {p.name: get_managed_prompt(p.name, p.default, kind=p.kind).version for p in PROMPTS},
        "models": config["llm"],
        "judge_model": config["llm"][config["evals"]["judge_model"]],
        "guardrails_enabled": guardrails_enabled(),
    }


def main() -> int:
    """Parse arguments, run the experiment and apply the regression gate.

    Returns:
        int: Process exit code: 0 on success, 1 when a threshold is breached.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics", default=",".join(METRIC_GROUPS), help="comma list of: det, ragas, deepeval")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N items")
    parser.add_argument("--run-name", default=None, help="Langfuse run name (default: eval-<commit>-<time>)")
    parser.add_argument("--local", action="store_true", help="use data/evals/golden.jsonl instead of the Langfuse dataset")
    parser.add_argument("--check", action="store_true", help="validate the golden file and exit (no LLM calls)")
    parser.add_argument("--report", type=Path, default=None, help="write per-item answers and scores plus run totals to this JSON file")
    parser.add_argument("--no-gate", action="store_true", help="do not fail on threshold breaches")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    sys.stdout.reconfigure(encoding="utf-8")  # the result summary contains emoji; Windows consoles default to cp1252
    configure_logging("evals")
    logging.getLogger("hm_agent.evals").setLevel(logging.INFO)

    items = load_golden()
    if args.check:
        counts = Counter(item.metadata["suite"] for item in items)
        logger.info("golden items=%d suites=%s", len(items), dict(counts))
        return 0

    groups = {g.strip() for g in args.metrics.split(",") if g.strip()}
    unknown = groups - set(METRIC_GROUPS)
    if unknown:
        parser.error(f"unknown metric group(s): {sorted(unknown)}")
    if not guardrails_enabled():
        logger.warning("guardrails are off: guardrail_action_correct / guardrail_false_positive will not be scored")

    langfuse = get_client()
    run_name = args.run_name or f"eval-{git_commit()}-{time.strftime('%Y%m%d-%H%M%S')}"
    kwargs = dict(
        name="hm-golden-set",
        run_name=run_name,
        description="Golden-set evaluation: routing, tools, RAG, guardrails, performance",
        task=run_golden_task,
        evaluators=build_evaluators(groups),
        run_evaluators=[run_aggregates],
        max_concurrency=1,  # one item at a time: gentle on the Groq free tier, and latency is not skewed by parallel runs
        metadata=run_metadata(),
    )
    if args.local:
        data = [item.as_dataset_item() for item in items[: args.limit]]
        result = langfuse.run_experiment(data=data, **kwargs)
    else:
        dataset = langfuse.get_dataset(EVAL_DATASET)
        # The API returns items in no fixed order; sort by id so --limit always picks the same items.
        dataset.items = sorted(dataset.items, key=lambda item: item.id)[: args.limit]
        result = dataset.run_experiment(**kwargs)

    print(result.format(include_item_results=True))
    summary = aggregate(result.item_results)
    langfuse.flush()
    breaches = check_thresholds(summary, EVAL_THRESHOLDS)
    if args.report:
        write_report(args.report, run_name, result, summary, breaches)
        logger.info("report=%s", args.report)
    for breach in breaches:
        logger.error("threshold_breach %s", breach)
    logger.info("run=%s items=%d breaches=%d", run_name, len(result.item_results), len(breaches))
    return 1 if breaches and not args.no_gate else 0


if __name__ == "__main__":
    sys.exit(main())
