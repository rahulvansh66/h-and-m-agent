"""Online scores: deterministic per-turn quality, latency, cost and safety signals written to Langfuse.

Role in the system: after each supervisor turn (and each agent run) this module turns what was observed
(timings, routing, reply text, tool calls, guardrail outcomes, token usage) into Langfuse scores attached to
the turn's trace. Scores are cheap, need no LLM, and carry no PII (flags and numbers only), so they run on
every turn. Dashboards then show p50/p90/p95 latency, error and fallback rates, tool health, guardrail
activity and cost by release and prompt version. Scoring can never break a turn: every failure is logged
and swallowed.
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Literal

from langfuse import get_client

from hm_agent.a2a.push_receiver import FAILED_REPLY
from hm_agent.a2a.remote_node import PENDING_REPLY, UNAVAILABLE_REPLY
from hm_agent.core.settings import online_relevancy_enabled, online_scores_enabled
from hm_agent.core.state import CUSTOMER_SUPPORT, OUT_OF_SCOPE
from hm_agent.customer_support.rag.pipeline import NO_ANSWER
from hm_agent.evals.safety import has_citation, other_customer_ids
from hm_agent.evals.similarity import cosine_similarity
from hm_agent.evals.text import strip_sources
from hm_agent.evals.toolcalls import ToolCall, tool_health_scores
from hm_agent.guardrails import GuardrailResult
from hm_agent.supervisor.agent import OUT_OF_SCOPE_REPLY
from hm_agent.supervisor.synthesis import ALL_FAILED_REPLY

logger = logging.getLogger(__name__)

ScoreType = Literal["NUMERIC", "BOOLEAN", "CATEGORICAL"]
FAILURE_REPLIES = {UNAVAILABLE_REPLY, ALL_FAILED_REPLY, FAILED_REPLY}
# Replies written by code, not by a model: no relevancy or correctness to judge.
CANNED_REPLIES = FAILURE_REPLIES | {PENDING_REPLY, NO_ANSWER, OUT_OF_SCOPE_REPLY}


@dataclass(frozen=True)
class Score:
    """One Langfuse score to write.

    Attributes:
        name (str): Score name, e.g. ``ttft_s``.
        value (float | str): Number (or 0/1 for booleans), or a category label.
        data_type (ScoreType): Langfuse data type.
        comment (str | None): Short note; never contains customer text.
    """

    name: str
    value: float | str
    data_type: ScoreType = "NUMERIC"
    comment: str | None = None


@dataclass
class TurnObservation:
    """Everything observed about one supervisor turn.

    Attributes:
        customer_id (str): Customer the turn belongs to.
        ttft_s (float | None): Seconds to the first streamed token (None if nothing streamed).
        ttlt_s (float): Seconds until the last token / the final reply.
        agents (list[str]): Agents the supervisor picked.
        reply_agent (str): Name on the final reply (an agent, ``multi_agent`` or ``out_of_scope``).
        reply (str): Final reply text (screened by the output guardrail). Never logged.
        guardrail_input (GuardrailResult | None): Input screening outcome, when guardrails are on.
        guardrail_output (GuardrailResult | None): Output screening outcome, when guardrails are on.
        usage (dict[str, float]): Output of ``summarise_usage`` for the supervisor's own LLM calls.
    """

    customer_id: str
    ttft_s: float | None
    ttlt_s: float
    agents: list[str]
    reply_agent: str
    reply: str
    guardrail_input: GuardrailResult | None = None
    guardrail_output: GuardrailResult | None = None
    usage: dict[str, float] = field(default_factory=dict)


def _flag(name: str, value: bool, comment: str | None = None) -> Score:
    """Build a boolean score.

    Args:
        name (str): Score name.
        value (bool): Flag value.
        comment (str | None): Optional note.

    Returns:
        Score: A ``BOOLEAN`` score with value 1.0 or 0.0.
    """
    return Score(name, float(value), "BOOLEAN", comment)


def guardrail_scores(result: GuardrailResult | None, side: Literal["input", "output"]) -> list[Score]:
    """Build the scores describing one guardrail check.

    Args:
        result (GuardrailResult | None): The screening outcome; None (guardrails off) yields no scores.
        side (Literal["input", "output"]): Which check this was.

    Returns:
        list[Score]: Action (categorical), toxicity / prompt-attack / PII flags and latency.
    """
    if result is None:
        return []
    scores = [
        Score(f"guardrail_{side}_action", result.action, "CATEGORICAL"),
        _flag(f"guardrail_{side}_toxicity", result.toxicity),
        Score(f"guardrail_{side}_pii_entities", float(result.pii_count)),
        Score(f"guardrail_{side}_latency_s", round(result.latency_s, 3)),
        _flag(f"guardrail_{side}_error", result.error),
    ]
    if side == "input":
        scores.append(_flag("guardrail_prompt_injection", result.prompt_attack))
    return scores


def compute_turn_scores(obs: TurnObservation) -> list[Score]:
    """Turn a turn observation into the list of scores to write.

    Args:
        obs (TurnObservation): What was observed during the turn.

    Returns:
        list[Score]: Latency, routing, reliability, quality-proxy, safety, cost and guardrail scores.
    """
    reply, agents = obs.reply, obs.agents
    scores = [
        Score("ttlt_s", round(obs.ttlt_s, 3)),
        Score("routing_agent_count", float(len(agents))),
        _flag("split_turn", len(agents) > 1),
        _flag("routed_out_of_scope", OUT_OF_SCOPE in agents or obs.reply_agent == OUT_OF_SCOPE),
        _flag("agent_failed", reply in FAILURE_REPLIES),
        _flag("agent_pending", reply == PENDING_REPLY),
        _flag("guardrail_blocked", obs.guardrail_input is not None and obs.guardrail_input.action == "BLOCKED"),
        Score("reply_length_chars", float(len(reply))),
        _flag("customer_scope_leak", bool(other_customer_ids(reply, obs.customer_id))),
    ]
    if obs.ttft_s is not None:
        scores.append(Score("ttft_s", round(obs.ttft_s, 3)))
        generation_s = obs.ttlt_s - obs.ttft_s
        if generation_s > 0:  # decode speed in characters, since the stream carries text and not token counts
            scores.append(Score("output_chars_per_s", round(len(reply) / generation_s, 1)))
    if CUSTOMER_SUPPORT in agents or obs.reply_agent == CUSTOMER_SUPPORT:
        scores.append(_flag("rag_no_answer", NO_ANSWER in reply))
        scores.append(_flag("rag_has_citation", has_citation(reply)))
    if obs.usage:
        scores.append(Score("llm_total_tokens", obs.usage["total_tokens"], comment="agent=supervisor"))
        scores.append(Score("llm_cost_usd", round(obs.usage["cost_usd"], 6), comment="agent=supervisor notional"))
    scores += guardrail_scores(obs.guardrail_input, "input") + guardrail_scores(obs.guardrail_output, "output")
    return scores


def compute_agent_scores(
    agent: str, calls: list[ToolCall], duration_s: float, usage: dict[str, float]
) -> list[Score]:
    """Build the scores for one agent run (written onto the turn's trace by the agent service).

    Args:
        agent (str): Agent name, recorded in each comment so scores of different agents stay distinguishable.
        calls (list[ToolCall]): Tool calls the agent made (empty for agents without tools).
        duration_s (float): Wall time of the agent run.
        usage (dict[str, float]): Output of ``summarise_usage`` for the agent's LLM calls.

    Returns:
        list[Score]: Agent latency, tool-health, token and cost scores.
    """
    note = f"agent={agent}"
    scores = [Score("agent_latency_s", round(duration_s, 3), comment=note)]
    if agent in ("order_management", "fashion"):
        scores += [Score(name, value, comment=note) for name, value in tool_health_scores(calls, agent).items()]
    if usage:
        scores.append(Score("llm_total_tokens", usage["total_tokens"], comment=note))
        scores.append(Score("llm_cost_usd", round(usage["cost_usd"], 6), comment=f"{note} notional"))
    return scores


def publish_scores(trace_id: str | None, scores: list[Score]) -> None:
    """Write scores to Langfuse on a trace; never raises.

    Args:
        trace_id (str | None): Trace to attach the scores to; nothing is written when empty.
        scores (list[Score]): Scores to write.
    """
    if not trace_id or not scores or not online_scores_enabled():
        return
    try:
        client = get_client()
        for score in scores:
            client.create_score(
                name=score.name, value=score.value, trace_id=trace_id, data_type=score.data_type, comment=score.comment
            )
        logger.info("online_scores status=written trace_id=%s count=%d", trace_id, len(scores))
    except Exception:  # noqa: BLE001 - scoring must never break a customer turn
        logger.exception("online_scores status=failed trace_id=%s", trace_id)


def _publish_relevancy(trace_id: str, question: str, reply: str) -> None:
    """Score the reply's relevancy to the question with local embeddings and write it (worker thread body).

    Args:
        trace_id (str): Trace of the turn.
        question (str): The customer's message (screened); never logged.
        reply (str): The final reply; never logged.
    """
    try:
        similarity = cosine_similarity(question, strip_sources(reply))
    except Exception:  # noqa: BLE001 - the embedding model is optional telemetry, never a reason to fail
        logger.exception("online_scores status=relevancy_failed trace_id=%s", trace_id)
        return
    publish_scores(trace_id, [Score("answer_relevancy_sim", round(similarity, 4), comment="cosine(question, reply)")])


def publish_relevancy_async(trace_id: str | None, question: str, reply: str) -> None:
    """Score reply relevancy in the background, so loading the embedding model never delays the reply.

    Skipped when relevancy scoring is off, and for replies written by code (nothing to judge).

    Args:
        trace_id (str | None): Trace of the turn.
        question (str): The customer's message.
        reply (str): The final reply.
    """
    if not trace_id or not reply or reply in CANNED_REPLIES:
        return
    if not (online_scores_enabled() and online_relevancy_enabled()):
        return
    threading.Thread(target=_publish_relevancy, args=(trace_id, question, reply), daemon=True).start()


def record_feedback(trace_id: str, positive: bool, comment: str | None = None) -> None:
    """Record explicit customer feedback (thumbs up / down) on a turn's trace.

    Args:
        trace_id (str): Trace of the turn being rated.
        positive (bool): True for thumbs up, False for thumbs down.
        comment (str | None): Optional free-text reason. Do not include personal data.
    """
    publish_scores(trace_id, [Score("user_feedback", float(positive), "BOOLEAN", comment)])
