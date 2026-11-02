"""One supervisor turn as a stream of events, shared by the CLI and the runtime entrypoint.

Role in the system: both front ends (``hm_agent.cli`` and ``hm_agent.runtime``) run a customer message
through the compiled supervisor graph the same way. This module builds the per-turn run config
(thread, customer, Langfuse tracing) and turns ``graph.stream`` output into small event dicts:

    {"type": "agent_selected", "agent": ...}   the supervisor chose an agent
    {"type": "started",  "agent": ..., "task_id": ...}   the agent accepted the task
    {"type": "progress", "agent": ..., "text": ...}      a status line while the agent works
    {"type": "token",    "agent": ..., "text": ...}      a piece of the reply, as it is produced
    {"type": "reply",    "agent": ..., "text": ...}      the full reply (always the last event)

A message split between several agents gets one ``agent_selected`` per agent and ``started`` /
``progress`` events labelled by agent. Their answers stream as ``token`` events with agent
``multi_agent``, one section at a time (each behind a short lead-in), starting as soon as the first
agent answers; the last event is one ``reply`` with the whole combined text.

Each turn also logs one ``turn status=done ... ttft_s=... total_s=...`` line (time to the first token
and total time) for log-based latency metrics, and writes online scores to Langfuse (see
``hm_agent.evals.online``).

When Bedrock Guardrails are enabled the customer message is screened before the graph runs: a blocked message
(toxic language, prompt injection, a card number) gets a fixed refusal and never reaches a model or the
checkpointer, and masked PII is replaced before it reaches any LLM. The final reply is screened too; a masked
or blocked reply replaces the text in the last ``reply`` event (tokens already streamed cannot be recalled, so
clients show the ``reply`` text as the authoritative answer).
"""

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langfuse import get_client
from langgraph.graph.state import CompiledStateGraph

from hm_agent.auth import USER_TOKEN_CONFIG_KEY
from hm_agent.core.logging_setup import trace_id_var
from hm_agent.core.settings import (
    GUARDRAIL_BLOCKED_REPLY,
    GUARDRAIL_CHECK_INPUT,
    GUARDRAIL_CHECK_OUTPUT,
    GUARDRAIL_FAIL_CLOSED_INPUT,
    guardrails_enabled,
)
from hm_agent.core.tracing import TRACE_ID_CONFIG_KEY, build_langfuse_config, new_trace_id
from hm_agent.customer_support.rag.pipeline import NO_ANSWER
from hm_agent.evals.online import TurnObservation, compute_turn_scores, publish_relevancy_async, publish_scores
from hm_agent.evals.usage import summarise_usage, usage_handler
from hm_agent.guardrails import GuardrailResult, screen
from hm_agent.supervisor.agent import OUT_OF_SCOPE_REPLY

logger = logging.getLogger(__name__)

# Replies written by code, not by a model: nothing to screen, so the output guardrail is skipped (saves cost).
_CANNED_REPLIES = {OUT_OF_SCOPE_REPLY, NO_ANSWER}


def build_config(customer_id: str, thread_id: str, user_token: str | None = None) -> RunnableConfig:
    """Build the per-turn run config (thread, customer and Langfuse tracing metadata).

    A fresh ``CallbackHandler`` is created per turn so each turn is its own trace, while the
    shared ``langfuse_session_id`` groups all turns of the thread into one Langfuse session.

    Args:
        customer_id (str): Customer the conversation belongs to, e.g. ``cust001``.
        thread_id (str): Conversation thread id used by the checkpointer and Langfuse.
        user_token (str | None): The end user's access token, forwarded to the agents so they can verify
            it. Kept out of the checkpoint and Langfuse metadata; never log it.

    Returns:
        RunnableConfig: Config to pass to ``graph.stream`` / ``graph.invoke``.
    """
    # The trace id is created here (not by the handler) so it can be sent to the agents and used for scoring.
    return build_langfuse_config(
        customer_id,
        thread_id,
        "hm-supervisor",
        trace_id=new_trace_id(),
        extra_configurable={USER_TOKEN_CONFIG_KEY: user_token} if user_token else None,
    )


@contextmanager
def _guardrail_span(trace_id: str, name: str):
    """Open a Langfuse ``guardrail`` observation inside the turn's trace (a no-op if Langfuse fails).

    Args:
        trace_id (str): Trace of the current turn.
        name (str): Observation name, ``guardrail:input`` or ``guardrail:output``.

    Yields:
        The Langfuse observation, or None when it could not be created.
    """
    try:
        manager = get_client().start_as_current_observation(
            trace_context={"trace_id": trace_id}, name=name, as_type="guardrail"
        )
        span = manager.__enter__()
    except Exception:  # noqa: BLE001 - tracing problems must not affect the turn
        logger.exception("guardrail_span status=failed name=%s", name)
        yield None
        return
    try:
        yield span
    finally:
        manager.__exit__(None, None, None)


def _screen(text: str, source: str, trace_id: str) -> GuardrailResult:
    """Screen text with the guardrail inside a traced ``guardrail`` observation.

    Args:
        text (str): Text to screen. Never logged or put in the span.
        source (str): ``INPUT`` or ``OUTPUT``.
        trace_id (str): Trace of the current turn.

    Returns:
        GuardrailResult: The screening outcome.
    """
    fail_closed = GUARDRAIL_FAIL_CLOSED_INPUT if source == "INPUT" else False  # a reply is never held back on error
    with _guardrail_span(trace_id, f"guardrail:{source.lower()}") as span:
        result = screen(text, source, fail_closed=fail_closed)
        if span is not None:
            # Category names and counts only: no message text, no PII values.
            span.update(
                output={
                    "action": result.action,
                    "toxicity": result.toxicity,
                    "prompt_attack": result.prompt_attack,
                    "pii_entities": result.pii_entities,
                    "error": result.error,
                }
            )
    return result


def stream_turn(
    graph: CompiledStateGraph, query: str, customer_id: str, thread_id: str, user_token: str | None = None
) -> Iterator[dict]:
    """Run one customer message through the graph and yield events as they happen.

    Args:
        graph (CompiledStateGraph): Compiled supervisor graph (with a checkpointer for multi-turn use).
        query (str): The customer's message for this turn.
        customer_id (str): Customer the conversation belongs to.
        thread_id (str): Conversation thread id; earlier turns are loaded from the checkpointer.
        user_token (str | None): End user's access token to forward to the agents (None when auth is off).

    Yields:
        dict: Event dicts as described in the module docstring. The last one has ``type == "reply"``
        (it is missing only when the graph produced no AI message).
    """
    start = time.perf_counter()
    config = build_config(customer_id, thread_id, user_token)
    trace_id = config["configurable"][TRACE_ID_CONFIG_KEY]
    trace_id_var.set(trace_id)  # log lines of this turn carry the trace id
    guard_in: GuardrailResult | None = None
    guard_out: GuardrailResult | None = None

    if guardrails_enabled() and GUARDRAIL_CHECK_INPUT:
        guard_in = _screen(query, "INPUT", trace_id)
        if guard_in.action == "BLOCKED":
            # Refuse here: the message never reaches a model, the agents or the checkpointer.
            total_s = time.perf_counter() - start
            logger.info("turn status=blocked thread_id=%s total_s=%.2f", thread_id, total_s)
            publish_scores(
                trace_id,
                compute_turn_scores(
                    TurnObservation(customer_id, None, total_s, [], "guardrail", GUARDRAIL_BLOCKED_REPLY, guard_in)
                ),
            )
            yield {"type": "reply", "agent": "guardrail", "text": GUARDRAIL_BLOCKED_REPLY}
            return
        query = guard_in.text  # PII already masked, if any

    first_token_s: float | None = None  # time to first token (ttft), set once on the first "token" event
    agent_names: list[str] = []  # agents the supervisor picked; more than one means a split turn
    reply: dict | None = None  # the final reply event, held back until the output screening and scoring are done
    # Two stream modes at once: "updates" gives per-node state changes (routing, final replies), "custom"
    # gives the events the nodes write themselves (started / progress / token).
    stream = graph.stream(
        {"messages": [HumanMessage(query)]},
        config,
        stream_mode=["updates", "custom"],
    )
    for mode, data in stream:
        if mode == "custom":
            if data.get("type") == "token" and first_token_s is None:
                first_token_s = time.perf_counter() - start
            yield data  # written by the remote-agent node: started / progress / token
            continue
        # "updates" payload: {node_name: state_update}.
        for node, update in data.items():
            if node == "supervisor":
                for task in update["tasks"]:
                    agent_names.append(task.agent)
                    yield {"type": "agent_selected", "agent": task.agent}
            elif update and update.get("messages"):
                # Branch nodes of a split turn write ``results`` (no reply); ``synthesize`` writes the combined one.
                message = update["messages"][-1]
                reply = {"type": "reply", "agent": message.name or node, "text": message.text}

    if reply is not None and guardrails_enabled() and GUARDRAIL_CHECK_OUTPUT and reply["text"] not in _CANNED_REPLIES:
        guard_out = _screen(reply["text"], "OUTPUT", trace_id)
        if guard_out.action == "BLOCKED":
            reply["text"] = GUARDRAIL_BLOCKED_REPLY
        elif guard_out.action == "MASKED":
            reply["text"] = guard_out.text

    total_s = time.perf_counter() - start
    # One line per turn, shaped for log-based metrics: alert on a high p95 of ttft_s, split-turn share, etc.
    logger.info(
        "turn status=done thread_id=%s split=%s agents=%d ttft_s=%s total_s=%.2f",
        thread_id,
        len(agent_names) > 1,
        len(agent_names),
        f"{first_token_s:.2f}" if first_token_s is not None else "none",
        total_s,
    )
    handler = usage_handler(config)
    publish_scores(
        trace_id,
        compute_turn_scores(
            TurnObservation(
                customer_id=customer_id,
                ttft_s=first_token_s,
                ttlt_s=total_s,
                agents=agent_names,
                reply_agent=reply["agent"] if reply else "",
                reply=reply["text"] if reply else "",
                guardrail_input=guard_in,
                guardrail_output=guard_out,
                usage=summarise_usage(handler.usage_metadata) if handler else {},
            )
        ),
    )
    if reply is not None and reply["agent"] != "guardrail":
        publish_relevancy_async(trace_id, query, reply["text"])
    # The reply is yielded last, after scoring, because a consumer may stop reading once it has the reply.
    if reply is not None:
        yield reply
