"""Offline task: run one golden item through the real components in-process and record what happened.

Role in the system: the ``task`` of the Langfuse experiment. It does not need the A2A services running; it
calls the same code the services run (supervisor routing, the RAG graph, the MCP-backed agents) and the same
guardrail hooks as ``hm_agent.turns``. What each suite runs:

    supervisor   routing decision only (one small-model call)
    rag          the customer-support RAG graph (answer + retrieved contexts for RAGAS)
    agent_tools  the order_management / fashion agent with its MCP tools (tool calls recorded)
    guardrail    input guardrail -> routing -> first specialist -> output guardrail

The agents need the local services the app needs: Neon (``neondb``/test DB) for orders, and for fashion the
fashion MCP server (``python -m hm_agent.fashion.mcp_server``) plus Qdrant. The output is a plain dict so
Langfuse can store it and the evaluators can read it.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from functools import cache
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda

from hm_agent.core.settings import GUARDRAIL_BLOCKED_REPLY, guardrails_enabled
from hm_agent.core.state import CUSTOMER_SUPPORT, FASHION, ORDER_MANAGEMENT, OUT_OF_SCOPE
from hm_agent.core.tracing import build_langfuse_config
from hm_agent.customer_support.rag.pipeline import NO_ANSWER, build_rag_graph
from hm_agent.evals.toolcalls import extract_tool_calls
from hm_agent.evals.usage import summarise_usage, usage_handler
from hm_agent.guardrails import screen
from hm_agent.supervisor.agent import OUT_OF_SCOPE_REPLY, supervisor_node

logger = logging.getLogger(__name__)

_CANNED_REPLIES = {OUT_OF_SCOPE_REPLY, NO_ANSWER, GUARDRAIL_BLOCKED_REPLY}


def _field(item: Any, name: str) -> Any:
    """Read ``input`` / ``expected_output`` / ``metadata`` from a Langfuse dataset item or a local dict.

    Args:
        item (Any): ``DatasetItem`` object or ``{"input", "expected_output", "metadata"}`` dict.
        name (str): Field name.

    Returns:
        Any: The field value (``None`` if absent).
    """
    return item.get(name) if isinstance(item, dict) else getattr(item, name, None)


@cache
def _agent_graph(name: str):
    """Build an MCP-backed agent once per run.

    Args:
        name (str): ``order_management`` or ``fashion``.

    Returns:
        CompiledStateGraph: The agent graph.
    """
    if name == ORDER_MANAGEMENT:
        from hm_agent.order_management.agent import build_order_agent

        return build_order_agent()
    from hm_agent.fashion.agent import build_fashion_agent

    return build_fashion_agent()


def _route(query: str, config: RunnableConfig) -> list[str]:
    """Run the supervisor's routing step.

    Args:
        query (str): The (screened) customer message.
        config (RunnableConfig): Run config with tracing and usage callbacks.

    Returns:
        list[str]: Agents the supervisor chose, e.g. ``["order_management"]``. ``out_of_scope`` is reported as
        an empty list, which is how the golden set writes "no agent".
    """
    update = RunnableLambda(supervisor_node).invoke({"messages": [HumanMessage(query)]}, config)
    agents = [task.agent for task in update["tasks"]]
    return [] if agents == [OUT_OF_SCOPE] else agents


def _run_rag(query: str, config: RunnableConfig) -> tuple[str, list[dict]]:
    """Run the support RAG graph.

    Args:
        query (str): The (screened) customer question.
        config (RunnableConfig): Run config with tracing and usage callbacks.

    Returns:
        tuple[str, list[dict]]: The cited answer and the documents it was built from (those that passed the
        relevance gate, in rank order), each with ``text`` and ``chunk_id``.
    """
    result = build_rag_graph().invoke({"query": query}, config)
    return result["answer"], result.get("documents", [])


def _run_agent(agent: str, query: str, config: RunnableConfig) -> tuple[str, list[dict]]:
    """Run an MCP-backed agent and collect its tool calls.

    The MCP tool wrappers drive their own event loop (``asyncio.run``), which is not allowed inside the
    experiment runner's running loop. The agent is therefore built and run in a worker thread, which has none.

    Args:
        agent (str): ``order_management`` or ``fashion``.
        query (str): The (screened) customer message.
        config (RunnableConfig): Run config; ``configurable.customer_id`` scopes the tools to the customer.

    Returns:
        tuple[str, list[dict]]: The agent's reply and its tool calls as ``{"name", "args", "ok", "invalid_args"}``.
    """

    def run() -> dict:
        return _agent_graph(agent).invoke({"messages": [HumanMessage(query)]}, config)

    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(copy_context().run, run).result()
    calls = [vars(call) for call in extract_tool_calls(result["messages"])]
    return result["messages"][-1].text, calls


def run_golden_task(*, item: Any, **_kwargs: Any) -> dict:
    """Run one golden item and return everything the evaluators need.

    Args:
        item (Any): Langfuse ``DatasetItem`` or local dict (see ``_field``).
        **_kwargs (Any): Ignored extras passed by the experiment runner.

    Returns:
        dict: ``answer``, ``agents``, ``tool_calls``, ``contexts`` and ``context_ids`` (retrieved chunk texts and ids, rank order), ``downstream_query`` (the text the models
        saw, i.e. after PII masking), ``guardrail_input`` / ``guardrail_output`` (None when guardrails are
        off), ``blocked``, timings (``ttlt_s``, ``routing_s``, ``agent_s``), ``usage`` (tokens and notional
        cost) and ``error`` (empty when the run succeeded).
    """
    payload, metadata = _field(item, "input"), _field(item, "metadata") or {}
    query, customer_id, suite = payload["question"], payload["customer_id"], metadata["suite"]
    config = build_langfuse_config(customer_id, "eval", "eval-item", extra_tags=["eval", f"suite:{suite}"])
    out: dict = {
        "answer": "", "agents": [], "tool_calls": [], "contexts": [], "context_ids": [], "downstream_query": query,
        "guardrail_input": None, "guardrail_output": None, "blocked": False,
        "routing_s": 0.0, "agent_s": 0.0, "error": "",
    }
    start = time.perf_counter()
    try:
        if guardrails_enabled():
            guard_in = screen(query, "INPUT", fail_closed=True)
            out["guardrail_input"] = {"action": guard_in.action, "toxicity": guard_in.toxicity,
                                      "prompt_attack": guard_in.prompt_attack, "pii": guard_in.pii_entities}
            if guard_in.action == "BLOCKED":
                out.update(answer=GUARDRAIL_BLOCKED_REPLY, blocked=True)
            else:
                query = out["downstream_query"] = guard_in.text
        if not out["blocked"]:
            _execute(suite, metadata, query, config, out)
        if guardrails_enabled() and out["answer"] and out["answer"] not in _CANNED_REPLIES:
            guard_out = screen(out["answer"], "OUTPUT", fail_closed=False)
            out["guardrail_output"] = {"action": guard_out.action, "toxicity": guard_out.toxicity}
            if guard_out.action == "BLOCKED":
                out["answer"] = GUARDRAIL_BLOCKED_REPLY
            elif guard_out.action == "MASKED":
                out["answer"] = guard_out.text
    except Exception as exc:  # noqa: BLE001 - a failing item is a data point (task_success=0), not a crashed run
        logger.exception("eval_task status=failed")
        out["error"] = type(exc).__name__
    out["ttlt_s"] = time.perf_counter() - start
    handler = usage_handler(config)
    out["usage"] = summarise_usage(handler.usage_metadata) if handler else {}
    return out


def _execute(suite: str, metadata: dict, query: str, config: RunnableConfig, out: dict) -> None:
    """Run the components a suite needs and fill ``out`` in place.

    Args:
        suite (str): ``supervisor``, ``rag``, ``agent_tools`` or ``guardrail``.
        metadata (dict): Item metadata; ``agent`` names the agent for ``agent_tools`` items.
        query (str): The (screened) customer message.
        config (RunnableConfig): Run config with tracing and usage callbacks.
        out (dict): Task output being built (``agents``, ``answer``, ``contexts``, ``tool_calls``, timings).
    """
    if suite == "rag":
        start = time.perf_counter()
        out["answer"], docs = _run_rag(query, config)
        out["contexts"], out["context_ids"] = [d["text"] for d in docs], [d["chunk_id"] for d in docs]
        out["agent_s"] = time.perf_counter() - start
        out["agents"] = [CUSTOMER_SUPPORT]
        return
    if suite == "agent_tools":
        start = time.perf_counter()
        out["answer"], out["tool_calls"] = _run_agent(metadata["agent"], query, config)
        out["agent_s"] = time.perf_counter() - start
        out["agents"] = [metadata["agent"]]
        return

    start = time.perf_counter()
    out["agents"] = _route(query, config)
    out["routing_s"] = time.perf_counter() - start
    if suite == "supervisor":
        return
    # guardrail suite: continue with the first specialist the supervisor chose, as a real turn would.
    start = time.perf_counter()
    first = out["agents"][0] if out["agents"] else OUT_OF_SCOPE
    if first in (ORDER_MANAGEMENT, FASHION):
        out["answer"], out["tool_calls"] = _run_agent(first, query, config)
    elif first == CUSTOMER_SUPPORT:
        out["answer"], docs = _run_rag(query, config)
        out["contexts"], out["context_ids"] = [d["text"] for d in docs], [d["chunk_id"] for d in docs]
    else:
        out["answer"] = OUT_OF_SCOPE_REPLY
    out["agent_s"] = time.perf_counter() - start
