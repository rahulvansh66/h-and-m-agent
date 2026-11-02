"""Tests for the golden dataset, deterministic metrics, tool extraction, online scoring and run aggregation.

Role in the system: checks ``hm_agent.evals`` (everything except the LLM-judge metrics, which are not called):
the golden file's shape, routing / tool / safety / guardrail metrics, tool-call extraction from agent messages,
the per-turn online scores and the regression gate. Pure functions only; no network and no model.
"""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langfuse import Evaluation

from hm_agent.evals import metrics_det as det
from hm_agent.evals import online
from hm_agent.evals.golden import load_golden
from hm_agent.evals.toolcalls import extract_tool_calls, tool_health_scores
from hm_agent.evals.usage import summarise_usage
from hm_agent.guardrails import GuardrailResult


def _values(evals: list[Evaluation]) -> dict[str, float | None]:
    """Map evaluation names to values.

    Args:
        evals (list[Evaluation]): Evaluations to index.

    Returns:
        dict[str, float | None]: Name -> value.
    """
    return {e.name: e.value for e in evals}


def test_golden_set_shape() -> None:
    """Ten unique items: seven functional and three guardrail cases (toxic, PII, injection)."""
    items = load_golden()
    assert len(items) == 10 == len({i.id for i in items})
    guardrail = [i for i in items if i.metadata["suite"] == "guardrail"]
    assert {t for i in guardrail for t in i.metadata["tags"]} >= {"toxicity", "pii", "prompt_injection"}
    assert all(i.expected["expected_guardrail_action"] for i in guardrail)
    assert all(i.as_dataset_item()["metadata"]["id"] == i.id for i in items)


def test_rag_items_have_gold_chunks() -> None:
    """Every RAG item names the exact chunks a correct answer needs, so retrieval can be scored without an LLM."""
    rag = [i for i in load_golden() if i.metadata["suite"] == "rag"]
    assert rag and all(i.expected["expected_chunks"] for i in rag)
    assert all(chunk.count("|") == 3 for i in rag for chunk in i.expected["expected_chunks"])


def test_retrieval_metrics() -> None:
    """Precision, recall, hit rate and MRR against gold chunks, including a miss and an empty retrieval."""
    gold = ["a.pdf|p1|prose|0"]
    top = _values(det.retrieval_metrics(["a.pdf|p1|prose|0", "a.pdf|p3|table|2"], gold))
    assert top == {"retrieval_precision": 0.5, "retrieval_recall": 1.0, "retrieval_hit_rate": 1.0, "retrieval_mrr": 1.0}
    second = _values(det.retrieval_metrics(["a.pdf|p3|table|2", "a.pdf|p1|prose|0"], gold))
    assert second["retrieval_mrr"] == 0.5 and second["retrieval_recall"] == 1.0
    miss = _values(det.retrieval_metrics(["a.pdf|p3|table|2"], gold))
    assert (miss["retrieval_recall"], miss["retrieval_hit_rate"], miss["retrieval_mrr"]) == (0.0, 0.0, 0.0)
    assert _values(det.retrieval_metrics([], gold))["retrieval_precision"] == 0.0


def test_answer_quality_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Relevancy to the question and similarity to the golden answer, with the Sources footer ignored."""
    seen = []
    monkeypatch.setattr(det, "cosine_similarity", lambda a, b: seen.append((a, b)) or 0.8)
    scores = _values(det.answer_quality_metrics("q?", "answer [1]\n\nSources:\n[1] Doc", "golden"))
    assert scores == {"answer_relevancy_sim": 0.8, "answer_similarity": 0.8}
    assert seen == [("q?", "answer [1]"), ("answer [1]", "golden")]
    assert list(_values(det.answer_quality_metrics("q?", "answer", None))) == ["answer_relevancy_sim"]


def test_cosine_similarity_orders_related_above_unrelated() -> None:
    """The local embedder rates a paraphrase closer than an unrelated sentence, and empty text is 0."""
    from hm_agent.evals.similarity import cosine_similarity

    close = cosine_similarity("How do I return a damaged jacket?", "To return a damaged jacket, open Your Orders and choose Return.")
    far = cosine_similarity("How do I return a damaged jacket?", "The weather in Stockholm is cold in January.")
    assert close > far and cosine_similarity("", "x") == 0.0


def test_online_relevancy_is_async_and_skips_canned_replies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Relevancy is scored in a background thread, only for model-written replies, and only when switched on."""
    published = []
    monkeypatch.setattr(online, "cosine_similarity", lambda a, b: 0.5)
    monkeypatch.setattr(online, "publish_scores", lambda trace_id, scores: published.append(scores))
    monkeypatch.setenv("ONLINE_SCORES_ENABLED", "1")
    online.publish_relevancy_async("a" * 32, "q", "a real reply")  # relevancy switched off by the fixture
    monkeypatch.setenv("ONLINE_RELEVANCY_ENABLED", "1")
    online.publish_relevancy_async("a" * 32, "q", online.NO_ANSWER)  # canned: skipped
    online.publish_relevancy_async("a" * 32, "q", "a real reply")
    for thread in __import__("threading").enumerate():
        if thread is not __import__("threading").current_thread() and thread.daemon:
            thread.join(timeout=5)
    assert [[(s.name, s.value) for s in scores] for scores in published] == [[("answer_relevancy_sim", 0.5)]]


def test_safety_checks_see_through_lookalike_characters() -> None:
    """Non-breaking hyphens and no-break spaces (which models emit) do not hide a leaked id or card number."""
    from hm_agent.evals.safety import has_pii, other_customer_ids
    from hm_agent.evals.text import normalise

    assert normalise("ORD‑1007 ok") == "ORD-1007 ok"
    assert other_customer_ids("see cust‑002? no: cust002", "cust001") == {"cust002"}
    assert has_pii("card 4111 1111 1111 1111")
    leaked = _output("Your card 4111 1111 1111 1111", original="my card 4111 1111 1111 1111")
    assert _values(det.safety_metrics(leaked, "cust001", [], False))["pii_leak"] == 1.0


def test_routing_metrics() -> None:
    """Exact match, partial overlap and the empty (out-of-scope) case."""
    assert _values(det.routing_metrics(["fashion"], ["fashion"]))["routing_exact_match"] == 1.0
    assert _values(det.routing_metrics([], []))["routing_f1"] == 1.0
    partial = _values(det.routing_metrics(["order_management"], ["order_management", "customer_support"]))
    assert partial["routing_exact_match"] == 0.0 and partial["routing_f1"] == pytest.approx(2 / 3)


def test_tool_metrics_name_and_args() -> None:
    """A correct tool with correct args scores 1; a wrong tool lowers precision and fails the args check."""
    good = [{"name": "get_order_details", "args": {"order_id": "ORD-1007"}, "ok": True, "invalid_args": False}]
    scores = _values(det.tool_metrics(good, ["get_order_details"], {"get_order_details": {"order_id": "ORD-1007"}}))
    assert scores["tool_name_f1"] == 1.0 and scores["tool_args_correct"] == 1.0 and scores["tool_call_success"] == 1.0
    bad = [{"name": "list_orders", "args": {}, "ok": False, "invalid_args": False}]
    scores = _values(det.tool_metrics(bad, ["get_order_details"], {"get_order_details": {"order_id": "ORD-1007"}}))
    assert scores["tool_name_f1"] == 0.0 and scores["tool_args_correct"] == 0.0 and scores["tool_call_success"] == 0.0


def _output(answer: str, downstream: str = "q", **extra) -> dict:
    """Build a minimal task output.

    Args:
        answer (str): Reply text.
        downstream (str): Text the models saw.
        **extra: Overrides for other task-output fields.

    Returns:
        dict: Task output as ``run_golden_task`` returns it.
    """
    return {
        "answer": answer, "downstream_query": downstream, "original_query": extra.pop("original", "q"),
        "agents": [], "tool_calls": [], "contexts": [], "guardrail_input": None, "blocked": False,
        "error": "", "ttlt_s": 1.0, "routing_s": 0.5, "agent_s": 0.5, "usage": {}, **extra,
    }


def test_safety_metrics_detect_leaks() -> None:
    """PII from the question surviving into the reply, a prompt marker and a foreign customer id are all flagged."""
    leaked = _output("Your card 4111 1111 1111 1111 and cust002 # RULES", original="my card 4111 1111 1111 1111")
    scores = _values(det.safety_metrics(leaked, "cust001", ["cust002"], is_injection=True))
    assert scores["pii_leak"] == scores["prompt_leak"] == scores["cross_customer_leak"] == 1.0
    assert scores["injection_resisted"] == 0.0 and scores["must_not_contain_violations"] == 1.0
    clean = _values(det.safety_metrics(_output("I can't help with that."), "cust001", [], is_injection=True))
    assert clean["injection_resisted"] == 1.0 and clean["pii_leak"] == 0.0


def test_guardrail_metrics() -> None:
    """Guardrail items are checked against the expected action; benign items must not trigger it."""
    blocked = _output("sorry", guardrail_input={"action": "BLOCKED"})
    assert _values(det.guardrail_metrics(blocked, ["BLOCKED"], "guardrail"))["guardrail_action_correct"] == 1.0
    assert _values(det.guardrail_metrics(blocked, None, "supervisor"))["guardrail_false_positive"] == 1.0
    assert det.guardrail_metrics(_output("x"), ["BLOCKED"], "guardrail") == []  # guardrails off -> not scored


def test_aggregate_and_thresholds() -> None:
    """Run-level means, latency percentiles, throughput and the regression gate."""
    results = [SimpleNamespace(evaluations=[Evaluation(name="ttlt_s", value=v), Evaluation(name="cost_usd", value=0.01)]) for v in (1.0, 2.0, 3.0, 10.0)]
    summary = det.aggregate(results)
    assert summary["ttlt_s"] == 4.0 and summary["ttlt_s_p50"] == 2.5 and summary["ttlt_s_max"] == 10.0
    assert summary["cost_usd_total"] == pytest.approx(0.04) and summary["items_per_min"] == pytest.approx(15.0)
    breaches = det.check_thresholds(summary, {"ttlt_s_p90": {"max": 5.0}, "missing_metric": {"min": 1}})
    assert len(breaches) == 1 and breaches[0].startswith("ttlt_s_p90")


def test_extract_tool_calls_and_health() -> None:
    """Calls are paired with their results; an error result and an out-of-allowlist tool are reported."""
    messages = [
        AIMessage("", tool_calls=[{"name": "list_orders", "args": {}, "id": "1"}, {"name": "drop_tables", "args": {}, "id": "2"}]),
        ToolMessage("ok", tool_call_id="1", name="list_orders"),
        ToolMessage("1 validation error: Field required", tool_call_id="2", name="drop_tables", status="error"),
    ]
    calls = extract_tool_calls(messages)
    assert [c.ok for c in calls] == [True, False] and calls[1].invalid_args
    health = tool_health_scores(calls, "order_management")
    assert health["tool_in_allowlist"] == 0.0 and health["tool_args_valid"] == 0.0 and health["tool_success_rate"] == 0.5


def test_summarise_usage_prices_known_models() -> None:
    """Tokens are totalled; cost uses the configured per-million prices and ignores unknown models."""
    usage = {"openai/gpt-oss-20b": {"input_tokens": 1_000_000, "output_tokens": 1_000_000, "total_tokens": 2_000_000}, "other": {"input_tokens": 5}}
    totals = summarise_usage(usage)
    assert totals["total_tokens"] == 2_000_005 and totals["cost_usd"] == pytest.approx(0.60)


def test_turn_scores_cover_latency_routing_safety_and_guardrails() -> None:
    """A split support turn produces the expected scores, flags a foreign customer id, and carries guardrail data."""
    obs = online.TurnObservation(
        customer_id="cust001", ttft_s=1.0, ttlt_s=3.0, agents=["order_management", "customer_support"],
        reply_agent="multi_agent", reply="See cust002 [1]\n\nSources:\n[1] Policy",
        guardrail_input=GuardrailResult("MASKED", "x", pii_entities={"EMAIL": 2}, latency_s=0.2),
        usage={"total_tokens": 100.0, "cost_usd": 0.001},
    )
    scores = {s.name: s.value for s in online.compute_turn_scores(obs)}
    assert scores["ttft_s"] == 1.0 and scores["ttlt_s"] == 3.0 and scores["split_turn"] == 1.0
    assert scores["customer_scope_leak"] == 1.0 and scores["rag_has_citation"] == 1.0
    assert scores["guardrail_input_action"] == "MASKED" and scores["guardrail_input_pii_entities"] == 2.0
    assert scores["llm_total_tokens"] == 100.0 and "output_chars_per_s" in scores


def test_failure_and_blocked_flags() -> None:
    """The apology reply counts as a failed agent; a blocked input is flagged."""
    failed = online.TurnObservation("cust001", None, 1.0, ["fashion"], "fashion", online.UNAVAILABLE_REPLY)
    assert {s.name: s.value for s in online.compute_turn_scores(failed)}["agent_failed"] == 1.0
    blocked = online.TurnObservation("cust001", None, 0.1, [], "guardrail", "no", GuardrailResult("BLOCKED", "x"))
    assert {s.name: s.value for s in online.compute_turn_scores(blocked)}["guardrail_blocked"] == 1.0


def test_publish_scores_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Langfuse failure while scoring is swallowed, so it can never break a customer turn."""

    class Broken:
        def create_score(self, **_kwargs):
            raise RuntimeError("langfuse down")

    monkeypatch.setenv("ONLINE_SCORES_ENABLED", "1")
    monkeypatch.setattr(online, "get_client", lambda: Broken())
    online.publish_scores("a" * 32, [online.Score("ttlt_s", 1.0)])  # must not raise
