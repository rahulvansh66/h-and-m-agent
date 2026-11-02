"""Tool-call extraction and tool-health metrics, shared by online scoring and offline evaluation.

Role in the system: turns the message list of a finished agent run into ``ToolCall`` records (name,
arguments, success) and derives the metrics that answer "did the agent pick the right tool, send the right
arguments, and did the call succeed". Pure functions, no LLM and no network, so they are cheap to test.
"""

from dataclasses import dataclass

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage

# Tools each MCP-backed agent is allowed to call; a call outside this set means the model picked a wrong tool.
AGENT_TOOLS: dict[str, frozenset[str]] = {
    "order_management": frozenset({"get_last_bought_product", "get_purchase_count", "list_orders", "get_order_details"}),
    "fashion": frozenset({"get_style_profile", "get_purchase_history", "get_styling_guide", "search_products"}),
}

# Phrases a tool wrapper puts in the result when it rejected the arguments (pydantic / MCP validation).
_INVALID_ARGS_MARKERS = ("validation error", "field required", "invalid arguments", "missing required", "input should be")


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation made by an agent.

    Attributes:
        name (str): Tool name.
        args (dict): Arguments the model sent.
        ok (bool): The tool returned without an error status.
        invalid_args (bool): The tool result says the arguments were rejected.
    """

    name: str
    args: dict
    ok: bool
    invalid_args: bool


def extract_tool_calls(messages: list[AnyMessage]) -> list[ToolCall]:
    """Pair every tool call in an agent run with its result.

    Args:
        messages (list[AnyMessage]): Messages of the run (AI messages carry ``tool_calls``, tool messages the results).

    Returns:
        list[ToolCall]: One record per tool call, in call order. A call with no result message counts as failed.
    """
    results = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
    calls: list[ToolCall] = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls:
            result = results.get(call["id"])
            text = result.text.lower() if result is not None else ""
            calls.append(
                ToolCall(
                    name=call["name"],
                    args=call.get("args") or {},
                    ok=result is not None and result.status != "error",
                    invalid_args=any(marker in text for marker in _INVALID_ARGS_MARKERS),
                )
            )
    return calls


def redundant_calls(calls: list[ToolCall]) -> int:
    """Count calls that repeat an earlier call with the same tool and arguments.

    Args:
        calls (list[ToolCall]): Tool calls of one run.

    Returns:
        int: Number of repeated calls (wasted latency and tokens).
    """
    seen: set[tuple[str, str]] = set()
    repeats = 0
    for call in calls:
        key = (call.name, repr(sorted(call.args.items())))
        repeats += key in seen
        seen.add(key)
    return repeats


def tool_health_scores(calls: list[ToolCall], agent: str) -> dict[str, float]:
    """Compute the online tool-health metrics for one agent run.

    The correct-tool check is a proxy without ground truth: every called tool must belong to the agent's own tool set.

    Args:
        calls (list[ToolCall]): Tool calls of the run (see ``extract_tool_calls``).
        agent (str): Agent name, used to look up its allowed tools in ``AGENT_TOOLS``.

    Returns:
        dict[str, float]: ``tool_call_count``, ``tool_success_rate``, ``tool_error_count``, ``tool_in_allowlist``
        (1.0 when all calls are allowed), ``tool_args_valid`` (1.0 when no call was rejected for its arguments)
        and ``redundant_tool_calls``. ``tool_success_rate`` is omitted when no tool was called.
    """
    allowed = AGENT_TOOLS.get(agent, frozenset())
    scores: dict[str, float] = {
        "tool_call_count": float(len(calls)),
        "tool_error_count": float(sum(not c.ok for c in calls)),
        "tool_in_allowlist": float(all(c.name in allowed for c in calls)),
        "tool_args_valid": float(not any(c.invalid_args for c in calls)),
        "redundant_tool_calls": float(redundant_calls(calls)),
    }
    if calls:
        scores["tool_success_rate"] = sum(c.ok for c in calls) / len(calls)
    return scores
