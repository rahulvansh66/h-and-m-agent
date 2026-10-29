"""Supervisor as an AgentCore Runtime app: an LLM routes the message to the order agent (MCP) or shipment agent (A2A)."""

import json
import logging
import os
import re
import time

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.helpers.proto_helpers import get_artifact_text, get_message_text, new_text_part
from a2a.types.a2a_pb2 import Message, Role, SendMessageRequest, TaskState
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from aws_common import SESSION_HEADER, SigV4Auth, env_flag, new_session_id
from llm import chat, groq_enabled

logging.basicConfig(level=logging.INFO, format="[supervisor] %(message)s")
logger = logging.getLogger(__name__)

# Where the other pieces live. Local defaults: direct URLs, no signing. On AWS Terraform sets all four.
ORDER_MCP_URL = os.environ.get("ORDER_MCP_URL", "http://127.0.0.1:8001/mcp")   # AWS: the Gateway URL (ends in /mcp)
ORDER_MCP_SIGV4 = env_flag("ORDER_MCP_SIGV4")                                   # AWS: true (Gateway uses IAM auth)
SHIPMENT_A2A_URL = os.environ.get("SHIPMENT_A2A_URL", "http://127.0.0.1:9000")  # AWS: shipment Runtime URL
SHIPMENT_SIGV4 = env_flag("SHIPMENT_SIGV4")                                     # AWS: true (Runtime uses IAM auth)
ORDER_ID_PATTERN = re.compile(r"\bORD-\d+\b", re.IGNORECASE)
AGENTS = ("order_management", "shipment_tracking", "out_of_scope")

ROUTER_PROMPT = """You are the supervisor of the H&M customer support team. You never answer the customer; you only choose
which specialist handles the latest message. Reply with JSON only: {"agent": "<name>", "reason": "<one short sentence>"}.
Agents:
- order_management: the customer's own orders or purchases (what they bought, last purchase, order history, order contents).
- shipment_tracking: where an order or parcel is now, shipping or delivery status.
- out_of_scope: anything else."""

ORDER_AGENT_PROMPT = """You are the H&M order management agent. Answer the customer's question by calling exactly one of the
available tools. The customer is already identified by the system: never ask for or use a customer id.
If the customer names an order id (like ORD-1007) use it. Never invent order data.
After you receive the tool result, reply in 1-3 friendly sentences."""

app = BedrockAgentCoreApp()  # provides GET /ping and POST /invocations


def route_by_keywords(message: str) -> str:
    """Fallback router (no LLM): decide the agent from keywords.

    Args:
        message: The customer's message.

    Returns:
        str: `shipment_tracking`, `order_management` or `out_of_scope`.
    """
    text = message.lower()
    if any(word in text for word in ("where", "track", "deliver", "parcel", "shipping", "shipment")):
        return "shipment_tracking"
    if any(word in text for word in ("order", "bought", "buy", "purchase")):
        return "order_management"
    return "out_of_scope"


async def route(message: str) -> str:
    """Decide which agent should handle the message: ask the LLM, fall back to keyword rules if that is not possible.

    Args:
        message: The customer's message.

    Returns:
        str: One of `AGENTS`.
    """
    if groq_enabled():
        try:
            reply = await chat([{"role": "system", "content": ROUTER_PROMPT}, {"role": "user", "content": message}], json_mode=True)
            decision = json.loads(reply["content"])
            if decision.get("agent") in AGENTS:
                logger.info("route source=llm agent=%s", decision["agent"])
                return decision["agent"]
            logger.warning("route source=llm status=invalid_agent")
        except Exception:  # rate limit, network, bad JSON: a cheap rule is better than failing the customer's request
            logger.exception("route source=llm status=failed, using keyword rules")
    return route_by_keywords(message)


def find_tool(tool_names: list[str], wanted: str) -> str:
    """Find a tool by name, also when a Gateway added a target prefix (`order-management___list_orders`).

    Args:
        tool_names: Names returned by `list_tools()`.
        wanted: The plain tool name, e.g. `list_orders`.

    Returns:
        str: The real name to call.

    Raises:
        LookupError: If no tool matches.
    """
    for name in tool_names:
        if name == wanted or name.endswith(f"___{wanted}"):
            return name
    raise LookupError(f"tool {wanted!r} not offered; available: {tool_names}")


def tool_spec_without_customer(tool) -> dict:
    """Turn an MCP tool into an LLM function spec that hides `customer_id` from the model.

    The model must not choose whose data is read; the system supplies the customer id itself.

    Args:
        tool: An MCP tool as returned by `list_tools()`.

    Returns:
        dict: OpenAI-style function spec (name, description, JSON-schema parameters).
    """
    schema = json.loads(json.dumps(tool.inputSchema))  # deep copy
    schema.get("properties", {}).pop("customer_id", None)
    schema["required"] = [name for name in schema.get("required", []) if name != "customer_id"]
    return {"type": "function", "function": {"name": tool.name, "description": tool.description, "parameters": schema}}


async def llm_order_agent(session: ClientSession, tools: list, message: str, customer_id: str) -> str:
    """LLM order agent: the model picks an MCP tool from its schema, we call it, the model words the answer.

    Args:
        session: An initialised MCP session (to the MCP server or the Gateway).
        tools: Tools from `session.list_tools()`.
        message: The customer's message.
        customer_id: Customer id supplied by the system.

    Returns:
        str: The final reply text.
    """
    messages = [{"role": "system", "content": ORDER_AGENT_PROMPT}, {"role": "user", "content": message}]
    reply = await chat(messages, tools=[tool_spec_without_customer(t) for t in tools])
    calls = reply.get("tool_calls")
    if not calls:
        return reply.get("content") or "Sorry, I could not find that information."

    call = calls[0]
    args = json.loads(call["function"].get("arguments") or "{}")
    args["customer_id"] = customer_id   # always overwritten by the system, whatever the model wrote
    logger.info("order_agent source=llm tool=%s", call["function"]["name"])
    result = await session.call_tool(call["function"]["name"], args)
    tool_text = result.content[0].text

    messages += [
        {"role": "assistant", "content": reply.get("content") or "", "tool_calls": calls},
        {"role": "tool", "tool_call_id": call["id"], "content": tool_text},
    ]
    final = await chat(messages)
    return final.get("content") or tool_text


async def run_order_agent(message: str, customer_id: str) -> str:
    """Order agent: connect to the MCP endpoint (the Gateway on AWS) and answer with the LLM, or with rules as fallback.

    Args:
        message: The customer's message.
        customer_id: Customer id supplied by the system (never by the model).

    Returns:
        str: The reply text.
    """
    auth = SigV4Auth() if ORDER_MCP_SIGV4 else None
    async with streamablehttp_client(ORDER_MCP_URL, auth=auth) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            if groq_enabled():
                try:
                    return await llm_order_agent(session, tools, message, customer_id)
                except Exception:
                    logger.exception("order_agent source=llm status=failed, using keyword rules")
            names = [t.name for t in tools]
            order_id = ORDER_ID_PATTERN.search(message)
            if order_id:
                wanted, args = "get_order_details", {"order_id": order_id.group(0).upper()}
            elif "last" in message.lower():
                wanted, args = "get_last_bought_product", {}
            else:
                wanted, args = "list_orders", {}
            tool = find_tool(names, wanted)
            logger.info("order_agent source=rules tool=%s", tool)
            result = await session.call_tool(tool, {"customer_id": customer_id, **args})
            return result.content[0].text


async def run_shipment_agent(message: str, customer_id: str):
    """Call the shipment agent over A2A and yield its progress lines and answer pieces.

    Args:
        message: The customer's message.
        customer_id: Customer id, forwarded as request metadata.

    Yields:
        tuple[str, str]: `("progress", text)` or `("answer", text)`.
    """
    auth = SigV4Auth() if SHIPMENT_SIGV4 else None
    async with httpx.AsyncClient(timeout=60, auth=auth, headers={SESSION_HEADER: new_session_id()}) as http:
        card = await A2ACardResolver(http, SHIPMENT_A2A_URL).get_agent_card()
        card.supported_interfaces[0].url = SHIPMENT_A2A_URL  # the card cannot know its own Runtime URL
        client = ClientFactory(ClientConfig(streaming=card.capabilities.streaming, httpx_client=http)).create(card)
        request = SendMessageRequest(
            message=Message(message_id=f"msg-{time.time_ns()}", role=Role.ROLE_USER, parts=[new_text_part(message)]),
            metadata={"customer_id": customer_id},
        )
        async for response in client.send_message(request):
            if response.HasField("status_update"):
                status = response.status_update.status
                if status.state == TaskState.TASK_STATE_WORKING and status.message.parts:
                    yield "progress", get_message_text(status.message)
            elif response.HasField("artifact_update"):
                yield "answer", get_artifact_text(response.artifact_update.artifact, delimiter="")
        await client.close()


@app.entrypoint
async def invoke(payload: dict, context):
    """Runtime entrypoint: handle one customer message and stream events.

    Args:
        payload: Request body, `{"message": "...", "customer_id": "cust001"}`.
        context: Runtime request context; `context.session_id` is the conversation id.

    Yields:
        dict: Events: `agent_selected`, `progress`, `reply`, then `done`.
    """
    message = str(payload.get("message", ""))
    customer_id = str(payload.get("customer_id", "cust001"))
    agent = await route(message)
    logger.info("session=%s agent=%s", getattr(context, "session_id", ""), agent)
    yield {"type": "agent_selected", "agent": agent}

    try:
        if agent == "order_management":
            yield {"type": "reply", "agent": agent, "text": await run_order_agent(message, customer_id)}
        elif agent == "shipment_tracking":
            reply = ""
            async for kind, text in run_shipment_agent(message, customer_id):
                if kind == "progress":
                    yield {"type": "progress", "agent": agent, "text": text}
                else:
                    reply += text
            yield {"type": "reply", "agent": agent, "text": reply}
        else:
            yield {"type": "reply", "agent": agent, "text": "I can help with orders and parcel tracking."}
    except Exception as exc:  # show the cause while we are learning; a real service would return a generic message
        logger.exception("agent=%s failed", agent)
        yield {"type": "error", "agent": agent, "text": f"{type(exc).__name__}: {exc}"}
    yield {"type": "done"}


if __name__ == "__main__":
    # Local defaults 127.0.0.1:8090 (8080 is often taken); on Runtime Terraform sets HOST=0.0.0.0, PORT=8080.
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8090")))
