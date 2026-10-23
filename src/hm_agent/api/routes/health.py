"""Health routes: liveness and readiness (are the A2A agents reachable).

Role in the system: lets a load balancer or a person check the API without calling the LLM.
No authentication, and no customer data in the responses.
"""

import asyncio

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from hm_agent.api.schemas import AgentStatus, ReadyResponse
from hm_agent.core.settings import agent_url
from hm_agent.core.state import CUSTOMER_SUPPORT, ORDER_MANAGEMENT, SHIPMENT_TRACKING

router = APIRouter()

AGENTS = (ORDER_MANAGEMENT, CUSTOMER_SUPPORT, SHIPMENT_TRACKING)
CARD_PATH = "/.well-known/agent-card.json"
CARD_TIMEOUT_S = 2.0


async def _card_ready(http: httpx.AsyncClient, agent: str) -> AgentStatus:
    """Check whether an agent's Agent Card answers.

    Args:
        http (httpx.AsyncClient): Client used for the request.
        agent (str): Agent name.

    Returns:
        AgentStatus: ``ready`` is True when the card answered with HTTP 200.
    """
    try:
        response = await http.get(agent_url(agent) + CARD_PATH)
        return AgentStatus(agent=agent, ready=response.status_code == 200)
    except httpx.HTTPError:
        return AgentStatus(agent=agent, ready=False)


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Report that the process is up.

    Returns:
        dict[str, str]: ``{"status": "ok"}``.
    """
    return {"status": "ok"}


@router.get("/readyz", response_model=ReadyResponse)
async def readyz() -> JSONResponse:
    """Report whether every A2A agent answers, so a chat can be served.

    Returns:
        JSONResponse: A ``ReadyResponse`` body; HTTP 200 when all agents are up, otherwise 503.
    """
    async with httpx.AsyncClient(timeout=CARD_TIMEOUT_S) as http:
        statuses = await asyncio.gather(*(_card_ready(http, agent) for agent in AGENTS))
    body = ReadyResponse(ready=all(s.ready for s in statuses), agents=list(statuses))
    return JSONResponse(body.model_dump(), status_code=200 if body.ready else 503)
