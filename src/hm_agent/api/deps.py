"""FastAPI dependencies: API-key check, customer identity, rate limit and access to shared services.

Role in the system: the authentication and abuse-protection layer of ``hm_agent.api``. The API key
proves the caller is a trusted client; the customer id header says which customer it acts for.
"""

import hmac
import os
import re

from fastapi import Depends, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from hm_agent.api.service import Services, thread_owner

API_KEY_ENV = "API_KEY"  # name of the env var holding the shared secret
CUSTOMER_ID_PATTERN = re.compile(r"^cust\d{3,}$")  # shape of customer ids, e.g. cust001


def get_services(request: Request) -> Services:
    """Return the shared services created in the app lifespan.

    Args:
        request (Request): Current request.

    Returns:
        Services: Graph, rate limiter, locks and push hub.
    """
    return request.app.state.services


def require_api_key(x_api_key: str = Header(default="")) -> None:
    """Reject requests whose ``X-API-Key`` header does not match the configured key.

    Fails closed: when no key is configured on the server every request is rejected.

    Args:
        x_api_key (str): Value of the ``X-API-Key`` header.

    Raises:
        HTTPException: 401 when the key is missing or wrong.
    """
    expected = os.environ.get(API_KEY_ENV, "")
    if not expected or not hmac.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="invalid or missing API key")


def get_customer_id(x_customer_id: str = Header()) -> str:
    """Read and validate the customer the request acts for.

    Args:
        x_customer_id (str): Value of the ``X-Customer-Id`` header, e.g. ``cust001``.

    Returns:
        str: The validated customer id.

    Raises:
        HTTPException: 422 when the id does not look like ``cust<digits>``.
    """
    if not CUSTOMER_ID_PATTERN.match(x_customer_id):
        raise HTTPException(status_code=422, detail="invalid X-Customer-Id")
    return x_customer_id


def enforce_rate_limit(
    customer_id: str = Depends(get_customer_id), services: Services = Depends(get_services)
) -> None:
    """Apply the per-customer rate limit.

    Args:
        customer_id (str): Customer making the request.
        services (Services): Shared services holding the limiter.

    Raises:
        HTTPException: 429 with a ``Retry-After`` header when the customer is over the limit.
    """
    retry_after = services.limiter.check(customer_id)
    if retry_after is not None:
        raise HTTPException(
            status_code=429, detail="too many requests", headers={"Retry-After": str(int(retry_after) + 1)}
        )


async def require_thread_access(thread_id: str, customer_id: str, services: Services, must_exist: bool) -> None:
    """Check that a thread belongs to the customer, hiding other customers' threads as missing.

    Args:
        thread_id (str): Conversation thread id.
        customer_id (str): Customer making the request.
        services (Services): Shared services holding the graph.
        must_exist (bool): True to also fail when the thread has no stored state (read endpoints);
            False to allow a thread id that has not been used yet (starting a new thread).

    Raises:
        HTTPException: 404 when the thread is another customer's, or is missing and ``must_exist``.
    """
    owner = await run_in_threadpool(thread_owner, services.graph, thread_id)
    if (owner is None and must_exist) or (owner is not None and owner != customer_id):
        raise HTTPException(status_code=404, detail="thread not found")
