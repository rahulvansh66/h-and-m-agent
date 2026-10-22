"""Optional A2A wire log: records every HTTP request/response body the client exchanges.

Role in the system: debug aid for the A2A client (``hm_agent.a2a.client``). Setting ``A2A_WIRE_LOG``
to a file path appends each request/response body there as one JSON line, the A2A counterpart of an
MCP wire log. Debug only: the log contains customer text, so keep it off in production.
"""

import json
import os
from datetime import datetime, timezone

import httpx

WIRE_LOG_ENV = "A2A_WIRE_LOG"  # path of the optional JSONL wire log


def _write_wire(direction: str, url: str, body: str, status: int | None = None) -> None:
    """Append one request/response record to the wire log, if enabled.

    Args:
        direction (str): ``C->S`` for a request, ``S->C`` for a response.
        url (str): URL the record belongs to.
        body (str): Raw HTTP body text (parsed as JSON when possible).
        status (int | None): HTTP status code for responses, None for requests.
    """
    path = os.environ.get(WIRE_LOG_ENV)
    if not path:
        return
    try:
        parsed = json.loads(body) if body else None
    except ValueError:
        parsed = body
    record = {
        "time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "dir": direction,
        "url": url,
        "http_status": status,
        "body": parsed,
    }
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


async def _log_request(request: httpx.Request) -> None:
    """httpx hook: record an outgoing request.

    Args:
        request (httpx.Request): Request about to be sent.
    """
    _write_wire("C->S", f"{request.method} {request.url}", request.content.decode("utf-8", "replace"))


async def _log_response(response: httpx.Response) -> None:
    """httpx hook: record an incoming response.

    Args:
        response (httpx.Response): Response just received.
    """
    await response.aread()
    _write_wire("S->C", str(response.request.url), response.text, response.status_code)


def wire_log_hooks() -> dict[str, list]:
    """Build the httpx ``event_hooks`` that feed the wire log.

    Returns:
        dict[str, list]: Request/response hooks when ``A2A_WIRE_LOG`` is set, otherwise empty.
    """
    if not os.environ.get(WIRE_LOG_ENV):
        return {}
    return {"request": [_log_request], "response": [_log_response]}
