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
REDACTED_KEYS = frozenset({"user_token"})  # secrets that must never reach the log file


def _redact(value: object) -> object:
    """Replace the values of secret keys anywhere inside parsed JSON.

    Args:
        value (object): Parsed JSON (dict, list or scalar).

    Returns:
        object: Copy with every ``REDACTED_KEYS`` value replaced by ``"***"``.
    """
    if isinstance(value, dict):
        return {k: "***" if k in REDACTED_KEYS else _redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


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
        "body": _redact(parsed),
    }
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


async def _log_request(request: httpx.Request) -> None:
    """httpx hook: record an outgoing request.

    Args:
        request (httpx.Request): Request about to be sent.
    """
    _write_wire("C->S", f"{request.method} {request.url}", request.content.decode("utf-8", "replace"))


class _SseLoggingStream(httpx.AsyncByteStream):
    """Response body wrapper that logs each Server-Sent-Events ``data:`` line as it streams past.

    The bytes are passed on untouched and immediately, so the caller still sees events live.
    """

    def __init__(self, inner: httpx.AsyncByteStream, url: str, status: int) -> None:
        """Wrap a response body stream.

        Args:
            inner (httpx.AsyncByteStream): The original body stream.
            url (str): URL of the response, recorded with each event.
            status (int): HTTP status code of the response.
        """
        self._inner = inner
        self._url = url
        self._status = status
        self._pending = ""  # text of an incomplete line carried over to the next network chunk

    async def __aiter__(self):
        """Yield the original chunks, logging each complete ``data:`` line.

        Yields:
            bytes: The unmodified body chunks.
        """
        async for chunk in self._inner:
            self._pending += chunk.decode("utf-8", "replace")
            *lines, self._pending = self._pending.split("\n")
            for line in lines:
                if line.startswith("data:"):
                    _write_wire("S->C SSE", self._url, line[5:].strip(), self._status)
            yield chunk

    async def aclose(self) -> None:
        """Close the wrapped stream."""
        await self._inner.aclose()


async def _log_response(response: httpx.Response) -> None:
    """httpx hook: record an incoming response.

    A normal body is read and logged whole. An event stream must not be read here (that would wait
    for the end of the stream), so its body is wrapped to log each event as it arrives.

    Args:
        response (httpx.Response): Response just received.
    """
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        response.stream = _SseLoggingStream(response.stream, str(response.request.url), response.status_code)
        return
    await response.aread()
    _write_wire("S->C", str(response.request.url), response.text, response.status_code)


def log_push_notification(url: str, body: str) -> None:
    """Record a push notification received from an agent (agent -> supervisor direction).

    Args:
        url (str): URL of the receiver endpoint the agent called.
        body (str): Raw HTTP body text of the notification.
    """
    _write_wire("PUSH A->S", f"POST {url}", body)


def wire_log_hooks() -> dict[str, list]:
    """Build the httpx ``event_hooks`` that feed the wire log.

    Returns:
        dict[str, list]: Request/response hooks when ``A2A_WIRE_LOG`` is set, otherwise empty.
    """
    if not os.environ.get(WIRE_LOG_ENV):
        return {}
    return {"request": [_log_request], "response": [_log_response]}
