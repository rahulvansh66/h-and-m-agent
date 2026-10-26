"""Tests for SigV4 signing of A2A requests.

Role in the system: checks that ``SigV4Auth`` adds correct, deterministic signing headers with fake
credentials, that the body is covered by the signature, and that signing is off unless switched on.
No network and no real AWS credentials.
"""

import datetime

import httpx
import pytest
from botocore.credentials import Credentials

from hm_agent.a2a import sigv4

FIXED_NOW = datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc)
URL = "https://runtime.test/runtimes/agent/invocations"


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Freeze botocore's signing clock so signatures are reproducible.

    Args:
        monkeypatch (pytest.MonkeyPatch): Used to replace botocore's clock.
    """
    monkeypatch.setattr("botocore.auth.get_current_datetime", lambda: FIXED_NOW)


def signed_headers(body: bytes, token: str | None = None) -> httpx.Headers:
    """Send a POST through a mock transport with the signer attached and return the headers it carried.

    Args:
        body (bytes): Request body.
        token (str | None): Optional session token for the fake credentials.

    Returns:
        httpx.Headers: Headers of the request as the transport received it.
    """
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200)

    auth = sigv4.SigV4Auth("us-east-1", credentials_provider=lambda: Credentials("AKIDEXAMPLE", "secret", token))
    with httpx.Client(transport=httpx.MockTransport(handler), auth=auth) as http:
        http.post(URL, content=body, headers={"content-type": "application/json"})
    return seen[0].headers


def test_signing_headers_are_present_and_stable() -> None:
    """Authorization and X-Amz-Date are set, and the same input signs identically."""
    first = signed_headers(b'{"a": 1}')
    assert first["x-amz-date"] == "20260102T030405Z"
    assert first["authorization"].startswith("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20260102/us-east-1/bedrock-agentcore/aws4_request")
    assert "x-amz-security-token" not in first
    assert signed_headers(b'{"a": 1}')["authorization"] == first["authorization"]


def test_signature_covers_the_body() -> None:
    """A different body gives a different signature."""
    assert signed_headers(b"one")["authorization"] != signed_headers(b"two")["authorization"]


def test_session_token_is_forwarded() -> None:
    """Temporary credentials add the security token header."""
    assert signed_headers(b"x", token="session-token")["x-amz-security-token"] == "session-token"


def test_request_auth_follows_the_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """No signer when SIGV4_ENABLED is off (default); a signer when it is on."""
    monkeypatch.delenv("SIGV4_ENABLED", raising=False)
    assert sigv4.request_auth() is None
    monkeypatch.setenv("SIGV4_ENABLED", "true")
    assert isinstance(sigv4.request_auth(), sigv4.SigV4Auth)


def test_unsigned_request_has_no_signing_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the switch off, a client built with ``auth=request_auth()`` sends no signing headers."""
    monkeypatch.delenv("SIGV4_ENABLED", raising=False)
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(lambda r: (seen.append(r), httpx.Response(200))[1])
    with httpx.Client(transport=transport, auth=sigv4.request_auth()) as http:
        http.post(URL, content=b"x")
    assert "authorization" not in seen[0].headers and "x-amz-date" not in seen[0].headers
