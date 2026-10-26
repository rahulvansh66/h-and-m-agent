"""SigV4 request signing for A2A calls to AgentCore Runtimes.

Role in the system: Runtime agent endpoints use IAM (SigV4) inbound auth, so the supervisor must sign every
A2A HTTP request (card fetch, ``SendMessage``, ``GetTask``) with its IAM credentials. ``SigV4Auth`` is an
``httpx.Auth`` plugged into the A2A client's ``httpx.AsyncClient``; ``request_auth`` returns it only when
signing is switched on, so localhost calls stay unsigned.
"""

import logging
from collections.abc import Callable, Generator

import httpx
from botocore.auth import SigV4Auth as BotocoreSigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials
from botocore.session import get_session

from hm_agent.core import settings

logger = logging.getLogger(__name__)

SERVICE_NAME = "bedrock-agentcore"  # signing name of AgentCore Runtime
# Only these request headers are signed. Proxies and HTTP clients add or rewrite the others
# (Accept-Encoding, Connection, ...), which would invalidate a signature that covered them.
_SIGNED_REQUEST_HEADERS = ("content-type",)
# Headers botocore produces that must be copied back onto the httpx request.
_RESULT_HEADERS = ("Authorization", "X-Amz-Date", "X-Amz-Security-Token")

CredentialsProvider = Callable[[], Credentials]


def _default_credentials() -> Credentials:
    """Resolve the caller's IAM credentials through the standard AWS chain (env, profile, role).

    Returns:
        Credentials: Frozen credentials valid right now (refreshed by botocore when temporary).

    Raises:
        RuntimeError: If no credentials can be found.
    """
    credentials = get_session().get_credentials()
    if credentials is None:
        raise RuntimeError("SigV4 signing is enabled but no AWS credentials were found")
    return credentials.get_frozen_credentials()


class SigV4Auth(httpx.Auth):
    """``httpx.Auth`` that signs each request with AWS Signature Version 4.

    The exact request body bytes are signed, so the body must be readable before sending
    (``requires_request_body``). Streaming *responses* are unaffected: only the request is signed.

    Args:
        region (str): AWS region the target Runtime lives in.
        service (str): SigV4 signing name. Defaults to ``bedrock-agentcore``.
        credentials_provider (CredentialsProvider | None): Returns the credentials to sign with, called
            once per request so rotated temporary credentials are picked up. Injectable for tests;
            defaults to the standard AWS credential chain.
    """

    requires_request_body = True

    def __init__(
        self,
        region: str,
        service: str = SERVICE_NAME,
        credentials_provider: CredentialsProvider | None = None,
    ) -> None:
        self._region = region
        self._service = service
        self._credentials_provider = credentials_provider or _default_credentials

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        """Add the SigV4 headers to a request before it is sent.

        Args:
            request (httpx.Request): Outgoing request, body already read.

        Yields:
            httpx.Request: The same request with ``Authorization``, ``X-Amz-Date`` and, for temporary
            credentials, ``X-Amz-Security-Token`` set.

        Raises:
            RuntimeError: If no AWS credentials are available.
        """
        signed_headers = {n: v for n, v in request.headers.items() if n.lower() in _SIGNED_REQUEST_HEADERS}
        aws_request = AWSRequest(method=request.method, url=str(request.url), data=request.content, headers=signed_headers)
        BotocoreSigV4Auth(self._credentials_provider(), self._service, self._region).add_auth(aws_request)
        for name in _RESULT_HEADERS:
            if name in aws_request.headers:
                request.headers[name] = aws_request.headers[name]
        logger.debug("component=sigv4 status=signed method=%s", request.method)
        yield request


def request_auth() -> SigV4Auth | None:
    """Return the signer for outgoing A2A requests, or None when signing is off.

    Read at call time so the ``SIGV4_ENABLED`` switch (and ``.env``) take effect after import.

    Returns:
        SigV4Auth | None: Signer for the configured AWS region when ``settings.sigv4_enabled()``, else None.
    """
    return SigV4Auth(settings.AWS_REGION) if settings.sigv4_enabled() else None
