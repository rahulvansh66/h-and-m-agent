"""Helpers shared by the supervisor and the notebook: env flags and SigV4 request signing."""

import os
import uuid

import httpx
from botocore.auth import SigV4Auth as BotocoreSigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.session import Session

SESSION_HEADER = "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"  # Runtime wants a session id of 33+ characters


def env_flag(name: str) -> bool:
    """Read a boolean switch from the environment.

    Args:
        name: Environment variable name.

    Returns:
        bool: True if the variable is set to true/1/yes.
    """
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def new_session_id() -> str:
    """Create a Runtime session id (must be at least 33 characters).

    Returns:
        str: A random 36-character id.
    """
    return str(uuid.uuid4())


class SigV4Auth(httpx.Auth):
    """httpx authentication that signs each request with the current AWS credentials (SigV4)."""

    requires_request_body = True  # the exact body bytes are part of the signature

    def __init__(self, service: str = "bedrock-agentcore", region: str | None = None) -> None:
        """Remember what to sign for.

        Args:
            service: AWS signing name. Gateway and Runtime both use `bedrock-agentcore`.
            region: AWS region; defaults to AWS_REGION / AWS_DEFAULT_REGION / us-east-1.
        """
        self._service = service
        self._region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
        self._session = Session()  # resolves credentials: env vars, profile, or the Runtime's role

    def auth_flow(self, request: httpx.Request):
        """Add the SigV4 headers to the request before it is sent.

        Args:
            request: The outgoing httpx request.

        Yields:
            httpx.Request: The same request, now carrying Authorization / X-Amz-* headers.
        """
        credentials = self._session.get_credentials().get_frozen_credentials()
        # Only content-type is signed besides host/x-amz-*: proxies may rewrite other headers.
        to_sign = {k: v for k, v in request.headers.items() if k.lower() == "content-type"}
        aws_request = AWSRequest(method=request.method, url=str(request.url), data=request.content, headers=to_sign)
        BotocoreSigV4Auth(credentials, self._service, self._region).add_auth(aws_request)
        for name in ("Authorization", "X-Amz-Date", "X-Amz-Security-Token"):
            if name in aws_request.headers:
                request.headers[name] = aws_request.headers[name]
        yield request
