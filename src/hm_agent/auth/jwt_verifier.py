"""Verify an OIDC access token (JWT) and extract the verified customer id.

Role in the system: the single trust check for end-user identity. The supervisor entrypoint and the agent
services call ``verify_token``; nothing else decodes tokens. Signing keys come from the issuer's JWKS
document and are cached in memory so there is no network fetch per call.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
import jwt
from jwt import PyJWK

from hm_agent.core import settings

logger = logging.getLogger(__name__)

JwksFetcher = Callable[[str], dict[str, Any]]  # url -> parsed JWKS document

_ALGORITHMS = ["RS256"]  # asymmetric only; never accept HS* (key confusion) or "none"
_LEEWAY_S = 30  # tolerated clock skew between issuer and this service


class AuthError(Exception):
    """Raised when a token is missing, malformed, untrusted, expired or lacks the customer claim.

    Attributes:
        reason (str): Short machine-readable code, safe to log (never contains token data).
    """

    def __init__(self, reason: str) -> None:
        """Create the error.

        Args:
            reason (str): Short code such as ``expired`` or ``bad_audience``.
        """
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class VerifiedUser:
    """Identity extracted from a verified token.

    Attributes:
        customer_id (str): Value of the namespaced customer claim.
    """

    customer_id: str


def _http_fetch(url: str) -> dict[str, Any]:
    """Fetch a JWKS document over HTTPS.

    Args:
        url (str): Location of the JWKS document.

    Returns:
        dict[str, Any]: Parsed JSON body.

    Raises:
        httpx.HTTPError: If the request fails or returns an error status.
    """
    response = httpx.get(url, timeout=5.0)
    response.raise_for_status()
    return response.json()


class JwtVerifier:
    """Verifies access tokens against one issuer, audience and customer claim.

    Args:
        issuer (str): Expected ``iss``; the JWKS is read from ``<issuer>.well-known/jwks.json``.
        audience (str): Expected ``aud``.
        customer_claim (str): Name of the claim carrying the customer id.
        cache_ttl_s (float): Seconds a fetched JWKS is reused.
        fetcher (JwksFetcher | None): Function returning the JWKS for a URL; injectable so tests need
            no network. Defaults to an HTTPS GET.
        clock (Callable[[], float]): Time source for the cache; injectable for tests.
    """

    def __init__(
        self,
        issuer: str,
        audience: str,
        customer_claim: str,
        cache_ttl_s: float,
        fetcher: JwksFetcher | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._issuer = issuer
        self._audience = audience
        self._customer_claim = customer_claim
        self._cache_ttl_s = cache_ttl_s
        self._fetcher = fetcher or _http_fetch
        self._clock = clock
        self._jwks_url = issuer.rstrip("/") + "/.well-known/jwks.json"
        self._keys: dict[str, PyJWK] = {}  # kid -> signing key
        self._fetched_at: float | None = None

    def _load_keys(self) -> None:
        """Fetch the JWKS and replace the cached keys.

        Raises:
            AuthError: With reason ``jwks_unavailable`` if the document cannot be fetched or parsed.
        """
        try:
            document = self._fetcher(self._jwks_url)
            keys = {k["kid"]: PyJWK(k) for k in document["keys"] if k.get("kid")}
        except Exception as exc:  # network, JSON or key-format failure: all mean "cannot trust"
            logger.warning("component=auth status=jwks_fetch_failed error=%s", type(exc).__name__)
            raise AuthError("jwks_unavailable") from exc
        self._keys = keys
        self._fetched_at = self._clock()
        logger.info("component=auth status=jwks_loaded keys=%d", len(keys))

    def _signing_key(self, kid: str) -> PyJWK:
        """Return the signing key for a key id.

        Refreshes the cache when it is older than the TTL, and once more when the kid is unknown
        (the issuer may have rotated keys).

        Args:
            kid (str): ``kid`` from the token header.

        Returns:
            PyJWK: The matching key.

        Raises:
            AuthError: ``jwks_unavailable`` if keys cannot be loaded, ``unknown_key`` if none matches.
        """
        stale = self._fetched_at is None or self._clock() - self._fetched_at > self._cache_ttl_s
        if stale:
            self._load_keys()
        if kid not in self._keys and not stale:
            self._load_keys()  # possible rotation: one refresh before giving up
        key = self._keys.get(kid)
        if key is None:
            raise AuthError("unknown_key")
        return key

    def verify(self, token: str) -> VerifiedUser:
        """Verify a token and return the identity it carries.

        Checks signature, issuer, audience, expiry and the presence of the customer claim.

        Args:
            token (str): Raw bearer token (without the ``Bearer `` prefix).

        Returns:
            VerifiedUser: The verified customer.

        Raises:
            AuthError: On any failure; ``reason`` says why. Token and claim values are never logged.
        """
        try:
            user = self._verify(token)
        except AuthError as exc:
            logger.warning("component=auth status=rejected reason=%s", exc.reason)
            raise
        logger.info("component=auth status=ok")
        return user

    def _verify(self, token: str) -> VerifiedUser:
        """Do the verification without logging (see ``verify``).

        Args:
            token (str): Raw bearer token.

        Returns:
            VerifiedUser: The verified customer.

        Raises:
            AuthError: On any verification failure.
        """
        if not token:
            raise AuthError("missing_token")
        try:
            kid = jwt.get_unverified_header(token).get("kid")
        except jwt.PyJWTError as exc:
            raise AuthError("malformed") from exc
        if not kid:
            raise AuthError("malformed")

        key = self._signing_key(kid)
        try:
            claims = jwt.decode(
                token,
                key.key,
                algorithms=_ALGORITHMS,
                audience=self._audience,
                issuer=self._issuer,
                leeway=_LEEWAY_S,
                options={"require": ["exp", "iss", "aud"]},
            )
        except jwt.ExpiredSignatureError as exc:
            raise AuthError("expired") from exc
        except jwt.InvalidAudienceError as exc:
            raise AuthError("bad_audience") from exc
        except jwt.InvalidIssuerError as exc:
            raise AuthError("bad_issuer") from exc
        except jwt.InvalidSignatureError as exc:
            raise AuthError("bad_signature") from exc
        except jwt.PyJWTError as exc:
            raise AuthError("invalid_token") from exc

        customer_id = claims.get(self._customer_claim)
        if not isinstance(customer_id, str) or not customer_id:
            raise AuthError("missing_claim")
        return VerifiedUser(customer_id=customer_id)


_default_verifier: JwtVerifier | None = None


def verify_token(token: str) -> VerifiedUser:
    """Verify a token using the issuer, audience and claim from the settings.

    Args:
        token (str): Raw bearer token (without the ``Bearer `` prefix).

    Returns:
        VerifiedUser: The verified customer.

    Raises:
        AuthError: If the token is rejected for any reason.
    """
    global _default_verifier
    if _default_verifier is None:
        _default_verifier = JwtVerifier(
            issuer=settings.auth_issuer(),
            audience=settings.AUTH_AUDIENCE,
            customer_claim=settings.AUTH_CUSTOMER_CLAIM,
            cache_ttl_s=settings.AUTH_JWKS_CACHE_TTL_S,
        )
    return _default_verifier.verify(token)
