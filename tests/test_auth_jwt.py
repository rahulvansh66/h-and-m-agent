"""Tests for the JWT verifier: valid path and every rejection path, with a stub JWKS (no network)."""

import json
import time
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from hm_agent.auth import AuthError, JwtVerifier

ISSUER = "https://tenant.example.com/"
AUDIENCE = "https://hm-agent/api"
CLAIM = "https://hm-agent/customer_id"


def _key() -> rsa.RSAPrivateKey:
    """Generate a throwaway RSA key."""
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="module")
def signing_key() -> rsa.RSAPrivateKey:
    """RSA key whose public half is published in the stub JWKS."""
    return _key()


def _jwks(key: rsa.RSAPrivateKey, kid: str = "k1") -> dict[str, Any]:
    """Build a JWKS document holding the public half of ``key``."""
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
    return {"keys": [{**jwk, "kid": kid, "alg": "RS256", "use": "sig"}]}


def _token(key: rsa.RSAPrivateKey, kid: str = "k1", **overrides: Any) -> str:
    """Sign a token that is valid unless ``overrides`` change a claim (``None`` drops it)."""
    claims: dict[str, Any] = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "exp": int(time.time()) + 300,
        CLAIM: "cust001",
    }
    claims.update(overrides)
    claims = {k: v for k, v in claims.items() if v is not None}
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": kid})


@pytest.fixture
def verifier(signing_key: rsa.RSAPrivateKey) -> JwtVerifier:
    """Verifier wired to a stub JWKS fetcher."""
    return JwtVerifier(ISSUER, AUDIENCE, CLAIM, 3600, fetcher=lambda url: _jwks(signing_key))


def test_valid_token_returns_customer(verifier: JwtVerifier, signing_key: rsa.RSAPrivateKey) -> None:
    assert verifier.verify(_token(signing_key)).customer_id == "cust001"


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"exp": int(time.time()) - 3600}, "expired"),
        ({"aud": "https://other/api"}, "bad_audience"),
        ({"iss": "https://evil.example.com/"}, "bad_issuer"),
        ({CLAIM: None}, "missing_claim"),
    ],
)
def test_claim_rejections(
    verifier: JwtVerifier, signing_key: rsa.RSAPrivateKey, overrides: dict[str, Any], reason: str
) -> None:
    with pytest.raises(AuthError) as exc:
        verifier.verify(_token(signing_key, **overrides))
    assert exc.value.reason == reason


def test_bad_signature(verifier: JwtVerifier) -> None:
    with pytest.raises(AuthError) as exc:
        verifier.verify(_token(_key()))  # signed by a key the JWKS does not hold
    assert exc.value.reason == "bad_signature"


def test_unknown_kid(verifier: JwtVerifier, signing_key: rsa.RSAPrivateKey) -> None:
    with pytest.raises(AuthError) as exc:
        verifier.verify(_token(signing_key, kid="nope"))
    assert exc.value.reason == "unknown_key"


@pytest.mark.parametrize("token", ["", "not-a-jwt"])
def test_missing_or_malformed(verifier: JwtVerifier, token: str) -> None:
    with pytest.raises(AuthError):
        verifier.verify(token)


def test_jwks_fetch_failure(signing_key: rsa.RSAPrivateKey) -> None:
    def boom(url: str) -> dict[str, Any]:
        raise OSError("down")

    with pytest.raises(AuthError) as exc:
        JwtVerifier(ISSUER, AUDIENCE, CLAIM, 3600, fetcher=boom).verify(_token(signing_key))
    assert exc.value.reason == "jwks_unavailable"


def test_jwks_is_cached(signing_key: rsa.RSAPrivateKey) -> None:
    calls: list[str] = []

    def fetch(url: str) -> dict[str, Any]:
        calls.append(url)
        return _jwks(signing_key)

    v = JwtVerifier(ISSUER, AUDIENCE, CLAIM, 3600, fetcher=fetch)
    v.verify(_token(signing_key))
    v.verify(_token(signing_key))
    assert calls == [ISSUER + ".well-known/jwks.json"]


def test_jwks_refetched_after_ttl(signing_key: rsa.RSAPrivateKey) -> None:
    now = [0.0]
    calls: list[str] = []

    def fetch(url: str) -> dict[str, Any]:
        calls.append(url)
        return _jwks(signing_key)

    v = JwtVerifier(ISSUER, AUDIENCE, CLAIM, 10, fetcher=fetch, clock=lambda: now[0])
    v.verify(_token(signing_key))
    now[0] = 11.0
    v.verify(_token(signing_key))
    assert len(calls) == 2


def test_key_rotation_refetches_on_unknown_kid(signing_key: rsa.RSAPrivateKey) -> None:
    new_key = _key()
    docs = [_jwks(signing_key, "old"), _jwks(new_key, "new")]
    v = JwtVerifier(ISSUER, AUDIENCE, CLAIM, 3600, fetcher=lambda url: docs.pop(0))
    v.verify(_token(signing_key, kid="old"))
    assert v.verify(_token(new_key, kid="new")).customer_id == "cust001"
