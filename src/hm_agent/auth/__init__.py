"""End-user authentication: OIDC access-token verification.

Role in the system: lets the supervisor entrypoint and the agent services learn the customer from a
verified token instead of trusting a value supplied by a caller.
"""

from hm_agent.auth.jwt_verifier import AuthError, JwtVerifier, VerifiedUser, verify_token

# Where the user's access token travels between the supervisor and an agent.
# The run-config key starts with "__" so LangGraph keeps it out of checkpoint metadata (and so out of Neon).
USER_TOKEN_CONFIG_KEY = "__user_token"
USER_TOKEN_METADATA_KEY = "user_token"  # key in the A2A request metadata

__all__ = [
    "AuthError",
    "JwtVerifier",
    "VerifiedUser",
    "verify_token",
    "USER_TOKEN_CONFIG_KEY",
    "USER_TOKEN_METADATA_KEY",
]
