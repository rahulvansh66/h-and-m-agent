"""Auth0 browser login (Universal Login) for local development.

Role in the system: a small standalone Flask app that signs a person in through Auth0 with the
official Auth0 Python SDK and shows their profile. It is separate from the FastAPI service, which
only verifies bearer tokens (see ``jwt_verifier``). Run it with ``uv run python -m hm_agent.auth.web_login``.

Reads ``AUTH0_DOMAIN``, ``AUTH0_CLIENT_ID``, ``AUTH0_CLIENT_SECRET``, ``AUTH0_SECRET`` (cookie
encryption key) and ``APP_BASE_URL`` from the environment (``.env``).
"""

import json
import logging
from os import environ as env
from typing import Any
from urllib.parse import urlparse

from auth0_server_python.auth_server.server_client import ServerClient
from auth0_server_python.auth_types import LogoutOptions, StartInteractiveLoginOptions, StateData, TransactionData
from auth0_server_python.store.abstract import AbstractDataStore
from dotenv import load_dotenv
from flask import Flask, Response, after_this_request, redirect, request
from markupsafe import escape

from hm_agent.core.settings import AUTH_AUDIENCE

load_dotenv()

logger = logging.getLogger(__name__)

app = Flask(__name__)


class CookieStore(AbstractDataStore):
    """Keeps SDK state in an encrypted, HTTP-only cookie.

    The SDK needs one store for the session (user profile, tokens) and one for short-lived login
    data (PKCE verifier, state). Both are cookie-backed here, so the app holds no server-side state.
    """

    def __init__(self, secret: str, cookie_name: str, max_age: int, model: type) -> None:
        """Create a cookie-backed store.

        Args:
            secret (str): Key used to encrypt the cookie value.
            cookie_name (str): Name of the cookie.
            max_age (int): Cookie lifetime in seconds.
            model (type): Pydantic model the stored data is validated against.
        """
        super().__init__({"secret": secret})
        self.cookie_name = cookie_name
        self.max_age = max_age
        self.model = model

    async def set(self, identifier: str, state: Any, **_: Any) -> None:
        """Write the encrypted state to the response cookie.

        Args:
            identifier (str): Key the SDK uses to bind the data (used for encryption).
            state (Any): Data to store; a pydantic model or a dict.
        """

        @after_this_request
        def apply(response: Response) -> Response:
            data = state.model_dump() if hasattr(state, "model_dump") else state
            response.set_cookie(
                self.cookie_name,
                self.encrypt(identifier, data),
                httponly=True,
                samesite="Lax",
                secure=not env.get("APP_BASE_URL", "").startswith("http://"),
                max_age=self.max_age,
            )
            return response

    async def get(self, identifier: str, options: dict | None = None) -> Any | None:
        """Read and decrypt the state from the request cookie.

        Args:
            identifier (str): Key the SDK used when storing the data.
            options (dict | None): Store options; ``options["request"]`` is the Flask request.

        Returns:
            Any | None: The validated model, or None if the cookie is missing or cannot be decrypted.
        """
        try:
            encrypted = options["request"].cookies.get(self.cookie_name)
            return self.model.model_validate(self.decrypt(identifier, encrypted)) if encrypted else None
        except Exception:
            logger.warning("store=cookie status=decrypt_failed cookie=%s", self.cookie_name, exc_info=True)
            return None

    async def delete(self, *_: Any, **__: Any) -> None:
        """Clear the cookie on the response."""

        @after_this_request
        def apply(response: Response) -> Response:
            response.delete_cookie(self.cookie_name)
            return response


def auth0() -> ServerClient:
    """Build the Auth0 SDK client from environment settings.

    Returns:
        ServerClient: Client with cookie-backed session and transaction stores.
    """
    session_secret = env.get("AUTH0_SECRET")
    return ServerClient(
        domain=env.get("AUTH0_DOMAIN"),
        client_id=env.get("AUTH0_CLIENT_ID"),
        client_secret=env.get("AUTH0_CLIENT_SECRET"),
        redirect_uri=env.get("APP_BASE_URL") + "/callback",
        # The audience makes Auth0 issue an access token the FastAPI service accepts.
        authorization_params={"scope": "openid profile email", "audience": AUTH_AUDIENCE},
        secret=session_secret,
        state_store=CookieStore(session_secret, "_a0_session", 259200, StateData),  # 3 days
        transaction_store=CookieStore(session_secret, "_a0_tx", 300, TransactionData),  # 5 min
    )


@app.route("/")
async def home() -> str:
    """Show the signed-in user's profile, or the login and signup links.

    Returns:
        str: HTML page.
    """
    user = await auth0().get_user({"request": request})
    head = "<!DOCTYPE html><title>H&M Assistant login</title>"
    if user:
        return f"""
            {head}
            <p>Logged in as {escape(user.get("email", ""))}</p>
            <h1>User Profile</h1>
            <pre>{escape(json.dumps(user, indent=2))}</pre>
            <a href="/logout">Logout</a>
        """
    return f"""
        {head}
        <a href="/login?screen_hint=signup">Signup</a>
        <a href="/login">Login</a>
    """


@app.route("/login")
async def login() -> Response:
    """Redirect to Auth0 Universal Login.

    Returns:
        Response: Redirect to the Auth0 authorize URL.
    """
    url = await auth0().start_interactive_login(
        options=StartInteractiveLoginOptions(authorization_params=dict(request.args)),
        store_options={"request": request},
    )
    return redirect(url)


@app.route("/callback")
async def callback() -> Response | tuple[str, int]:
    """Finish the login after Auth0 redirects back.

    Returns:
        Response | tuple[str, int]: Redirect home on success, or an error message with status 400.
    """
    try:
        await auth0().complete_interactive_login(url=request.url, store_options={"request": request})
        return redirect("/")
    except Exception:
        logger.exception("route=callback status=failed")
        return "Something went wrong. Check server logs for details.", 400


@app.route("/logout")
async def logout() -> Response:
    """Clear the session and sign out of Auth0.

    Returns:
        Response: Redirect to the Auth0 logout URL.
    """
    url = await auth0().logout(
        options=LogoutOptions(return_to=env.get("APP_BASE_URL")),
        store_options={"request": request},
    )
    return redirect(url)


def main() -> None:
    """Run the login app on the host and port taken from ``APP_BASE_URL``."""
    logging.basicConfig(level=logging.INFO)
    url = urlparse(env.get("APP_BASE_URL"))
    app.run(host=url.hostname, port=url.port or 5000)


if __name__ == "__main__":
    main()
