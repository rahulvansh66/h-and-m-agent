"""Shared pytest fixtures.

Role in the system: test setup for the whole suite. Tests never touch the main Neon database; the
``db`` fixture hands integration tests the test database DSN and refuses to run if it equals the main one.
"""

import os

import pytest
from dotenv import load_dotenv

from hm_agent.core.config import REPO_ROOT

load_dotenv(REPO_ROOT / ".env")


@pytest.fixture(autouse=True)
def offline_langfuse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests off the network: prompts use the in-code fallback and no scores are written.

    Args:
        monkeypatch (pytest.MonkeyPatch): Sets the switches for the duration of one test.
    """
    monkeypatch.setenv("LANGFUSE_PROMPTS_DISABLED", "1")
    monkeypatch.setenv("ONLINE_SCORES_ENABLED", "0")
    monkeypatch.setenv("GUARDRAILS_ENABLED", "0")
    monkeypatch.setenv("ONLINE_RELEVANCY_ENABLED", "0")
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")


@pytest.fixture
def db() -> str:
    """Return the DSN of the test database (``neondb_test``).

    Returns:
        str: Value of ``NEON_POSTGRES_TEST_CONNECTION_STRING``.
    """
    test_dsn = os.environ.get("NEON_POSTGRES_TEST_CONNECTION_STRING")
    if not test_dsn:
        pytest.skip("NEON_POSTGRES_TEST_CONNECTION_STRING is not set")
    if test_dsn == os.environ.get("NEON_POSTGRES_CONNECTION_STRING"):
        pytest.fail("test DSN equals the main DSN; refusing to run against the main database")
    return test_dsn
