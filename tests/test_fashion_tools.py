"""Tests for the fashion MCP tools (``hm_agent.fashion.mcp_server``).

Role in the system: checks segment resolution, the swimwear/underwear exclusion, the relevance gate and
hydration from the database. Qdrant, the embedder and (for the pure-logic cases) the database are faked, so
no LLM, network or cloud call is made. One test reads the seeded test database through the ``db`` fixture.
"""

from types import SimpleNamespace

import pytest

from hm_agent.fashion import mcp_server


class FakeEmbedder:
    """Stands in for the fastembed model: returns a fixed vector for any query."""

    def query_embed(self, query: str):
        return iter([SimpleNamespace(tolist=lambda: [0.1, 0.2])])


class FakeQdrant:
    """Records the filter it was queried with and returns canned points."""

    def __init__(self, points: list) -> None:
        self.points = points
        self.query_filter = None

    def query_points(self, **kwargs):
        self.query_filter = kwargs["query_filter"]
        return SimpleNamespace(points=self.points)


def _point(product_id: str, score: float) -> SimpleNamespace:
    return SimpleNamespace(payload={"product_id": product_id}, score=score)


@pytest.fixture
def fake_search(monkeypatch):
    """Patch the embedder, Qdrant and catalog lookup; return a function that sets the canned points."""

    def install(points: list) -> FakeQdrant:
        qdrant = FakeQdrant(points)
        monkeypatch.setattr(mcp_server, "get_embedder", lambda model_name: FakeEmbedder())
        monkeypatch.setattr(mcp_server, "get_client", lambda: qdrant)
        monkeypatch.setattr(
            mcp_server,
            "fetch_all",
            lambda sql, params: [("prod008", "Robin blazer", "Blazer", "Garment Upper body", "Black", "Suit", None)],
        )
        return qdrant

    return install


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ([("Ladieswear", 3), ("Menswear", 1)], "women"),
        ([("Menswear", 2)], "men"),
        ([("Ladieswear", 2), ("Menswear", 2)], None),  # tie: no segment
        ([], None),  # new customer: no signal
    ],
)
def test_resolve_segment(monkeypatch, counts, expected) -> None:
    """The larger of Ladieswear/Menswear wins; a tie or no purchases gives no segment."""
    monkeypatch.setattr(mcp_server, "fetch_all", lambda sql, params: counts)
    assert mcp_server._resolve_segment("cust001") == expected


def test_styling_guide_is_neutral_without_segment(monkeypatch) -> None:
    """No segment means no guide is loaded and the agent is told to stay neutral."""
    monkeypatch.setattr(mcp_server, "fetch_all", lambda sql, params: [])
    assert "neutral" in mcp_server.get_styling_guide("cust001")


def test_search_always_excludes_swimwear_and_underwear(fake_search) -> None:
    """The exclusion is applied in code, merged with the groups the model asked to exclude."""
    qdrant = fake_search([_point("prod008", 0.8)])
    result = mcp_server.search_products("cust001", "smart dinner", exclude_product_groups=["Garment Lower body"])
    excluded = qdrant.query_filter.must_not[0].match.any
    assert {"Swimwear", "Underwear", "Garment Lower body"} <= set(excluded)
    assert "Robin blazer" in result


def test_search_can_allow_swimwear_when_asked(fake_search) -> None:
    """With the explicit flag, swimwear and underwear are no longer excluded."""
    qdrant = fake_search([_point("prod008", 0.8)])
    mcp_server.search_products("cust001", "bikini", allow_swimwear_underwear=True)
    assert qdrant.query_filter.must_not == []


def test_search_below_min_score_reports_no_match(fake_search) -> None:
    """Hits under the relevance threshold are dropped instead of padded into the answer."""
    fake_search([_point("prod008", 0.01)])
    assert mcp_server.search_products("cust001", "spacesuit") == "No good match found in the catalog."


def test_segment_on_seeded_test_database(db, monkeypatch) -> None:
    """Against the seeded data (Ladieswear/Divided only) a customer resolves to women or no segment."""
    monkeypatch.setenv("NEON_POSTGRES_CONNECTION_STRING", db)
    assert mcp_server._resolve_segment("cust001") in ("women", None)
