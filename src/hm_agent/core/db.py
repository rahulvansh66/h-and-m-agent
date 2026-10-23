"""Neon Postgres helpers: read-only queries and the LangGraph checkpointer.

Role in the system: the only module that opens database connections.
"""

import os
from collections.abc import Generator
from contextlib import contextmanager

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from hm_agent.core.settings import CHECKPOINT_SCHEMA


def fetch_one(sql: str, params: tuple[str, ...]) -> tuple | None:
    """Run a single read-only query on Neon and return the first row.

    Args:
        sql: Parameterised SQL statement (``%s`` placeholders).
        params: Values bound to the placeholders.

    Returns:
        tuple | None: The first row, or None if the query returned no rows.
    """
    with psycopg.connect(os.environ["NEON_POSTGRES_CONNECTION_STRING"]) as conn:
        conn.read_only = True
        return conn.execute(sql, params).fetchone()


def fetch_all(sql: str, params: tuple[object, ...]) -> list[tuple]:
    """Run a read-only query on Neon and return every row.

    Args:
        sql: Parameterised SQL statement (``%s`` placeholders).
        params: Values bound to the placeholders.

    Returns:
        list[tuple]: All rows (empty list if the query returned none).
    """
    with psycopg.connect(os.environ["NEON_POSTGRES_CONNECTION_STRING"]) as conn:
        conn.read_only = True
        return conn.execute(sql, params).fetchall()


@contextmanager
def neon_checkpointer() -> Generator[PostgresSaver]:
    """Open a Postgres checkpointer backed by Neon, with tables in ``checkpoints``.

    The tables live in their own schema (created if missing) so they stay separate
    from the per-agent schemas. ``setup()`` is idempotent, so it is safe on every run.

    Yields:
        PostgresSaver: Checkpointer ready to pass to ``StateGraph.compile``.
    """
    dsn = os.environ["NEON_POSTGRES_CONNECTION_STRING"]
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {CHECKPOINT_SCHEMA}")

    # autocommit + dict_row are required by PostgresSaver; search_path routes its tables.
    with psycopg.connect(
        dsn,
        autocommit=True,
        row_factory=dict_row,
        options=f"-c search_path={CHECKPOINT_SCHEMA}",
    ) as conn:
        saver = PostgresSaver(conn)
        saver.setup()
        yield saver


@contextmanager
def neon_checkpointer_pool(max_size: int) -> Generator[PostgresSaver]:
    """Open a Postgres checkpointer on a small connection pool, for concurrent callers (the API).

    ``neon_checkpointer`` shares one connection, which is fine for the single-user CLI but not for
    several requests at once. Here every checkpoint operation borrows a connection from a pool.

    Args:
        max_size (int): Most connections the pool opens to Neon.

    Yields:
        PostgresSaver: Checkpointer ready to pass to ``StateGraph.compile``.
    """
    dsn = os.environ["NEON_POSTGRES_CONNECTION_STRING"]
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {CHECKPOINT_SCHEMA}")

    with ConnectionPool(
        dsn,
        min_size=1,
        max_size=max_size,
        kwargs={"autocommit": True, "row_factory": dict_row, "options": f"-c search_path={CHECKPOINT_SCHEMA}"},
        open=True,
    ) as pool:
        saver = PostgresSaver(pool)
        saver.setup()
        yield saver
