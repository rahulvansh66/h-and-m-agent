"""Ad-hoc read-only query against the main Neon database (throwaway local script).

Edit ``QUERY`` and run from the repo root: ``uv run python scripts/neon-db-query/get-orders.py``.
Tables are schema-qualified, e.g. ``order_management.orders`` or ``order_management.order_items``.
"""

from pathlib import Path

import psycopg

QUERY = "SELECT * FROM order_management.orders ORDER BY order_date"

# Read the DSN from .env without printing it
env: dict[str, str] = {}
for line in Path(".env").read_text().splitlines():
    if "=" in line and not line.lstrip().startswith("#"):
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")

with psycopg.connect(env["NEON_POSTGRES_CONNECTION_STRING"]) as conn, conn.cursor() as cur:
    cur.execute(QUERY)
    print([d.name for d in cur.description])
    for row in cur.fetchall():
        print(row)
