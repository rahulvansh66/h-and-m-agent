"""Create the three agent schemas in Neon Postgres and seed them from the H&M mini dataset.

Role in the system: one-off setup script (not part of the running agents). It owns the
DDL and loads ``data/H_and_M_mini_dataset/csv-format`` (articles, customers, transactions)
into the schema each agent's tools query:

  order_management        customers, orders, order_items
  product_recommendation  product_catalog, purchase_history
  personalization         personalization

Mapping rules (the dataset has no order ids, order status or stock, so these are derived):
  * Readable ids: customers ``cust001``.. (most purchases first, then file order) and
    products ``prod001``.. (articles.csv order). The original H&M ids are kept in
    ``hm_customer_id`` / ``article_id``.
  * One order per (customer, purchase date): same-day purchases form one basket; order ids
    ``ORD-1001``.. follow date then customer. Every transaction becomes one order item.
  * Every order is ``delivered`` / shipping ``delivered`` with no return: all transactions
    are historical, and the dataset records no delivery date, so ``delivery_date`` is NULL.
  * ``price`` is the dataset's normalized price (not a currency amount) kept at full precision.
  * Nothing else is invented: no names, contact details, stock levels or ratings.

Foreign keys exist only inside a schema (each agent stays independent); cross-schema
consistency is verified by check_consistency() instead of enforced by the database.

Idempotent: safe to re-run; a table is seeded only when empty.
  python scripts/neon-db-creation/init_neon.py            create + seed what is missing
  python scripts/neon-db-creation/init_neon.py --reset    drop the three schemas first
  python scripts/neon-db-creation/init_neon.py --test     same, but on the separate test database

The test database (``neondb_test``, same Neon endpoint as the main one) lets the
integration tests write freely without touching the real dataset. ``--test`` creates it
on first use and reads its DSN from NEON_POSTGRES_TEST_CONNECTION_STRING; if that
variable is missing it is derived from the main DSN and appended to ``.env``.

Reads NEON_POSTGRES_CONNECTION_STRING from the environment or the repo-root .env file.
"""

import argparse
import csv
import logging
import os
from collections import Counter
from collections.abc import Callable
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import psycopg
import psycopg.sql

logger = logging.getLogger(__name__)

ENV_VAR = "NEON_POSTGRES_CONNECTION_STRING"
TEST_ENV_VAR = "NEON_POSTGRES_TEST_CONNECTION_STRING"
TEST_DB_NAME = "neondb_test"
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
SCHEMAS = ("order_management", "product_recommendation", "personalization")


def load_connection_string(env_var: str = ENV_VAR, required: bool = True) -> str | None:
    """Resolve a connection string from the environment or the repo-root ``.env``.

    The process environment wins; the ``.env`` file is only a fallback so the script
    also works when run outside a configured shell.

    Args:
        env_var: Name of the variable to read (main or test DSN).
        required: If True, a missing value aborts the script; if False, return None.

    Returns:
        str | None: The variable's value, or None when missing and not required.

    Raises:
        SystemExit: If ``required`` and the variable is set in neither place.
    """
    value = os.environ.get(env_var)
    if not value and ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            key, _, val = line.partition("=")
            if key.strip() == env_var:
                value = val.strip().strip("\"'")
    if not value and required:
        raise SystemExit(f"{env_var} is not set (environment or .env)")
    return value or None


def with_database(dsn: str, database: str) -> str:
    """Return ``dsn`` pointing at a different database name on the same server.

    Args:
        dsn: A ``postgresql://user:pass@host/dbname?params`` connection string.
        database: The database name to substitute.

    Returns:
        str: The DSN with only the database path segment replaced.
    """
    parts = urlsplit(dsn)
    return urlunsplit(parts._replace(path=f"/{database}"))


def ensure_test_database() -> str:
    """Create the test database if missing and return its DSN (persisted to ``.env``).

    Connects to the main database in autocommit mode (``CREATE DATABASE`` cannot run
    inside a transaction). The main database's data is never modified.

    Returns:
        str: The test database connection string.

    Raises:
        SystemExit: If the main DSN is missing, or the configured test DSN points at
            the same database as the main one (would defeat the isolation).
    """
    main_dsn = load_connection_string()
    test_dsn = load_connection_string(TEST_ENV_VAR, required=False) or with_database(
        main_dsn, TEST_DB_NAME
    )
    if urlsplit(test_dsn).path == urlsplit(main_dsn).path and urlsplit(
        test_dsn
    ).hostname == urlsplit(main_dsn).hostname:
        raise SystemExit(f"{TEST_ENV_VAR} must point at a different database than {ENV_VAR}")

    # Create the database only when absent, so re-runs are harmless
    name = urlsplit(test_dsn).path.lstrip("/")
    with psycopg.connect(main_dsn, autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone()
        if not exists:
            conn.execute(psycopg.sql.SQL("CREATE DATABASE {}").format(psycopg.sql.Identifier(name)))
            logger.info("step=create_test_db status=created database=%s", name)

    # Persist the DSN so tests can find it (never printed: it holds credentials)
    if load_connection_string(TEST_ENV_VAR, required=False) is None:
        with ENV_FILE.open("a", encoding="utf-8") as fh:
            fh.write(f"\n# Separate database for tests (created by init_neon.py --test)\n{TEST_ENV_VAR}={test_dsn}\n")
        logger.info("step=persist_test_dsn status=complete env_var=%s", TEST_ENV_VAR)
    return test_dsn


SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS order_management;
CREATE SCHEMA IF NOT EXISTS product_recommendation;
CREATE SCHEMA IF NOT EXISTS personalization;

CREATE TABLE IF NOT EXISTS order_management.customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    hm_customer_id VARCHAR(64) NOT NULL UNIQUE,
    club_member_status VARCHAR(30),
    fashion_news_frequency VARCHAR(30),
    fashion_news_subscriber BOOLEAN NOT NULL DEFAULT FALSE,
    active BOOLEAN NOT NULL DEFAULT FALSE,
    age INTEGER CHECK (age IS NULL OR age BETWEEN 0 AND 120),
    postal_code VARCHAR(64),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS order_management.orders (
    order_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL REFERENCES order_management.customers(customer_id),
    order_status VARCHAR(50) NOT NULL
        CHECK (order_status IN ('processing', 'shipped', 'delivered', 'cancelled')),
    shipping_status VARCHAR(50)
        CHECK (shipping_status IN ('preparing', 'in_transit', 'delivered', 'cancelled')),
    return_exchange_status VARCHAR(50)
        CHECK (return_exchange_status IN ('return_requested', 'exchange_completed')),
    sales_channel VARCHAR(20) NOT NULL CHECK (sales_channel IN ('store', 'online')),
    order_date DATE NOT NULL,
    delivery_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT orders_cancelled_has_no_delivery CHECK (order_status <> 'cancelled' OR delivery_date IS NULL),
    CONSTRAINT orders_return_only_when_delivered CHECK (return_exchange_status IS NULL OR order_status = 'delivered')
);

CREATE TABLE IF NOT EXISTS order_management.order_items (
    order_id VARCHAR(50) NOT NULL REFERENCES order_management.orders(order_id),
    line_no INTEGER NOT NULL CHECK (line_no > 0),
    product_id VARCHAR(50) NOT NULL,
    product_name VARCHAR(255) NOT NULL,
    quantity INTEGER NOT NULL DEFAULT 1 CHECK (quantity > 0),
    price NUMERIC(12,9) NOT NULL CHECK (price >= 0),
    PRIMARY KEY (order_id, line_no)
);

CREATE INDEX IF NOT EXISTS idx_orders_customer_id ON order_management.orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_orders_status ON order_management.orders(order_status);
CREATE INDEX IF NOT EXISTS idx_orders_date ON order_management.orders(order_date);

CREATE TABLE IF NOT EXISTS product_recommendation.product_catalog (
    product_id TEXT PRIMARY KEY,
    article_id TEXT NOT NULL UNIQUE,
    product_code TEXT NOT NULL,
    product_name TEXT NOT NULL,
    product_type TEXT NOT NULL,
    product_group TEXT NOT NULL,
    colour TEXT,
    perceived_colour TEXT,
    department TEXT,
    section TEXT,
    garment_group TEXT,
    index_group TEXT,
    description TEXT
);

CREATE TABLE IF NOT EXISTS product_recommendation.purchase_history (
    id SERIAL PRIMARY KEY,
    customer_id TEXT NOT NULL,
    product_id TEXT NOT NULL REFERENCES product_recommendation.product_catalog(product_id),
    purchase_date DATE NOT NULL,
    price NUMERIC(12,9) NOT NULL CHECK (price >= 0),
    sales_channel TEXT NOT NULL CHECK (sales_channel IN ('store', 'online'))
);

CREATE INDEX IF NOT EXISTS idx_purchase_history_customer ON product_recommendation.purchase_history(customer_id);

CREATE TABLE IF NOT EXISTS personalization.personalization (
    customer_id TEXT PRIMARY KEY,
    age INTEGER,
    club_member_status TEXT,
    fashion_news_frequency TEXT,
    fashion_news_subscriber BOOLEAN NOT NULL DEFAULT FALSE,
    active BOOLEAN NOT NULL DEFAULT FALSE,
    top_product_group TEXT,
    purchase_count INTEGER NOT NULL DEFAULT 0
);
"""

# ---------------------------------------------------------------------------
# Source data and mapping
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "H_and_M_mini_dataset" / "csv-format"
SALES_CHANNELS = {"1": "store", "2": "online"}

# A seed row is a tuple of column values in INSERT order.
Row = tuple[Any, ...]


class HMData:
    """The mini dataset plus the readable ids and orders derived from it.

    Attributes:
        customers: Customer rows (dicts keyed by CSV column) ordered by assigned id.
        customer_ids: Original H&M customer hash -> ``custNNN``.
        articles: Article rows in file order.
        product_ids: Original ``article_id`` -> ``prodNNN``.
        transactions: Transaction rows in file order.
        orders: Derived orders: ``(order_id, customer_id, order_date, channel, items)`` where
            ``items`` is a list of ``(article_id, price)``, sorted by date then customer.
    """

    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        """Load the three CSV files and derive ids and orders.

        Args:
            data_dir: Folder holding ``articles.csv``, ``customers.csv``, ``transactions.csv``.

        Raises:
            SystemExit: If a CSV file is missing.
        """
        files = {name: data_dir / f"{name}.csv" for name in ("articles", "customers", "transactions")}
        missing = [str(path) for path in files.values() if not path.is_file()]
        if missing:
            raise SystemExit(f"H&M dataset file(s) not found: {', '.join(missing)}")

        def read(path: Path) -> list[dict[str, str]]:
            """Read a CSV file as dict rows (all values stay strings, keeping leading zeros).

            Args:
                path: CSV file to read.

            Returns:
                One dict per data row.
            """
            with path.open(encoding="utf-8", newline="") as fh:
                return list(csv.DictReader(fh))

        self.articles = read(files["articles"])
        self.transactions = read(files["transactions"])
        raw_customers = read(files["customers"])

        # Most purchases first (stable on file order), so cust001 has a real order history
        counts = Counter(t["customer_id"] for t in self.transactions)
        ranked = sorted(
            enumerate(raw_customers), key=lambda pair: (-counts[pair[1]["customer_id"]], pair[0])
        )
        self.customers = [row for _, row in ranked]
        self.customer_ids = {
            c["customer_id"]: f"cust{i:03d}" for i, c in enumerate(self.customers, 1)
        }
        self.product_ids = {a["article_id"]: f"prod{i:03d}" for i, a in enumerate(self.articles, 1)}

        # One order per (customer, purchase date): same-day purchases are one basket
        baskets: dict[tuple[str, str, str], list[tuple[str, Decimal]]] = {}
        for t in self.transactions:
            channel = SALES_CHANNELS.get(t["sales_channel_id"], "online")
            key = (t["t_dat"], self.customer_ids[t["customer_id"]], channel)
            baskets.setdefault(key, []).append((t["article_id"], Decimal(t["price"])))
        self.orders = [
            (f"ORD-{1000 + n}", cust, date.fromisoformat(day), channel, items)
            for n, ((day, cust, channel), items) in enumerate(sorted(baskets.items()), 1)
        ]


def flag(value: str) -> bool:
    """Convert the dataset's ``1.0`` / empty flags to a boolean (empty means "no").

    Args:
        value: Raw CSV cell.

    Returns:
        True only for a non-empty, non-zero value.
    """
    return value.strip() not in ("", "0", "0.0")


def optional(value: str) -> str | None:
    """Turn an empty CSV cell into None.

    Args:
        value: Raw CSV cell.

    Returns:
        The stripped text, or None if empty.
    """
    return value.strip() or None


def age_of(row: dict[str, str]) -> int | None:
    """Parse the customer's age (the CSV stores it as ``53.0``).

    Args:
        row: A customers.csv row.

    Returns:
        The age as an int, or None if blank.
    """
    return int(float(row["age"])) if row["age"].strip() else None


def build_customers(data: HMData) -> list[Row]:
    """Build rows for ``order_management.customers``.

    Args:
        data: Loaded dataset.

    Returns:
        One row per H&M customer.
    """
    return [
        (data.customer_ids[c["customer_id"]], c["customer_id"],
         optional(c["club_member_status"]), optional(c["fashion_news_frequency"]),
         flag(c["FN"]), flag(c["Active"]), age_of(c), optional(c["postal_code"]))
        for c in data.customers
    ]


def build_products(data: HMData) -> list[Row]:
    """Build rows for ``product_recommendation.product_catalog``.

    Args:
        data: Loaded dataset.

    Returns:
        One row per article; ``description`` is NULL when the dataset has none.
    """
    return [
        (data.product_ids[a["article_id"]], a["article_id"], a["product_code"], a["prod_name"],
         a["product_type_name"], a["product_group_name"], optional(a["colour_group_name"]),
         optional(a["perceived_colour_master_name"]), optional(a["department_name"]),
         optional(a["section_name"]), optional(a["garment_group_name"]),
         optional(a["index_group_name"]), optional(a["detail_desc"]))
        for a in data.articles
    ]


def build_orders(data: HMData) -> list[Row]:
    """Build rows for ``order_management.orders`` (all delivered; see module docstring).

    Args:
        data: Loaded dataset.

    Returns:
        One row per derived order.
    """
    return [
        (order_id, cust, "delivered", "delivered", None, channel, day, None)
        for order_id, cust, day, channel, _items in data.orders
    ]


def build_order_items(data: HMData) -> list[Row]:
    """Build rows for ``order_management.order_items``.

    Args:
        data: Loaded dataset.

    Returns:
        One row per transaction, numbered per order.
    """
    names = {a["article_id"]: a["prod_name"] for a in data.articles}
    return [
        (order_id, line, data.product_ids[article], names[article], 1, price)
        for order_id, _cust, _day, _channel, items in data.orders
        for line, (article, price) in enumerate(items, 1)
    ]


def build_purchase_history(data: HMData) -> list[Row]:
    """Build rows for ``product_recommendation.purchase_history`` (one per transaction).

    Args:
        data: Loaded dataset.

    Returns:
        Rows in transaction file order.
    """
    return [
        (data.customer_ids[t["customer_id"]], data.product_ids[t["article_id"]],
         date.fromisoformat(t["t_dat"]), Decimal(t["price"]),
         SALES_CHANNELS.get(t["sales_channel_id"], "online"))
        for t in data.transactions
    ]


def build_personalization(data: HMData) -> list[Row]:
    """Build rows for ``personalization.personalization``.

    ``top_product_group`` and ``purchase_count`` are derived from the customer's own
    transactions (most frequent group, ties broken alphabetically).

    Args:
        data: Loaded dataset.

    Returns:
        One row per customer.
    """
    group_of = {a["article_id"]: a["product_group_name"] for a in data.articles}
    rows: list[Row] = []
    for c in data.customers:
        groups = Counter(
            group_of[t["article_id"]]
            for t in data.transactions
            if t["customer_id"] == c["customer_id"]
        )
        top = min(groups, key=lambda g: (-groups[g], g)) if groups else None
        rows.append((data.customer_ids[c["customer_id"]], age_of(c),
                     optional(c["club_member_status"]), optional(c["fashion_news_frequency"]),
                     flag(c["FN"]), flag(c["Active"]), top, sum(groups.values())))
    return rows


# (table, INSERT statement, row builder). Order matters: parents before children.
SEEDS: list[tuple[str, str, Callable[[HMData], list[Row]]]] = [
    ("order_management.customers",
     "INSERT INTO order_management.customers (customer_id, hm_customer_id, club_member_status, fashion_news_frequency, fashion_news_subscriber, active, age, postal_code) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
     build_customers),
    ("product_recommendation.product_catalog",
     "INSERT INTO product_recommendation.product_catalog (product_id, article_id, product_code, product_name, product_type, product_group, colour, perceived_colour, department, section, garment_group, index_group, description) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
     build_products),
    ("order_management.orders",
     "INSERT INTO order_management.orders (order_id, customer_id, order_status, shipping_status, return_exchange_status, sales_channel, order_date, delivery_date) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
     build_orders),
    ("order_management.order_items",
     "INSERT INTO order_management.order_items (order_id, line_no, product_id, product_name, quantity, price) VALUES (%s,%s,%s,%s,%s,%s)",
     build_order_items),
    ("product_recommendation.purchase_history",
     "INSERT INTO product_recommendation.purchase_history (customer_id, product_id, purchase_date, price, sales_channel) VALUES (%s,%s,%s,%s,%s)",
     build_purchase_history),
    ("personalization.personalization",
     "INSERT INTO personalization.personalization (customer_id, age, club_member_status, fashion_news_frequency, fashion_news_subscriber, active, top_product_group, purchase_count) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
     build_personalization),
]

# Each query returns offending rows; every one must return nothing.
CONSISTENCY_CHECKS: dict[str, str] = {
    "order item product not in catalog": """
        SELECT i.order_id FROM order_management.order_items i
        LEFT JOIN product_recommendation.product_catalog c ON c.product_id = i.product_id
        WHERE c.product_id IS NULL""",
    "order item name differs from catalog": """
        SELECT i.order_id FROM order_management.order_items i
        JOIN product_recommendation.product_catalog c ON c.product_id = i.product_id
        WHERE i.product_name <> c.product_name""",
    "purchase_history customer not in customers": """
        SELECT h.id FROM product_recommendation.purchase_history h
        LEFT JOIN order_management.customers c ON c.customer_id = h.customer_id
        WHERE c.customer_id IS NULL""",
    "personalization customer not in customers": """
        SELECT p.customer_id FROM personalization.personalization p
        LEFT JOIN order_management.customers c ON c.customer_id = p.customer_id
        WHERE c.customer_id IS NULL""",
    "purchase_history row without a matching order item": """
        SELECT h.id FROM product_recommendation.purchase_history h
        LEFT JOIN (
            SELECT o.customer_id, i.product_id, o.order_date, i.price
            FROM order_management.order_items i
            JOIN order_management.orders o ON o.order_id = i.order_id
        ) m ON m.customer_id = h.customer_id AND m.product_id = h.product_id
           AND m.order_date = h.purchase_date AND m.price = h.price
        WHERE m.customer_id IS NULL""",
    "order item without a purchase_history row": """
        SELECT i.order_id FROM order_management.order_items i
        JOIN order_management.orders o ON o.order_id = i.order_id
        LEFT JOIN product_recommendation.purchase_history h
          ON h.customer_id = o.customer_id AND h.product_id = i.product_id
         AND h.purchase_date = o.order_date AND h.price = i.price
        WHERE h.id IS NULL""",
    "personalization purchase_count differs from purchase_history": """
        SELECT p.customer_id FROM personalization.personalization p
        WHERE p.purchase_count <> (
            SELECT COUNT(*) FROM product_recommendation.purchase_history h
            WHERE h.customer_id = p.customer_id)""",
}


def check_consistency(conn: psycopg.Connection) -> None:
    """Verify the cross-schema rules the database cannot enforce (no cross-schema FKs).

    Runs every query in ``CONSISTENCY_CHECKS``; each must return zero rows.

    Args:
        conn (psycopg.Connection): Open connection (same transaction as the seed).

    Raises:
        SystemExit: If any check returns offending rows; the message lists them.
    """
    failures: list[str] = []
    for name, sql in CONSISTENCY_CHECKS.items():
        bad = conn.execute(sql).fetchall()
        if bad:
            failures.append(f"{name}: {[r[0] for r in bad]}")
    if failures:
        raise SystemExit("Consistency check failed:\n  " + "\n  ".join(failures))
    logger.info("step=consistency_check status=passed checks=%d", len(CONSISTENCY_CHECKS))


def main() -> None:
    """Create schemas/tables, seed empty tables, run consistency checks, and commit.

    Entry point: configures logging, parses ``--reset`` and ``--test``, and runs everything in one
    transaction so a failed check leaves the database untouched.

    Raises:
        SystemExit: If the connection string or dataset is missing, or a consistency check fails.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reset", action="store_true", help="drop the three agent schemas before creating them")
    parser.add_argument("--test", action="store_true", help=f"target the separate {TEST_DB_NAME} database instead of the main one")
    args = parser.parse_args()

    data = HMData()
    dsn = ensure_test_database() if args.test else load_connection_string()
    with psycopg.connect(dsn) as conn:
        # Optional wipe of the agent schemas (destructive; they hold only seeded data)
        if args.reset:
            for schema in SCHEMAS:
                conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            logger.info("step=reset status=complete schemas=%s", ",".join(SCHEMAS))
        conn.execute(SCHEMA_SQL)

        # Seed only empty tables so re-runs never duplicate rows
        for table, insert_sql, build_rows in SEEDS:
            (count,) = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            if count:
                logger.info("table=%s status=skipped rows=%d", table, count)
                continue
            rows = build_rows(data)
            with conn.cursor() as cur:
                cur.executemany(insert_sql, rows)
            logger.info("table=%s status=seeded rows=%d", table, len(rows))

        check_consistency(conn)
        conn.commit()

        # Final row counts as a quick sanity summary
        total = 0
        for table, _, _ in SEEDS:
            (n,) = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            total += n
            logger.info("table=%s rows=%d", table, n)
        logger.info("total_rows=%d", total)


if __name__ == "__main__":
    main()
