"""
database/connection.py — PostgreSQL connection helper and data loading.

Provides connection pooling, parameterized queries, and
functions to load/retrieve data from the techsight database.

MOCK MODE: When database is unavailable, returns mock data.
"""

import logging
import json
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Try to import psycopg2; if unavailable, mark as mock mode
try:
    import psycopg2
    import psycopg2.extras
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False
    logger.warning("psycopg2 not installed — running in mock mode")

from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD


def get_connection():
    """
    Get a new database connection.

    Returns:
        psycopg2 connection object, or None if unavailable
    """
    if not HAS_PSYCOPG2:
        return None

    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
        )
        return conn
    except Exception as e:
        logger.error("Database connection failed: %s", e)
        return None


@contextmanager
def db_cursor():
    """
    Context manager for database operations.

    Usage:
        with db_cursor() as cur:
            cur.execute("SELECT * FROM documents")
            rows = cur.fetchall()
    """
    conn = get_connection()
    if conn is None:
        yield None
        return

    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        yield cur
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error("Database error: %s", e)
        raise
    finally:
        conn.close()


def save_documents(df):
    """
    Insert documents into the database using parameterized queries.

    Uses ON CONFLICT to skip duplicates.
    """
    with db_cursor() as cur:
        if cur is None:
            logger.warning("No database — skipping save_documents")
            return 0

        sql = """
            INSERT INTO documents
                (source, source_id, title, abstract, year, classifications,
                 authors, organizations, citation_count, url, domain)
            VALUES
                (%(source)s, %(source_id)s, %(title)s, %(abstract)s,
                 %(year)s, %(classifications)s, %(authors)s,
                 %(organizations)s, %(citation_count)s, %(url)s, %(domain)s)
            ON CONFLICT (source, source_id) DO NOTHING
        """
        records = df.to_dict("records")
        count = 0
        for record in records:
            cur.execute(sql, record)
            count += cur.rowcount

        logger.info("Saved %d new documents to database", count)
        return count


def save_technologies(df):
    """Insert technology topics into the database."""
    with db_cursor() as cur:
        if cur is None:
            logger.warning("No database — skipping save_technologies")
            return 0

        sql = """
            INSERT INTO technologies (name, cluster_id, top_terms, document_count, description)
            VALUES (%(name)s, %(cluster_id)s, %(top_terms)s, %(document_count)s, %(description)s)
            RETURNING id
        """
        ids = []
        for record in df.to_dict("records"):
            cur.execute(sql, record)
            ids.append(cur.fetchone()["id"])

        return ids


def save_trend_scores(df):
    """Insert trend scores into the database."""
    with db_cursor() as cur:
        if cur is None:
            return 0

        sql = """
            INSERT INTO trend_scores
                (technology_id, score, label, growth_rate, consistency,
                 acceleration, activity_volume, citation_signal,
                 lookback_years, config_snapshot)
            VALUES
                (%(technology_id)s, %(score)s, %(label)s, %(growth_rate)s,
                 %(consistency)s, %(acceleration)s, %(activity_volume)s,
                 %(citation_signal)s, %(lookback_years)s, %(config_snapshot)s)
            ON CONFLICT (technology_id) DO UPDATE
            SET score = EXCLUDED.score, label = EXCLUDED.label,
                computed_at = NOW()
        """
        for record in df.to_dict("records"):
            cur.execute(sql, record)
        return len(df)
