from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool

from app.core.config import get_settings

_pool: SimpleConnectionPool | None = None


def get_pool() -> SimpleConnectionPool:
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = SimpleConnectionPool(minconn=1, maxconn=10, dsn=settings.database_url)
    return _pool


@contextmanager
def get_connection(cursor_factory=None) -> Iterator:
    pool = get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor(cursor_factory=cursor_factory) as cursor:
            yield conn, cursor
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def init_db() -> None:
    schema_path = Path(__file__).parent / "schema.sql"
    schema_sql = schema_path.read_text(encoding="utf-8")

    with get_connection() as (_, cursor):
        cursor.execute(schema_sql)


def dict_cursor():
    return RealDictCursor
