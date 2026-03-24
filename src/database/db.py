"""Database initialization and session management."""

import os
import time
from sqlalchemy import create_engine, event
from sqlalchemy.pool import StaticPool, NullPool
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from loguru import logger
from .models import Base

DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///data/trading_bot.db')

engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={'timeout': 120, 'check_same_thread': False},
    poolclass=NullPool,
)
SessionLocal = sessionmaker(bind=engine)

# Max retries and backoff for commit-time "database is locked" errors
_COMMIT_MAX_RETRIES = 3
_COMMIT_BACKOFF_BASE = 2  # seconds


@event.listens_for(engine, "connect")
def _set_sqlite_wal(dbapi_conn, connection_record):
    """Enable WAL mode and tune SQLite for concurrent access."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=30000")  # 30s wait before raising locked
    cursor.execute("PRAGMA wal_autocheckpoint=1000")  # checkpoint every 1000 pages
    cursor.close()


def init_db():
    """Create all tables if they don't exist."""
    os.makedirs('data', exist_ok=True)
    Base.metadata.create_all(engine)


@contextmanager
def get_session() -> Session:
    """Provide a transactional scope around a series of operations.

    Retries the commit up to 3 times on 'database is locked' errors
    with exponential backoff (2s, 4s, 8s).
    """
    session = SessionLocal()
    try:
        yield session
        # Retry commit on "database is locked"
        for attempt in range(_COMMIT_MAX_RETRIES):
            try:
                session.commit()
                break
            except Exception as e:
                if 'locked' in str(e).lower() and attempt < _COMMIT_MAX_RETRIES - 1:
                    wait = _COMMIT_BACKOFF_BASE * (2 ** attempt)
                    logger.warning(f"DB locked on commit, retry {attempt+1}/{_COMMIT_MAX_RETRIES} in {wait}s...")
                    session.rollback()
                    time.sleep(wait)
                else:
                    raise
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session_direct() -> Session:
    """Get a session without context manager (for FastAPI dependency injection)."""
    return SessionLocal()
