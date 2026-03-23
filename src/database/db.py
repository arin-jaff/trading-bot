"""Database initialization and session management."""

import os
from sqlalchemy import create_engine, event
from sqlalchemy.pool import StaticPool, NullPool
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from .models import Base

DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///data/trading_bot.db')

engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={'timeout': 120, 'check_same_thread': False},
    poolclass=NullPool,
)
SessionLocal = sessionmaker(bind=engine)


@event.listens_for(engine, "connect")
def _set_sqlite_wal(dbapi_conn, connection_record):
    """Enable WAL mode for concurrent read/write access."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


def init_db():
    """Create all tables if they don't exist."""
    os.makedirs('data', exist_ok=True)
    Base.metadata.create_all(engine)


@contextmanager
def get_session() -> Session:
    """Provide a transactional scope around a series of operations."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session_direct() -> Session:
    """Get a session without context manager (for FastAPI dependency injection)."""
    return SessionLocal()
