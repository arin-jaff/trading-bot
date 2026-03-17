"""Database initialization and session management."""

import os
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool, NullPool
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from .models import Base

DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///data/trading_bot.db')

# SQLite doesn't benefit from connection pooling — NullPool avoids
# QueuePool exhaustion when many scheduler jobs overlap.
engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={'timeout': 30, 'check_same_thread': False},
    poolclass=NullPool,
)
SessionLocal = sessionmaker(bind=engine)


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
