"""
Database connection and session management
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from config.settings import settings
from src.database.models import Base
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

# Create engine
engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_pre_ping=True
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initialize database (create all tables)"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✓ Database initialized")
        print("✓ Database tables created")
    except Exception as e:
        logger.error(f"Database error: {e}")
        print(f"✗ Error: {e}")
        raise

@contextmanager
def get_db() -> Session:
    """Get database session with automatic cleanup"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"DB error: {e}")
        raise
    finally:
        db.close()

def get_db_session():
    """Get database session for FastAPI dependency injection"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()