"""
Database Connection Manager
===========================
Manages database connections and initialization.
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """
    Database connection manager.

    Supports SQLite (default) and can be extended for PostgreSQL.
    """

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.getenv(
            "DATABASE_URL", "sqlite:///./data/app.db"
        )
        self._engine = None
        self._session_factory = None

    def initialize(self):
        """Initialize database connection and create tables."""
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker

            # Ensure directory exists for SQLite
            if "sqlite" in self.database_url:
                db_path = self.database_url.replace("sqlite:///", "")
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)

            self._engine = create_engine(
                self.database_url,
                echo=False,
                pool_pre_ping=True,
            )

            self._session_factory = sessionmaker(bind=self._engine)

            logger.info(f"Database initialized: {self.database_url}")

        except ImportError:
            logger.warning("SQLAlchemy not available. Using in-memory storage.")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")

    def get_session(self):
        """Get a new database session."""
        if self._session_factory:
            return self._session_factory()
        return None

    def close(self):
        """Close database connections."""
        if self._engine:
            self._engine.dispose()
            logger.info("Database connections closed")

    @property
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._engine is not None
