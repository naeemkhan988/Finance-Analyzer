# API key manager — SQLite-backed API key authentication and management.
"""
API Key Manager
===============
SQLite-backed API key management with tiered access levels.
"""

import os
import hashlib
import secrets
import sqlite3
import logging
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class APIKeyManager:
    """
    API key management with SHA256 hashing.

    Features:
    - Tiered access (free, pro, admin)
    - SHA256 key hashing (never stores plaintext)
    - Usage tracking
    - Key generation and revocation
    - Auto-generates admin key on first run if not set
    """

    def __init__(self, db_path: str = "./data/api_keys.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._ensure_default_key()

    def _init_db(self):
        """Create api_keys table."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_hash TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    tier TEXT DEFAULT 'free',
                    requests_today INTEGER DEFAULT 0,
                    total_requests INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    last_used TEXT,
                    active INTEGER DEFAULT 1
                )
            """)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _hash_key(self, api_key: str) -> str:
        """Hash an API key with SHA256."""
        return hashlib.sha256(api_key.encode()).hexdigest()

    def _ensure_default_key(self):
        """Ensure an admin key exists. Generate one if ADMIN_API_KEY not set."""
        admin_key = os.getenv("ADMIN_API_KEY", "")

        if admin_key:
            key_hash = self._hash_key(admin_key)
            with self._connect() as conn:
                existing = conn.execute(
                    "SELECT id FROM api_keys WHERE key_hash = ?", (key_hash,)
                ).fetchone()

                if not existing:
                    conn.execute(
                        """
                        INSERT INTO api_keys (key_hash, name, tier, created_at)
                        VALUES (?, 'Admin (env)', 'admin', ?)
                        """,
                        (key_hash, datetime.utcnow().isoformat()),
                    )
                    conn.commit()
            logger.info("Admin API key registered from environment")
        else:
            # Check if any admin key exists
            with self._connect() as conn:
                existing = conn.execute(
                    "SELECT id FROM api_keys WHERE tier = 'admin' AND active = 1"
                ).fetchone()

            if not existing:
                new_key = self.generate_key("Auto-Admin", "admin")
                print("\n" + "=" * 60)
                print("  AUTO-GENERATED ADMIN API KEY")
                print("  Store this securely — it will not be shown again!")
                print(f"  Key: {new_key}")
                print("  Set ADMIN_API_KEY in .env to use a fixed key.")
                print("=" * 60 + "\n")

    def validate(self, api_key: str) -> Dict:
        """
        Validate an API key.

        Returns:
            Dict with 'valid' (bool), 'tier' (str), 'name' (str)
        """
        if not api_key:
            return {"valid": False, "tier": "none", "name": ""}

        key_hash = self._hash_key(api_key)

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, name, tier, active FROM api_keys
                WHERE key_hash = ?
                """,
                (key_hash,),
            ).fetchone()

        if not row or not row["active"]:
            return {"valid": False, "tier": "none", "name": ""}

        # Update usage stats
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE api_keys
                SET requests_today = requests_today + 1,
                    total_requests = total_requests + 1,
                    last_used = ?
                WHERE key_hash = ?
                """,
                (datetime.utcnow().isoformat(), key_hash),
            )
            conn.commit()

        return {
            "valid": True,
            "tier": row["tier"],
            "name": row["name"],
        }

    def generate_key(self, name: str, tier: str = "free") -> str:
        """Generate a new API key."""
        raw_key = f"fra_{secrets.token_urlsafe(32)}"
        key_hash = self._hash_key(raw_key)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO api_keys (key_hash, name, tier, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (key_hash, name, tier, datetime.utcnow().isoformat()),
            )
            conn.commit()

        logger.info(f"Generated new {tier} API key: '{name}'")
        return raw_key

    def revoke_key(self, key_hash: str):
        """Deactivate an API key by its hash."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE api_keys SET active = 0 WHERE key_hash = ?",
                (key_hash,),
            )
            conn.commit()
        logger.info(f"Revoked API key: {key_hash[:16]}...")
