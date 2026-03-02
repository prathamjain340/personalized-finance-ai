import os
import sqlite3
import threading
from pathlib import Path


_INIT_LOCK = threading.Lock()
_INITIALIZED = False


def get_db_path() -> str:
    return os.getenv("FINANCE_DB_PATH", os.path.join("data", "finance.db"))


def get_connection() -> sqlite3.Connection:
    db_path = get_db_path()
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=5, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return

    with _INIT_LOCK:
        if _INITIALIZED:
            return

        with get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS user_profile_current (
                    user_id TEXT NOT NULL,
                    field_name TEXT NOT NULL,
                    field_value TEXT,
                    value_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, field_name)
                );

                CREATE TABLE IF NOT EXISTS user_profile_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    field_name TEXT NOT NULL,
                    old_value TEXT,
                    new_value TEXT,
                    value_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS user_memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    importance REAL NOT NULL,
                    exposure_count INTEGER NOT NULL DEFAULT 0,
                    last_used_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_user_memories_unique
                ON user_memories(user_id, domain, type, content);

                CREATE INDEX IF NOT EXISTS idx_user_memories_lookup
                ON user_memories(user_id, domain, updated_at DESC);

                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    request TEXT NOT NULL,
                    response TEXT NOT NULL,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_conversation_turns_lookup
                ON conversation_turns(user_id, domain, created_at DESC);
                """
            )

        _INITIALIZED = True

