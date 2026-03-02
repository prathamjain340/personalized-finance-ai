import datetime
import json
from typing import Any

from finance_project.core.storage.sqlite_db import get_connection, init_db


def save_conversation_turn(
    user_id: str,
    domain: str,
    request: str,
    response: str,
    metadata: dict[str, Any] | None = None,
    created_at: str | None = None,
) -> None:
    """
    Persist an immutable conversation turn for audit/debug/history usage.
    """
    init_db()
    timestamp = created_at or datetime.datetime.utcnow().isoformat()
    metadata_json = json.dumps(metadata or {}, ensure_ascii=True)

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO conversation_turns
                (user_id, domain, request, response, metadata_json, created_at)
            VALUES
                (?, ?, ?, ?, ?, ?)
            """,
            (user_id, domain, request, response, metadata_json, timestamp),
        )
        conn.commit()

