import re
from datetime import datetime
from typing import List

from finance_project.core.memory.types import Memory
from finance_project.core.storage.sqlite_db import get_connection, init_db


def _tokenize(text: str) -> set[str]:
    if not text:
        return set()
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) >= 3}


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _score_memory(memory: Memory, query_tokens: set[str]) -> float:
    if not query_tokens:
        return (0.55 * memory.confidence) + (0.45 * memory.importance)

    memory_tokens = _tokenize(memory.content)
    overlap = len(query_tokens & memory_tokens)
    relevance = overlap / len(query_tokens)
    return (0.45 * relevance) + (0.3 * memory.confidence) + (0.25 * memory.importance)


def retrieve_memories(
    user_id: str,
    domain: str,
    query: str,
    session_memory_usage: dict,
    limit: int = 3,
) -> List[Memory]:
    if session_memory_usage is None:
        session_memory_usage = {}
    if domain != "finance":
        return []

    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, type, content, confidence, importance, exposure_count, last_used_at
            FROM user_memories
            WHERE user_id = ? AND domain = ?
            ORDER BY updated_at DESC
            LIMIT 80
            """,
            (user_id, domain),
        ).fetchall()

    query_tokens = _tokenize(query)
    ranked: list[tuple[float, Memory]] = []

    for row in rows:
        memory = Memory(
            id=row["id"],
            type=row["type"],
            content=row["content"],
            confidence=float(row["confidence"]),
            importance=float(row["importance"]),
            exposure_count=int(row["exposure_count"] or 0),
            last_used_at=_parse_timestamp(row["last_used_at"]),
        )

        if memory.confidence < 0.3 or memory.importance < 0.3:
            continue

        if session_memory_usage.get(memory.id, 0) >= 2:
            continue

        ranked.append((_score_memory(memory, query_tokens), memory))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [memory for _, memory in ranked[:limit]]

