import re
from datetime import datetime
from typing import List, Optional

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


def _score_memory(memory: Memory, query_tokens: set[str], preferred_types: set[str] | None = None) -> float:
    if not query_tokens:
        base = (0.55 * memory.confidence) + (0.45 * memory.importance)
        if preferred_types and memory.type in preferred_types:
            base += 0.2
        return base

    memory_tokens = _tokenize(memory.content)
    overlap = len(query_tokens & memory_tokens)
    relevance = overlap / len(query_tokens)
    score = (0.45 * relevance) + (0.3 * memory.confidence) + (0.25 * memory.importance)
    if preferred_types and memory.type in preferred_types:
        score += 0.18
    return score


def retrieve_memories(
    user_id: str,
    domain: str,
    query: str,
    session_memory_usage: dict,
    limit: int = 3,
    allow_exposure_reuse: bool = False,
    preferred_types: Optional[list[str]] = None,
    candidate_scan_limit: int = 80,
    include_assistant_notes: bool = False,
    include_types: Optional[list[str]] = None,
    exclude_types: Optional[list[str]] = None,
) -> List[Memory]:
    if session_memory_usage is None:
        session_memory_usage = {}
    if domain != "finance":
        return []

    scan_limit = max(limit, min(max(20, int(candidate_scan_limit)), 300))
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, type, content, confidence, importance, exposure_count, last_used_at, updated_at
            FROM user_memories
            WHERE user_id = ? AND domain = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (user_id, domain, scan_limit),
        ).fetchall()

    query_tokens = _tokenize(query)
    preferred = {str(item or "").strip().lower() for item in (preferred_types or []) if str(item or "").strip()}
    include_set = {str(item or "").strip().lower() for item in (include_types or []) if str(item or "").strip()}
    exclude_set = {str(item or "").strip().lower() for item in (exclude_types or []) if str(item or "").strip()}
    if not include_assistant_notes and not include_set:
        exclude_set.add("assistant_note")
    ranked: list[tuple[float, Memory]] = []

    for row_index, row in enumerate(rows):
        memory = Memory(
            id=row["id"],
            type=row["type"],
            content=row["content"],
            confidence=float(row["confidence"]),
            importance=float(row["importance"]),
            exposure_count=int(row["exposure_count"] or 0),
            last_used_at=_parse_timestamp(row["last_used_at"]),
        )
        memory_type = str(memory.type or "").strip().lower()

        if memory.confidence < 0.3 or memory.importance < 0.3:
            continue

        if not allow_exposure_reuse and session_memory_usage.get(memory.id, 0) >= 2:
            continue

        if include_set and memory_type not in include_set:
            continue
        if memory_type in exclude_set:
            continue

        score = _score_memory(memory, query_tokens, preferred_types=preferred)
        if memory_type == "assistant_note":
            recency_bonus = max(0.0, 1.0 - (row_index / max(1, len(rows))))
            score += 0.2 * recency_bonus

        ranked.append((score, memory))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [memory for _, memory in ranked[:limit]]

