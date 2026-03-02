from datetime import datetime
import re
from uuid import uuid4

from finance_project.core.memory.types import Memory
from finance_project.core.storage.sqlite_db import get_connection, init_db


def _clamp_score(value: float, default: float = 0.5) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, numeric))


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _ensure_terminal(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return cleaned
    if cleaned[-1] in ".!?":
        return cleaned
    return f"{cleaned}."


def _clean_fragment(value: str) -> str:
    cleaned = _normalize_spaces(value).strip(" .,!?:;")
    return cleaned


def _split_preference_value(value: str) -> str:
    cleaned = _clean_fragment(value)
    if not cleaned:
        return ""
    # Keep the first factual clause only; avoid joining separate facts into one memory.
    clipped = re.split(
        r"\b(?:and i|and my|and spend|but i|also i|whereas|while|because|since)\b",
        cleaned,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return _clean_fragment(clipped)


def _canonicalize_preference(content: str) -> str:
    body = re.sub(r"^\s*user\s+", "", _normalize_spaces(content), flags=re.IGNORECASE)
    lowered = body.lower()

    if lowered.startswith("likes "):
        value = _split_preference_value(body[6:])
        if value:
            return f"User likes {value}."
    if lowered.startswith("dislikes "):
        value = _split_preference_value(body[9:])
        if value:
            return f"User dislikes {value}."
    if lowered.startswith("prefers "):
        value = _split_preference_value(body[8:])
        if value:
            return f"User prefers {value}."

    return _ensure_terminal(f"User {_clean_fragment(body)}")


def _canonicalize_behavioral(content: str) -> str:
    normalized = _normalize_spaces(content)
    match = re.search(r"major spending categories\s*:\s*(.+)", normalized, flags=re.IGNORECASE)
    if not match:
        return _ensure_terminal(_clean_fragment(normalized))

    categories_raw = match.group(1)
    parts = re.split(r",|/|&|\band\b", categories_raw, flags=re.IGNORECASE)
    categories: list[str] = []
    seen: set[str] = set()
    for part in parts:
        cleaned = _clean_fragment(part).lower()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        categories.append(cleaned)

    if not categories:
        return "User major spending categories."

    return f"User major spending categories: {', '.join(categories[:6])}."


def _canonicalize_content(memory_type: str, content: str) -> str:
    normalized = _normalize_spaces(content)
    if memory_type == "preference":
        return _canonicalize_preference(normalized)
    if memory_type == "behavioral":
        return _canonicalize_behavioral(normalized)
    return _ensure_terminal(_clean_fragment(normalized))


def normalize_memory_candidate(memory: dict) -> dict | None:
    if not isinstance(memory, dict):
        return None

    memory_type = str(memory.get("type") or "").strip().lower()
    raw_content = str(memory.get("content") or "").strip()
    if not memory_type or not raw_content:
        return None

    content = _canonicalize_content(memory_type, raw_content)
    if not content:
        return None

    return {
        "type": memory_type,
        "content": content,
        "confidence": _clamp_score(memory.get("confidence", 0.5)),
        "importance": _clamp_score(memory.get("importance", 0.5)),
    }


def store_memory(user_id: str, domain: str, memory: dict) -> Memory | None:
    """
    Upserts a long-term memory and returns its stored representation.
    """
    normalized = normalize_memory_candidate(memory)
    if not normalized:
        return None

    memory_type = normalized["type"]
    content = normalized["content"]
    confidence = normalized["confidence"]
    importance = normalized["importance"]
    now = datetime.utcnow().isoformat()
    init_db()

    with get_connection() as conn:
        existing = conn.execute(
            """
            SELECT id, confidence, importance, exposure_count, last_used_at
            FROM user_memories
            WHERE user_id = ? AND domain = ? AND type = ? AND content = ?
            """,
            (user_id, domain, memory_type, content),
        ).fetchone()

        if existing:
            memory_id = existing["id"]
            merged_confidence = max(confidence, _clamp_score(existing["confidence"], default=confidence))
            merged_importance = max(importance, _clamp_score(existing["importance"], default=importance))
            exposure_count = int(existing["exposure_count"] or 0)
            last_used_at = _parse_timestamp(existing["last_used_at"])

            conn.execute(
                """
                UPDATE user_memories
                SET confidence = ?, importance = ?, updated_at = ?
                WHERE id = ?
                """,
                (merged_confidence, merged_importance, now, memory_id),
            )
        else:
            memory_id = str(uuid4())
            merged_confidence = confidence
            merged_importance = importance
            exposure_count = 0
            last_used_at = None

            conn.execute(
                """
                INSERT INTO user_memories
                    (
                        id,
                        user_id,
                        domain,
                        type,
                        content,
                        confidence,
                        importance,
                        exposure_count,
                        last_used_at,
                        created_at,
                        updated_at
                    )
                VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    user_id,
                    domain,
                    memory_type,
                    content,
                    merged_confidence,
                    merged_importance,
                    exposure_count,
                    None,
                    now,
                    now,
                ),
            )

        conn.commit()

    return Memory(
        id=memory_id,
        type=memory_type,
        content=content,
        confidence=merged_confidence,
        importance=merged_importance,
        exposure_count=exposure_count,
        last_used_at=last_used_at,
    )
