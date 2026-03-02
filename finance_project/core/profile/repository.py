import datetime
import json
from typing import Any

from finance_project.core.storage.sqlite_db import get_connection, init_db


SOURCE_PRIORITY = {
    "inferred": 1,
    "api": 2,
    "explicit": 3,
}


def _source_rank(source: str) -> int:
    return SOURCE_PRIORITY.get(str(source or "").strip().lower(), 0)


def _clamp_confidence(value: float) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, num))


def _serialize_value(value: Any) -> tuple[str | None, str]:
    if value is None:
        return None, "null"
    if isinstance(value, bool):
        return ("1" if value else "0"), "bool"
    if isinstance(value, int):
        return str(value), "int"
    if isinstance(value, float):
        return repr(value), "float"
    if isinstance(value, str):
        return value, "str"
    return json.dumps(value, ensure_ascii=True), "json"


def _deserialize_value(value: str | None, value_type: str) -> Any:
    if value is None:
        return None

    if value_type == "int":
        try:
            return int(value)
        except ValueError:
            return None
    if value_type == "float":
        try:
            return float(value)
        except ValueError:
            return None
    if value_type == "bool":
        return value == "1"
    if value_type == "json":
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    if value_type == "null":
        return None
    return value


def _is_empty_value(value: str | None, value_type: str) -> bool:
    if value is None:
        return True
    if value_type == "str":
        return value.strip() == ""
    return False


def _should_overwrite(
    existing_source: str,
    existing_value: str | None,
    existing_type: str,
    existing_confidence: float,
    new_source: str,
    new_confidence: float,
) -> bool:
    old_rank = _source_rank(existing_source)
    new_rank = _source_rank(new_source)

    if new_rank > old_rank:
        return True
    if new_rank < old_rank:
        return _is_empty_value(existing_value, existing_type)

    if new_confidence >= existing_confidence:
        return True
    return _is_empty_value(existing_value, existing_type)


def get_profile(user_id: str) -> dict:
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT field_name, field_value, value_type
            FROM user_profile_current
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchall()

    profile: dict[str, Any] = {}
    for row in rows:
        profile[row["field_name"]] = _deserialize_value(row["field_value"], row["value_type"])
    return profile


def merge_profiles(base: dict | None, override: dict | None) -> dict:
    merged = dict(base or {})
    for key, value in (override or {}).items():
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        merged[key] = value
    return merged


def upsert_profile_fields(
    user_id: str,
    updates: dict[str, Any],
    source: str = "inferred",
    confidence: float = 0.7,
) -> dict[str, Any]:
    if not updates:
        return {}

    init_db()
    timestamp = datetime.datetime.utcnow().isoformat()
    source_name = str(source or "inferred").strip().lower()
    safe_confidence = _clamp_confidence(confidence)
    applied: dict[str, Any] = {}

    with get_connection() as conn:
        for field_name_raw, value in updates.items():
            field_name = str(field_name_raw or "").strip()
            if not field_name:
                continue

            serialized_value, value_type = _serialize_value(value)
            existing = conn.execute(
                """
                SELECT field_value, value_type, source, confidence
                FROM user_profile_current
                WHERE user_id = ? AND field_name = ?
                """,
                (user_id, field_name),
            ).fetchone()

            if existing:
                existing_confidence = _clamp_confidence(existing["confidence"])
                if not _should_overwrite(
                    existing_source=existing["source"],
                    existing_value=existing["field_value"],
                    existing_type=existing["value_type"],
                    existing_confidence=existing_confidence,
                    new_source=source_name,
                    new_confidence=safe_confidence,
                ):
                    continue

            old_value = existing["field_value"] if existing else None
            old_type = existing["value_type"] if existing else "null"
            old_python_value = _deserialize_value(old_value, old_type)

            conn.execute(
                """
                INSERT OR REPLACE INTO user_profile_current
                    (user_id, field_name, field_value, value_type, source, confidence, updated_at)
                VALUES
                    (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    field_name,
                    serialized_value,
                    value_type,
                    source_name,
                    safe_confidence,
                    timestamp,
                ),
            )

            conn.execute(
                """
                INSERT INTO user_profile_events
                    (
                        user_id,
                        field_name,
                        old_value,
                        new_value,
                        value_type,
                        source,
                        confidence,
                        created_at
                    )
                VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    field_name,
                    old_value,
                    serialized_value,
                    value_type,
                    source_name,
                    safe_confidence,
                    timestamp,
                ),
            )

            if old_python_value != value:
                applied[field_name] = value

        conn.commit()

    return applied
