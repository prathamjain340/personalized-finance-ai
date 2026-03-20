import argparse
import os
import re
from typing import Any

from finance_project.core.memory.store import normalize_memory_candidate, store_memory
from finance_project.core.storage.sqlite_db import get_connection, init_db


def _normalize_text_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _extract_list_values(content: str) -> list[str]:
    text = str(content or "").strip().rstrip(".")
    text = re.sub(r"^user\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"^(?:interest\s*:|likes?\s+|prefers?\s+|dislikes?\s+|hobbies?\s*:\s*|hobbies?\s+include\s+)",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    if "," not in text and " and " not in text.lower() and "/" not in text:
        return []

    parts = re.split(r",|/|\band\b|&", text, flags=re.IGNORECASE)
    atoms: list[str] = []
    seen: set[str] = set()
    for part in parts:
        cleaned = re.sub(r"\s+", " ", part).strip(" .,!?:;").lower()
        if not cleaned or len(cleaned.split()) > 5:
            continue
        tail = cleaned.split()[-1]
        if len(tail) <= 2 and tail not in {"tv", "ai", "vr", "ux"}:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        atoms.append(cleaned)
    if len(atoms) < 2:
        return []
    return atoms[:6]


def _split_obvious_compound(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    memory_type = str(candidate.get("type") or "").strip().lower()
    content = str(candidate.get("content") or "").strip()
    if memory_type not in {"preference", "interest", "dislike"}:
        return [candidate]

    values = _extract_list_values(content)
    if not values:
        return [candidate]

    target_type = "dislike" if memory_type == "dislike" else "interest"
    split_candidates: list[dict[str, Any]] = []
    for value in values:
        if target_type == "dislike":
            split_content = f"User dislikes {value}."
        else:
            split_content = f"User interest: {value}."
        split_candidates.append(
            {
                "type": target_type,
                "content": split_content,
                "confidence": candidate.get("confidence", 0.7),
                "importance": candidate.get("importance", 0.6),
            }
        )
    return split_candidates


def _is_low_quality_candidate(candidate: dict[str, Any]) -> bool:
    memory_type = str(candidate.get("type") or "").strip().lower()
    if memory_type not in {"interest", "preference", "dislike"}:
        return False
    content = str(candidate.get("content") or "").strip().lower().rstrip(".")
    content = re.sub(r"^user\s+", "", content, flags=re.IGNORECASE)
    content = re.sub(r"^(?:interest\s*:|likes?\s+|prefers?\s+|dislikes?\s+)", "", content, flags=re.IGNORECASE).strip()
    if not content:
        return True
    tail = content.split()[-1]
    return len(tail) <= 2 and tail not in {"tv", "ai", "vr", "ux"}


def _fetch_rows(user_id: str | None, domain: str) -> list[dict[str, Any]]:
    init_db()
    with get_connection() as conn:
        if user_id:
            rows = conn.execute(
                """
                SELECT id, user_id, domain, type, content, confidence, importance, updated_at
                FROM user_memories
                WHERE user_id = ? AND domain = ?
                ORDER BY updated_at DESC
                """,
                (user_id, domain),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, user_id, domain, type, content, confidence, importance, updated_at
                FROM user_memories
                WHERE domain = ?
                ORDER BY updated_at DESC
                """,
                (domain,),
            ).fetchall()
    return [dict(row) for row in rows]


def _delete_memory(memory_id: str) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM user_memories WHERE id = ?", (memory_id,))
        conn.commit()


def _dedupe_near_duplicates(user_id: str | None, domain: str, apply_changes: bool) -> int:
    rows = _fetch_rows(user_id=user_id, domain=domain)
    buckets: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["user_id"],
            row["domain"],
            str(row["type"] or "").strip().lower(),
            _normalize_text_key(row["content"]),
        )
        buckets.setdefault(key, []).append(row)

    duplicates_removed = 0
    for _, bucket in buckets.items():
        if len(bucket) <= 1:
            continue
        bucket_sorted = sorted(
            bucket,
            key=lambda item: (
                str(item.get("updated_at") or ""),
                float(item.get("confidence") or 0.0),
                float(item.get("importance") or 0.0),
            ),
            reverse=True,
        )
        for loser in bucket_sorted[1:]:
            duplicates_removed += 1
            if apply_changes:
                _delete_memory(loser["id"])
    return duplicates_removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize and clean user_memories rows.")
    parser.add_argument("--user-id", type=str, default=None, help="Optional user_id filter")
    parser.add_argument("--domain", type=str, default="finance", help="Domain filter (default: finance)")
    parser.add_argument("--db-path", type=str, default=None, help="Optional sqlite path override")
    parser.add_argument("--apply", action="store_true", help="Apply cleanup changes")
    args = parser.parse_args()

    if args.db_path:
        os.environ["FINANCE_DB_PATH"] = args.db_path

    rows = _fetch_rows(user_id=args.user_id, domain=args.domain)
    scanned = len(rows)
    rewritten = 0
    inserted = 0
    dropped = 0

    for row in rows:
        base_candidate = {
            "type": row["type"],
            "content": row["content"],
            "confidence": row["confidence"],
            "importance": row["importance"],
        }
        normalized = normalize_memory_candidate(base_candidate)
        if not normalized:
            dropped += 1
            continue

        split_candidates = _split_obvious_compound(normalized)
        final_candidates: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str]] = set()
        for candidate in split_candidates:
            normalized_split = normalize_memory_candidate(candidate)
            if not normalized_split:
                continue
            if _is_low_quality_candidate(normalized_split):
                continue
            key = (
                str(normalized_split["type"]).strip().lower(),
                str(normalized_split["content"]).strip().lower(),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            final_candidates.append(normalized_split)

        original_key = (
            str(row["type"]).strip().lower(),
            str(row["content"]).strip().lower(),
        )
        final_keys = {(item["type"], str(item["content"]).strip().lower()) for item in final_candidates}
        changed = len(final_candidates) != 1 or original_key not in final_keys
        if not changed:
            continue

        rewritten += 1
        inserted += len(final_candidates)
        if args.apply:
            for candidate in final_candidates:
                store_memory(
                    user_id=row["user_id"],
                    domain=row["domain"],
                    memory={
                        "type": candidate["type"],
                        "content": candidate["content"],
                        "confidence": candidate.get("confidence", row.get("confidence", 0.7)),
                        "importance": candidate.get("importance", row.get("importance", 0.6)),
                    },
                )
            _delete_memory(row["id"])

    duplicates_removed = _dedupe_near_duplicates(args.user_id, args.domain, apply_changes=args.apply)

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] scanned_rows={scanned}")
    print(f"[{mode}] rewritten_rows={rewritten}")
    print(f"[{mode}] inserted_rows={inserted}")
    print(f"[{mode}] dropped_invalid_rows={dropped}")
    print(f"[{mode}] near_duplicates_removed={duplicates_removed}")


if __name__ == "__main__":
    main()
