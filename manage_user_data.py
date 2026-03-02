"""
Small admin CLI for local finance DB user data.

Examples:
  python manage_user_data.py view --user-id 981
  python manage_user_data.py set --user-id 981 --field city --value Delhi
  python manage_user_data.py set --user-id 981 --field monthly_income --value 80000
  python manage_user_data.py delete --user-id 981 --target profile --field city
  python manage_user_data.py delete --user-id 981 --target memories
  python manage_user_data.py delete --user-id 981 --target all --yes
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Any

from finance_project.core.profile.repository import upsert_profile_fields
from finance_project.core.storage.sqlite_db import get_connection, get_db_path, init_db


def _parse_value(raw: str) -> Any:
    text = str(raw).strip()
    lowered = text.lower()

    if lowered in {"null", "none"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    if re.fullmatch(r"-?\d+", text):
        try:
            return int(text)
        except ValueError:
            pass
    if re.fullmatch(r"-?\d+\.\d+", text):
        try:
            return float(text)
        except ValueError:
            pass

    if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    return text


def _rows_to_dicts(rows) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def _print_json(title: str, payload: Any) -> None:
    print(f"\n[{title}]")
    print(json.dumps(payload, indent=2, ensure_ascii=True, default=str))


def cmd_view(args: argparse.Namespace) -> int:
    init_db()
    with get_connection() as conn:
        profile_rows = conn.execute(
            """
            SELECT field_name, field_value, value_type, source, confidence, updated_at
            FROM user_profile_current
            WHERE user_id = ?
            ORDER BY field_name ASC
            """,
            (args.user_id,),
        ).fetchall()

        memory_rows = conn.execute(
            """
            SELECT id, type, content, confidence, importance, exposure_count, last_used_at, updated_at
            FROM user_memories
            WHERE user_id = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (args.user_id, args.limit),
        ).fetchall()

        conversation_rows = conn.execute(
            """
            SELECT id, domain, request, response, metadata_json, created_at
            FROM conversation_turns
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (args.user_id, args.limit),
        ).fetchall()

        payload = {
            "db_path": get_db_path(),
            "user_id": args.user_id,
            "counts": {
                "profile_current": conn.execute(
                    "SELECT COUNT(1) AS c FROM user_profile_current WHERE user_id = ?",
                    (args.user_id,),
                ).fetchone()["c"],
                "profile_events": conn.execute(
                    "SELECT COUNT(1) AS c FROM user_profile_events WHERE user_id = ?",
                    (args.user_id,),
                ).fetchone()["c"],
                "memories": conn.execute(
                    "SELECT COUNT(1) AS c FROM user_memories WHERE user_id = ?",
                    (args.user_id,),
                ).fetchone()["c"],
                "conversation_turns": conn.execute(
                    "SELECT COUNT(1) AS c FROM conversation_turns WHERE user_id = ?",
                    (args.user_id,),
                ).fetchone()["c"],
            },
            "profile_current": _rows_to_dicts(profile_rows),
            "memories_recent": _rows_to_dicts(memory_rows),
            "conversation_recent": _rows_to_dicts(conversation_rows),
        }

    _print_json("USER DATA", payload)
    return 0


def cmd_set(args: argparse.Namespace) -> int:
    value = _parse_value(args.value)
    applied = upsert_profile_fields(
        user_id=args.user_id,
        updates={args.field: value},
        source=args.source,
        confidence=args.confidence,
    )

    result = {
        "db_path": get_db_path(),
        "user_id": args.user_id,
        "field": args.field,
        "input_value": args.value,
        "parsed_value": value,
        "applied": applied,
        "status": "updated" if args.field in applied else "no_change",
    }
    _print_json("SET RESULT", result)
    return 0


def _delete_profile(conn, user_id: str, field: str | None) -> dict[str, int]:
    if field:
        a = conn.execute(
            "DELETE FROM user_profile_current WHERE user_id = ? AND field_name = ?",
            (user_id, field),
        ).rowcount
        b = conn.execute(
            "DELETE FROM user_profile_events WHERE user_id = ? AND field_name = ?",
            (user_id, field),
        ).rowcount
        return {"profile_current_deleted": a, "profile_events_deleted": b}

    a = conn.execute(
        "DELETE FROM user_profile_current WHERE user_id = ?",
        (user_id,),
    ).rowcount
    b = conn.execute(
        "DELETE FROM user_profile_events WHERE user_id = ?",
        (user_id,),
    ).rowcount
    return {"profile_current_deleted": a, "profile_events_deleted": b}


def _delete_memories(conn, user_id: str, memory_id: str | None) -> dict[str, int]:
    if memory_id:
        deleted = conn.execute(
            "DELETE FROM user_memories WHERE user_id = ? AND id = ?",
            (user_id, memory_id),
        ).rowcount
        return {"memories_deleted": deleted}
    deleted = conn.execute(
        "DELETE FROM user_memories WHERE user_id = ?",
        (user_id,),
    ).rowcount
    return {"memories_deleted": deleted}


def _delete_conversations(conn, user_id: str) -> dict[str, int]:
    deleted = conn.execute(
        "DELETE FROM conversation_turns WHERE user_id = ?",
        (user_id,),
    ).rowcount
    return {"conversation_turns_deleted": deleted}


def cmd_delete(args: argparse.Namespace) -> int:
    target = args.target
    if target == "all" and not args.yes:
        print("Refused: use --yes with --target all.")
        return 2

    init_db()
    with get_connection() as conn:
        summary: dict[str, int] = {}
        if target in {"profile", "all"}:
            summary.update(_delete_profile(conn, args.user_id, args.field if target == "profile" else None))
        if target in {"memories", "all"}:
            summary.update(_delete_memories(conn, args.user_id, args.memory_id if target == "memories" else None))
        if target in {"conversations", "all"}:
            summary.update(_delete_conversations(conn, args.user_id))
        conn.commit()

    _print_json(
        "DELETE RESULT",
        {
            "db_path": get_db_path(),
            "user_id": args.user_id,
            "target": target,
            "summary": summary,
        },
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finance DB admin tool for one user.")
    sub = parser.add_subparsers(dest="command", required=True)

    view = sub.add_parser("view", help="View user profile, memories, and recent conversation.")
    view.add_argument("--user-id", required=True)
    view.add_argument("--limit", type=int, default=10, help="Recent rows limit for memories/conversations.")
    view.set_defaults(func=cmd_view)

    set_cmd = sub.add_parser("set", help="Set one profile field for a user.")
    set_cmd.add_argument("--user-id", required=True)
    set_cmd.add_argument("--field", required=True)
    set_cmd.add_argument("--value", required=True, help="Auto-parsed: int/float/bool/null/json/string.")
    set_cmd.add_argument("--source", default="explicit", choices=["explicit", "api", "inferred"])
    set_cmd.add_argument("--confidence", type=float, default=1.0)
    set_cmd.set_defaults(func=cmd_set)

    delete = sub.add_parser("delete", help="Delete user data by target.")
    delete.add_argument("--user-id", required=True)
    delete.add_argument("--target", required=True, choices=["profile", "memories", "conversations", "all"])
    delete.add_argument("--field", help="Only for --target profile: delete one profile field.")
    delete.add_argument("--memory-id", help="Only for --target memories: delete one memory id.")
    delete.add_argument("--yes", action="store_true", help="Required for --target all.")
    delete.set_defaults(func=cmd_delete)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

# # 1) View one user (profile + memories + recent conversation)
# python manage_user_data.py view --user-id 981

# # 2) Edit one profile field
# python manage_user_data.py set --user-id 981 --field city --value Delhi
# python manage_user_data.py set --user-id 981 --field monthly_income --value 80000

# # 3) Delete one profile field
# python manage_user_data.py delete --user-id 981 --target profile --field city

# # 4) Delete all profile rows/events for that user
# python manage_user_data.py delete --user-id 981 --target profile

# # 5) Delete memories
# python manage_user_data.py delete --user-id 981 --target memories

# # 6) Delete conversation history
# python manage_user_data.py delete --user-id 981 --target conversations

# # 7) Full local reset for one user (profile + memories + conversation)
# python manage_user_data.py delete --user-id 981 --target all --yes