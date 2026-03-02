# Conversation logging
# app/core/logging/logger.py

import datetime
from typing import Any, Dict, Optional

from finance_project.core.history.repository import save_conversation_turn


def log_event(
    event: str,
    metadata: Optional[Dict[str, Any]] = None,
    level: str = "INFO",
) -> None:
    """
    Lightweight structured log for runtime observability.
    """
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "level": level,
        "event": event,
        "metadata": metadata or {},
    }
    print(f"[SYSTEM LOG] {entry}")


def log_conversation(
    user_id: str,
    domain: str,
    request: str,
    response: str,
    metadata: Dict
) -> None:
    """
    Logs a conversation turn for audit and observability.

    This log is:
    - Immutable
    - Append-only
    - Never used directly for reasoning
    """

    timestamp = datetime.datetime.utcnow().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "user_id": user_id,
        "domain": domain,
        "request": request,
        "response": response,
        "metadata": metadata,
    }

    # TEMPORARY LOGGING
    # Replace with persistent storage (S3 / DB) later
    print(f"[CONVERSATION LOG] {log_entry}")

    try:
        save_conversation_turn(
            user_id=user_id,
            domain=domain,
            request=request,
            response=response,
            metadata=metadata,
            created_at=timestamp,
        )
    except Exception as exc:
        log_event(
            event="conversation_persist_failed",
            metadata={
                "user_id": user_id,
                "domain": domain,
                "error": str(exc),
            },
            level="WARNING",
        )
