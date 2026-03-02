import atexit
import queue
import threading
from typing import Any

from finance_project.core.logging.logger import log_event
from finance_project.core.memory.store import normalize_memory_candidate, store_memory
from finance_project.core.profile.repository import upsert_profile_fields


_QUEUE_MAXSIZE = 500
_TASK_QUEUE: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=_QUEUE_MAXSIZE)
_WORKER_LOCK = threading.Lock()
_WORKER_THREAD: threading.Thread | None = None
_STOP_EVENT = threading.Event()


def _process_task(task: dict[str, Any]) -> None:
    kind = task.get("kind")

    if kind == "profile_updates":
        upsert_profile_fields(
            user_id=task["user_id"],
            updates=task["updates"],
            source=task.get("source", "inferred"),
            confidence=float(task.get("confidence", 0.7)),
        )
        return

    if kind == "memory_candidates":
        user_id = task["user_id"]
        domain = task["domain"]
        for candidate in task.get("memory_candidates", []):
            try:
                store_memory(user_id=user_id, domain=domain, memory=candidate)
            except Exception as exc:
                log_event(
                    event="memory_candidate_store_failed",
                    metadata={"user_id": user_id, "domain": domain, "error": str(exc)},
                    level="WARNING",
                )
        return


def _worker_loop() -> None:
    while not _STOP_EVENT.is_set():
        try:
            task = _TASK_QUEUE.get(timeout=0.4)
        except queue.Empty:
            continue

        try:
            if task is None:
                return
            _process_task(task)
        except Exception as exc:
            log_event(
                event="postprocess_task_failed",
                metadata={"error": str(exc), "task_kind": (task or {}).get("kind")},
                level="WARNING",
            )
        finally:
            _TASK_QUEUE.task_done()


def start_postprocess_worker() -> None:
    global _WORKER_THREAD
    with _WORKER_LOCK:
        if _WORKER_THREAD and _WORKER_THREAD.is_alive():
            return

        _STOP_EVENT.clear()
        _WORKER_THREAD = threading.Thread(
            target=_worker_loop,
            name="finance-postprocess-worker",
            daemon=True,
        )
        _WORKER_THREAD.start()


def stop_postprocess_worker(timeout_seconds: float = 1.5) -> None:
    global _WORKER_THREAD
    with _WORKER_LOCK:
        if not _WORKER_THREAD:
            return

        _STOP_EVENT.set()
        try:
            _TASK_QUEUE.put_nowait(None)
        except queue.Full:
            pass
        _WORKER_THREAD.join(timeout=timeout_seconds)
        _WORKER_THREAD = None


def _enqueue_task(task: dict[str, Any]) -> None:
    start_postprocess_worker()
    try:
        _TASK_QUEUE.put_nowait(task)
    except queue.Full:
        log_event(
            event="postprocess_queue_full",
            metadata={"task_kind": task.get("kind"), "maxsize": _QUEUE_MAXSIZE},
            level="WARNING",
        )


def enqueue_profile_updates(
    user_id: str,
    updates: dict[str, Any],
    source: str = "inferred",
    confidence: float = 0.7,
) -> None:
    clean_updates = {k: v for k, v in (updates or {}).items() if str(k).strip()}
    if not clean_updates:
        return

    _enqueue_task(
        {
            "kind": "profile_updates",
            "user_id": user_id,
            "updates": clean_updates,
            "source": source,
            "confidence": confidence,
        }
    )


def enqueue_memory_candidates(
    user_id: str,
    domain: str,
    memory_candidates: list[dict[str, Any]],
) -> None:
    if not memory_candidates:
        return

    clean_candidates = []
    seen: set[tuple[str, str]] = set()
    for candidate in memory_candidates:
        normalized = normalize_memory_candidate(candidate)
        if not normalized:
            continue
        dedupe_key = (
            normalized["type"],
            str(normalized["content"]).strip().lower(),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        clean_candidates.append(normalized)

    if not clean_candidates:
        return

    _enqueue_task(
        {
            "kind": "memory_candidates",
            "user_id": user_id,
            "domain": domain,
            "memory_candidates": clean_candidates,
        }
    )


atexit.register(stop_postprocess_worker)
