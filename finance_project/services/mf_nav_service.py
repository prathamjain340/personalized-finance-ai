"""
Indian Mutual Fund NAV service.
Fetches the AMFI NAVAll.txt flat file and caches all ~15k schemes in SQLite.
Provides fuzzy fund search and MFApi historical NAV lookup.
"""

import logging
import threading
import traceback
from datetime import datetime, timezone, timedelta

import requests
from rapidfuzz import process, fuzz

from finance_project.core.storage.sqlite_db import get_connection

_AMFI_URL = "https://www.amfiindia.com/spages/NAVAll.txt"
_MFAPI_BASE = "https://api.mfapi.in/mf"
_REFRESH_INTERVAL_HOURS = 20
_HTTP_TIMEOUT = 30
_REFRESH_LOCK = threading.Lock()

logger = logging.getLogger(__name__)


def _ensure_mf_table() -> None:
    """Create mutual_funds table if it doesn't exist — safety net in case init_db() raced."""
    try:
        with get_connection() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS mutual_funds (
                    scheme_code TEXT PRIMARY KEY,
                    scheme_name TEXT NOT NULL,
                    nav REAL,
                    nav_date TEXT,
                    updated_at TIMESTAMP
                )"""
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mutual_funds_name ON mutual_funds(scheme_name)"
            )
        logger.info("[mf_nav] mutual_funds table ensured")
    except Exception:
        logger.error("[mf_nav] _ensure_mf_table error:\n%s", traceback.format_exc())


def _should_refresh() -> bool:
    try:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT updated_at FROM mutual_funds ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
        if not row:
            logger.info("[mf_nav] mutual_funds table is empty — refresh needed")
            return True
        last_updated = datetime.fromisoformat(row["updated_at"])
        if last_updated.tzinfo is None:
            last_updated = last_updated.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - last_updated
        logger.info("[mf_nav] last updated %s ago (%.1f hours)", age, age.total_seconds() / 3600)
        return age > timedelta(hours=_REFRESH_INTERVAL_HOURS)
    except Exception:
        logger.error("[mf_nav] _should_refresh error:\n%s", traceback.format_exc())
        return True


def refresh_nav_cache() -> None:
    """Fetch AMFI flat file and upsert all NAV records into SQLite. Thread-safe."""
    _ensure_mf_table()
    with _REFRESH_LOCK:
        if not _should_refresh():
            logger.info("[mf_nav] cache is fresh, skipping refresh")
            return
        logger.info("[mf_nav] starting AMFI refresh from %s", _AMFI_URL)
        try:
            resp = requests.get(_AMFI_URL, timeout=_HTTP_TIMEOUT)
            logger.info("[mf_nav] AMFI HTTP status=%s content_length=%s", resp.status_code, len(resp.content))
            resp.raise_for_status()
            text = resp.content.decode("latin-1")
            all_lines = text.splitlines()
            logger.error("[mf_nav] AMFI first 5 lines: %s", all_lines[:5])
            now = datetime.now(timezone.utc).isoformat()
            rows = []
            skipped = 0
            for line in all_lines:
                parts = line.split(";")
                if len(parts) < 6:
                    skipped += 1
                    continue
                scheme_code = parts[0].strip()
                if not scheme_code.isdigit():
                    skipped += 1
                    continue
                scheme_name = parts[3].strip()
                nav_str = parts[4].strip()
                nav_date = parts[-1].strip()  # last field — works for both 6-field and 8-field formats
                if not scheme_name or nav_str in ("", "N.A."):
                    skipped += 1
                    continue
                try:
                    nav = float(nav_str)
                except ValueError:
                    skipped += 1
                    continue
                rows.append((scheme_code, scheme_name, nav, nav_date, now))
            logger.error("[mf_nav] parsed %d valid rows, skipped %d lines", len(rows), skipped)
            if not rows:
                logger.error("[mf_nav] no valid rows parsed — aborting upsert")
                return
            with get_connection() as conn:
                conn.executemany(
                    """INSERT INTO mutual_funds (scheme_code, scheme_name, nav, nav_date, updated_at)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(scheme_code) DO UPDATE SET
                           scheme_name=excluded.scheme_name,
                           nav=excluded.nav,
                           nav_date=excluded.nav_date,
                           updated_at=excluded.updated_at""",
                    rows,
                )
            logger.error("[mf_nav] upserted %d rows into mutual_funds", len(rows))
        except Exception:
            logger.error("[mf_nav] refresh_nav_cache failed:\n%s", traceback.format_exc())


def search_mf(query: str, top_n: int = 3) -> list[dict]:
    """Fuzzy search mutual fund names. Returns top_n matches with scheme_code, name, nav, nav_date."""
    try:
        with get_connection() as conn:
            all_funds = conn.execute(
                "SELECT scheme_code, scheme_name, nav, nav_date FROM mutual_funds"
            ).fetchall()
        row_count = len(all_funds)
        logger.info("[mf_nav] search_mf query=%r table_rows=%d", query, row_count)
        if not all_funds:
            logger.warning("[mf_nav] search_mf: mutual_funds table is empty — cache not loaded yet")
            return []
        names = [row["scheme_name"] for row in all_funds]
        matches = process.extract(query, names, scorer=fuzz.WRatio, limit=top_n)
        logger.info("[mf_nav] search_mf top matches: %s", [(m[0], m[1]) for m in matches])
        results = []
        for _match_name, score, idx in matches:
            if score < 40:
                continue
            row = all_funds[idx]
            results.append({
                "scheme_code": row["scheme_code"],
                "name": row["scheme_name"],
                "nav": row["nav"],
                "nav_date": row["nav_date"],
            })
        return results
    except Exception:
        logger.error("[mf_nav] search_mf error:\n%s", traceback.format_exc())
        return []


def get_mf_history(scheme_code: str, days: int = 365) -> list[dict]:
    """Fetch historical NAV from MFApi. Returns list of {date, nav} newest-first, capped at days."""
    url = f"{_MFAPI_BASE}/{scheme_code}"
    logger.info("[mf_nav] get_mf_history scheme_code=%s days=%d url=%s", scheme_code, days, url)
    try:
        resp = requests.get(url, timeout=10)
        logger.info("[mf_nav] MFApi HTTP status=%s content_length=%s", resp.status_code, len(resp.content))
        resp.raise_for_status()
        data = resp.json()
        history = (data.get("data") or [])[:days]
        logger.info("[mf_nav] MFApi returned %d history entries (capped to %d)", len(data.get("data") or []), days)
        results = []
        for entry in history:
            try:
                results.append({"date": entry["date"], "nav": float(entry["nav"])})
            except (KeyError, ValueError):
                continue
        return results
    except Exception:
        logger.error("[mf_nav] get_mf_history error:\n%s", traceback.format_exc())
        return []
