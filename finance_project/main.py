import os
import threading
import traceback
from pathlib import Path

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from finance_project.api.finance_routes import router as finance_router
from finance_project.core.postprocess.dispatcher import start_postprocess_worker, stop_postprocess_worker
from finance_project.core.storage.sqlite_db import init_db, get_connection
from finance_project.services.mf_nav_service import refresh_nav_cache


def create_app() -> FastAPI:
    app = FastAPI(
        title="Finance Project API",
        version="0.1.0",
        description="Standalone finance assistant API",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(finance_router, prefix="/finance", tags=["finance"])

    @app.on_event("startup")
    def on_startup():
        init_db()
        start_postprocess_worker()
        threading.Thread(target=refresh_nav_cache, daemon=True).start()

    @app.on_event("shutdown")
    def on_shutdown():
        stop_postprocess_worker()

    @app.get("/", tags=["system"])
    def root():
        return {
            "service": "finance_project",
            "status": "running",
            "health": "/health",
            "docs": "/docs",
            "voice_lab": "/voice-lab",
        }

    @app.get("/health", tags=["system"])
    def health():
        return {"status": "ok"}

    @app.get("/debug/mf-status", tags=["debug"])
    def debug_mf_status():
        result = {}

        # --- DB check ---
        try:
            with get_connection() as conn:
                tables = [r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='mutual_funds'"
                ).fetchall()]
                result["table_exists"] = "mutual_funds" in tables
                if result["table_exists"]:
                    result["row_count"] = conn.execute(
                        "SELECT COUNT(*) FROM mutual_funds"
                    ).fetchone()[0]
                    row = conn.execute(
                        "SELECT updated_at FROM mutual_funds ORDER BY updated_at DESC LIMIT 1"
                    ).fetchone()
                    result["last_updated_at"] = row["updated_at"] if row else None
                    sample = conn.execute(
                        "SELECT scheme_code, scheme_name, nav, nav_date FROM mutual_funds LIMIT 1"
                    ).fetchone()
                    result["sample_row"] = dict(sample) if sample else None
                else:
                    result["row_count"] = 0
                    result["last_updated_at"] = None
                    result["sample_row"] = None
        except Exception:
            result["db_error"] = traceback.format_exc()

        # --- Live AMFI fetch test (first 3 lines) ---
        try:
            resp = requests.get(
                "https://www.amfiindia.com/spages/NAVAll.txt",
                timeout=10,
            )
            result["amfi_http_status"] = resp.status_code
            result["amfi_content_length"] = len(resp.content)
            result["amfi_first_lines"] = resp.text.splitlines()[:5]
        except Exception:
            result["amfi_error"] = traceback.format_exc()

        # --- Live MFApi test (scheme 120503 = Parag Parikh Flexi Cap) ---
        try:
            resp = requests.get("https://api.mfapi.in/mf/120503/latest", timeout=10)
            result["mfapi_http_status"] = resp.status_code
            result["mfapi_sample"] = resp.json()
        except Exception:
            result["mfapi_error"] = traceback.format_exc()

        return result

    @app.get("/voice-lab", tags=["system"])
    def voice_lab():
        page_path = Path(__file__).resolve().parent / "ui" / "voice_lab.html"
        return FileResponse(page_path)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "9001"))
    reload_enabled = os.getenv("RELOAD", "false").lower() == "true"
    uvicorn.run("finance_project.main:app", host=host, port=port, reload=reload_enabled)


# python -m uvicorn finance_project.main:app --host 127.0.0.1 --port 9003
