import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from finance_project.api.finance_routes import router as finance_router
from finance_project.core.postprocess.dispatcher import start_postprocess_worker, stop_postprocess_worker
from finance_project.core.storage.sqlite_db import init_db


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
