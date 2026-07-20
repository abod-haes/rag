import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat_v2 import router as chat_router
from app.api.documents_v2 import router as documents_router
from app.api.health import router as health_router
from app.core.config import settings
from app.core.http_middleware import (
    InMemoryRateLimitMiddleware,
    RequestObservabilityMiddleware,
)
from app.db.database import init_db


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Independent RAG API for PDF-based question answering.",
    )

    # Middleware is registered inner-to-outer so observability also covers
    # CORS and rate-limit responses.
    app.add_middleware(InMemoryRateLimitMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestObservabilityMiddleware)

    @app.on_event("startup")
    def on_startup() -> None:
        for attempt in range(10):
            try:
                init_db()
                return
            except Exception:
                if attempt == 9:
                    raise
                time.sleep(2)

    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(chat_router)

    return app


app = create_app()
