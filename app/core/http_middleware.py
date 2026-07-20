import hashlib
import json
import logging
import threading
import time
import uuid
from collections import defaultdict, deque

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import get_settings
from app.core.request_scope import DEFAULT_PROJECT_ID, DEFAULT_USER_ID


logger = logging.getLogger("rag.http")
logger.setLevel(logging.INFO)


class RequestObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        started = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            logger.exception(
                json.dumps(
                    {
                        "requestId": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "statusCode": 500,
                        "durationMs": duration_ms,
                    },
                    ensure_ascii=False,
                )
            )
            raise

        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        response.headers["X-Request-Id"] = request_id
        response.headers["X-Response-Time-Ms"] = str(duration_ms)
        logger.info(
            json.dumps(
                {
                    "requestId": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "statusCode": response.status_code,
                    "durationMs": duration_ms,
                },
                ensure_ascii=False,
            )
        )
        return response


class InMemoryRateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.settings = get_settings()
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)

        limit = self.settings.rate_limit_requests_per_minute
        if limit <= 0:
            return await call_next(request)

        user_id = request.headers.get("X-User-Id") or DEFAULT_USER_ID
        project_id = request.headers.get("X-Project-Id") or DEFAULT_PROJECT_ID
        api_key = request.headers.get("X-API-Key") or "missing"
        identity = f"{api_key}:{user_id}:{project_id}".encode("utf-8")
        key = hashlib.sha256(identity).hexdigest()
        now = time.monotonic()
        cutoff = now - 60

        with self._lock:
            bucket = self._requests[key]
            while bucket and bucket[0] < cutoff:
                bucket.popleft()

            if len(bucket) >= limit:
                retry_after = max(1, int(60 - (now - bucket[0])))
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Too many requests. Please retry shortly."},
                    headers={"Retry-After": str(retry_after)},
                )

            bucket.append(now)

        return await call_next(request)
