import logging
from urllib.parse import urlparse

from fastapi import Request
from fastapi.responses import JSONResponse

from detection.detection import app
from enhancement import router as enhancement_router
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("origin_allowlist")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


app.include_router(enhancement_router)


def _extract_origin(headers) -> str | None:
    origin = headers.get("origin") or headers.get("Origin")
    if origin:
        return origin
    referer = headers.get("referer") or headers.get("Referer")
    return referer


def _normalize_origin(value: str | None) -> str | None:
    if not value:
        return None
    parsed = urlparse(value)
    if not parsed.scheme or not parsed.netloc:
        return None
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"


ALLOWED_ORIGIN_INPUTS = {
    "https://test-web.real-ezy.com/",
    "https://staging-web.real-ezy.com/",
    "https://dev-web.realezyapp.com/",
    "https://realezy.com/",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://31.97.235.78:8005",
    "http://202.4.125.204",
    "https://dev-admin.realezyapp.com/", 
    "https://staging-admin.real-ezy.com/",
    "https://dev-ai-api.realezyapp.com/"
}

ALLOWED_ORIGINS = {
    normalized
    for normalized in (_normalize_origin(value) for value in ALLOWED_ORIGIN_INPUTS)
    if normalized
}

print(ALLOWED_ORIGINS)
_ALLOWED_NETLOCS = {urlparse(origin).netloc.lower() for origin in ALLOWED_ORIGINS}
_ALLOWED_HOSTS = {
    parsed.hostname.lower()
    for parsed in (urlparse(origin) for origin in ALLOWED_ORIGINS)
    if parsed.hostname
}


def _is_allowed(request: Request) -> bool:
    origin = _normalize_origin(_extract_origin(request.headers))
    if origin and origin in ALLOWED_ORIGINS:
        return True

    host = (request.headers.get("host") or "").lower()
    if host in _ALLOWED_NETLOCS or host in _ALLOWED_HOSTS:
        return True

    logger.warning("Blocked request host=%s origin=%s", host, origin)

    return False


app.add_middleware(
    CORSMiddleware,
    allow_origins=list(ALLOWED_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def enforce_allowlist(request: Request, call_next):
    if not _is_allowed(request):
        logger.info("Rejected request path=%s due to allowlist", request.url.path)
        return JSONResponse(
            status_code=403,
            content={
                "data": {
                    "status": 403,
                    "message": "Forbidden",
                    "info": {
                        "detail": "Origin not allowed for this service.",
                    },
                }
            },
        )

    return await call_next(request)