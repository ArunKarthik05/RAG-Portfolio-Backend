import os
import logging
from fastapi import FastAPI
from config import get_settings

# ── Logging setup ─────────────────────────────────────────────────────────────
# Set DEBUG_LOGS=true in .env (or Railway env) to enable verbose app logging.
# Third-party low-level libraries are always kept at WARNING regardless.

_settings = get_settings()
_app_level = logging.DEBUG if _settings.debug_logs else logging.INFO

logging.basicConfig(
    level=_app_level,
    format="%(levelname)s:%(name)s:%(message)s",
)

# Always suppress noisy internals, regardless of debug_logs flag
for _noisy in ("hpack", "httpcore", "httpx", "h2", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
from fastapi.middleware.cors import CORSMiddleware
from routers import chat, ingest, proof, sources
from routers.github_admin import router as github_admin_router
from routers.conversations import router as conversations_router
from routers.testimonials import router as testimonials_router

app = FastAPI(
    title="RAG Portfolio API",
    description="Arun Karthik's AI-powered portfolio backend",
    version="1.0.0",
)

# ALLOWED_ORIGINS env var: comma-separated list of allowed origins.
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
allowed_origins = [o.strip().rstrip("/") for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(ingest.router)
app.include_router(proof.router)
app.include_router(sources.router)
app.include_router(github_admin_router)
app.include_router(conversations_router)
app.include_router(testimonials_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
