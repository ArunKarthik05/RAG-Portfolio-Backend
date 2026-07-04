import os
from fastapi import FastAPI
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
