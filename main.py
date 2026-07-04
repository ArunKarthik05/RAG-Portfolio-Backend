from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
try:
    from mangum import Mangum as Mangum
except ImportError:
    Mangum = None  # type: ignore[assignment,misc]
from routers import chat, ingest, proof, sources
from routers.github_admin import router as github_admin_router
from routers.conversations import router as conversations_router
from routers.testimonials import router as testimonials_router

app = FastAPI(
    title="RAG Portfolio API",
    description="Arun Karthik's AI-powered portfolio backend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rag-portfolio-kappa.vercel.app/","http://localhost:3000"],  # tighten to your Vercel domain in production
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


# AWS Lambda handler (only active when deployed to Lambda)
handler = Mangum(app, "on", "/") if Mangum is not None else None
