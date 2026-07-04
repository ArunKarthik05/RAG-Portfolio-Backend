"""
Pydantic schemas shared across routers and the RAG pipeline.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel


# ── Ingest ────────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    source_type: str
    chunks_added: int
    chunks_updated: int
    status: str


# ── Chat ──────────────────────────────────────────────────────────────────────

class ConversationMessage(BaseModel):
    role: str   # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str
    source_types: Optional[list[str]] = None
    repo_filter: Optional[list[str]] = None
    conversation_history: Optional[list[ConversationMessage]] = None  # prior turns
    conversation_id: Optional[str] = None                             # for logging


class CitationChunk(BaseModel):
    chunk_id: str
    source_type: str
    source_url: Optional[str] = None
    source_title: str
    chunk_text: str
    similarity_score: float = 0.0
    rerank_score: Optional[float] = None
    rrf_score: Optional[float] = None
    date_indexed: Optional[Any] = None


class ChatResponse(BaseModel):
    answer: str
    proof_id: str
    citations: list[CitationChunk]
    model_used: str
    created_at: datetime


# ── Conversations ─────────────────────────────────────────────────────────────

class ConversationCreate(BaseModel):
    user_id: str
    is_guest: bool = False
    title: Optional[str] = None


class ConversationOut(BaseModel):
    id: str
    user_id: str
    is_guest: bool
    title: Optional[str]
    created_at: Any
    updated_at: Any


class MessageOut(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    citations: list[Any] = []
    proof_id: Optional[str] = None
    created_at: Any


# ── Proof ─────────────────────────────────────────────────────────────────────

class ProofRecord(BaseModel):
    id: str
    question: str
    answer: str
    citations: list[Any]
    model_used: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    created_at: Any
    visitor_ip_hash: Optional[str] = None
