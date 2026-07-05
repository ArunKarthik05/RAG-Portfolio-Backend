"""
Core RAG pipeline:
  1. Embed the question
  2. Hybrid retrieval — dense (pgvector) + sparse (FTS) fused with RRF
  3. Build grounded prompt with source citations
  4. Call GPT-4o and stream the response
  5. Log proof record to Supabase

Observability (Langfuse):
  Every request produces a trace with child spans/generations:
    trace: run_rag_chat | stream_rag_chat
      ├── span       : embed_question
      ├── span       : cache_check        (stream only)
      ├── span       : hybrid_retrieval
      ├── generation : gpt-4o             (tokens, latency)
      ├── span       : proof_log
      ├── span       : cache_populate     (stream only)
      └── generation : suggestions        (gpt-4o-mini tokens)
"""
import json
import uuid
import hashlib
from datetime import datetime, timezone
from typing import AsyncIterator

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from supabase import create_client

from config import get_settings
from rag.embeddings import get_query_embedding
from rag.store import hybrid_search
from rag.cache import check_cache, populate_cache
from langfuse.model import ModelUsage
from rag.observability import get_langfuse
from models.schemas import CitationChunk, ChatResponse, ProofRecord, ConversationMessage


SYSTEM_PROMPT = """\
You are an intelligent portfolio assistant for {owner_name}.

Answer questions from recruiters and visitors based ONLY on the provided context chunks.
Each chunk has a [SOURCE N] label — cite sources inline like [SOURCE 1] or [SOURCE 1, SOURCE 3].
If the context doesn't contain enough information, say so — do NOT invent details.

Format your response in Markdown, but use formatting sparingly and only where it genuinely helps:
- Write in clear prose paragraphs as the default. Separate every paragraph with a blank line.
- Use a `###` heading only when the answer has two or more clearly distinct sections.
- Use **bold** only for a proper noun or technical term the first time it appears — not repeatedly.
- Use a bullet list only for three or more parallel items (e.g. a list of skills or projects). Put a blank line before and after every list.
- Never bold entire sentences or wrap headings around single points — that makes it harder to read.
- Keep the response airy: short paragraphs, breathing room between sections.

Highlight {owner_name}'s strengths naturally and professionally.\
"""


def _build_messages(
    system: str,
    context_block: str,
    question: str,
    history: list[ConversationMessage] | None,
) -> list[ChatCompletionMessageParam]:
    """Assemble the GPT-4o messages array with optional conversation history."""
    msgs: list[ChatCompletionMessageParam] = [{"role": "system", "content": system}]
    # Inject prior turns (last 10 to stay within token budget)
    for turn in (history or [])[-10:]:
        msgs.append({"role": turn.role, "content": turn.content})  # type: ignore[arg-type]
    # Current question with RAG context
    msgs.append({
        "role": "user",
        "content": f"Context:\n{context_block}\n\nQuestion: {question}\n\nAnswer (cite sources inline):",
    })
    return msgs


async def run_rag_chat(
    question: str,
    visitor_ip: str = "",
    source_types: list[str] | None = None,
    repo_filter: list[str] | None = None,
    conversation_history: list[ConversationMessage] | None = None,
) -> ChatResponse:
    settings = get_settings()
    oai = AsyncOpenAI(api_key=settings.openai_api_key)
    lf = get_langfuse()
    ip_hash = hashlib.sha256(visitor_ip.encode()).hexdigest()[:16] if visitor_ip else None

    trace = lf.trace(
        name="run_rag_chat",
        input={"question": question},
        metadata={"source_types": source_types, "repo_filter": repo_filter, "visitor_ip_hash": ip_hash},
    )

    # 1. Embed question
    embed_span = trace.span(name="embed_question", input={"question": question})
    q_embedding = await get_query_embedding(question)
    embed_span.end(output={"embedding_dim": len(q_embedding)})

    # 2. Hybrid retrieval (dense + sparse, RRF-fused)
    retrieval_span = trace.span(
        name="hybrid_retrieval",
        input={"candidate_count": 20, "cutoff": settings.similarity_cutoff},
    )
    candidates = hybrid_search(
        query=question,
        query_embedding=q_embedding,
        candidate_count=20,
        similarity_cutoff=settings.similarity_cutoff,
        source_types=source_types,
        repo_filter=repo_filter,
    )
    raw_chunks = candidates[:settings.top_k_retrieval]
    retrieval_span.end(output={
        "candidates": len(candidates),
        "chunks_used": len(raw_chunks),
        "top_scores": [round(c.get("similarity", 0), 3) for c in raw_chunks[:3]],
    })

    # 3. Build citations list
    citations: list[CitationChunk] = [
        CitationChunk(
            chunk_id=c["id"],
            source_type=c["source_type"],
            source_url=c.get("source_url"),
            source_title=c.get("source_title", ""),
            chunk_text=c["chunk_text"],
            similarity_score=round(c.get("similarity", 0.0), 4),
            date_indexed=c.get("date_indexed"),
        )
        for c in raw_chunks
    ]

    # 4. Build context block for the prompt
    context_block = "\n\n".join(
        f"[SOURCE {i + 1}] ({cit.source_type} — {cit.source_title})\n{cit.chunk_text}"
        for i, cit in enumerate(citations)
    )

    messages = _build_messages(
        SYSTEM_PROMPT.format(owner_name=settings.app_owner_name),
        context_block, question, conversation_history,
    )

    # 5. Call GPT-4o
    generation = trace.generation(
        name="gpt-4o",
        model="gpt-4o",
        model_parameters={"temperature": "0.3", "max_tokens": 1024},
        input=messages,
    )
    completion = await oai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )
    answer = completion.choices[0].message.content or ""
    prompt_tokens = completion.usage.prompt_tokens if completion.usage else 0
    completion_tokens = completion.usage.completion_tokens if completion.usage else 0
    generation.end(
        output=answer,
        usage=ModelUsage(input=prompt_tokens, output=completion_tokens, unit="TOKENS"),  # type: ignore[call-overload]
    )

    # 6. Log proof to Supabase
    proof_id = str(uuid.uuid4())
    proof_span = trace.span(name="proof_log")
    supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)
    supabase.table("proof_records").insert({
        "id": proof_id,
        "question": question,
        "answer": answer,
        "citations": [c.model_dump(mode="json") for c in citations],
        "model_used": "gpt-4o",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "visitor_ip_hash": ip_hash,
    }).execute()
    proof_span.end(output={"proof_id": proof_id})

    trace.update(output={
        "proof_id": proof_id,
        "answer_length": len(answer),
        "citations_count": len(citations),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    })
    lf.flush()  # blocks until queue is drained

    return ChatResponse(
        answer=answer,
        proof_id=proof_id,
        citations=citations,
        model_used="gpt-4o",
        created_at=datetime.now(timezone.utc),
    )


async def stream_rag_chat(
    question: str,
    visitor_ip: str = "",
    source_types: list[str] | None = None,
    repo_filter: list[str] | None = None,
    conversation_history: list[ConversationMessage] | None = None,
) -> AsyncIterator[str]:
    """
    Yields SSE-compatible strings:
      - data chunks while streaming the answer
      - a final JSON line with proof_id and citations
    """
    settings = get_settings()
    oai = AsyncOpenAI(api_key=settings.openai_api_key)
    lf = get_langfuse()
    ip_hash = hashlib.sha256(visitor_ip.encode()).hexdigest()[:16] if visitor_ip else None

    trace = lf.trace(
        name="stream_rag_chat",
        input={"question": question},
        metadata={"source_types": source_types, "repo_filter": repo_filter, "visitor_ip_hash": ip_hash},
    )

    # 1. Embed
    embed_span = trace.span(name="embed_question", input={"question": question})
    q_embedding = await get_query_embedding(question)
    embed_span.end(output={"embedding_dim": len(q_embedding)})

    # ── Semantic cache check ──────────────────────────────────────────────────
    cache_span = trace.span(name="cache_check")
    cached = check_cache(q_embedding, source_types=source_types, repo_filter=repo_filter)
    cache_span.end(output={"hit": cached is not None})

    if cached:
        trace.update(output={"from_cache": True, "proof_id": cached.get("proof_id", "")})
        lf.flush()  # blocks until queue is drained
        cached_answer: str = cached.get("answer", "")
        chunk_size = 40
        for i in range(0, len(cached_answer), chunk_size):
            yield f"data: {json.dumps(cached_answer[i:i + chunk_size])}\n\n"
        yield f"data: [DONE]\n\n"
        yield (
            f"event: proof\ndata: {json.dumps({'proof_id': cached.get('proof_id', ''), 'citations': cached.get('citations', []), 'suggestions': cached.get('suggestions', []), 'from_cache': True})}\n\n"
        )
        return
    # ─────────────────────────────────────────────────────────────────────────

    # 2. Hybrid retrieval
    retrieval_span = trace.span(
        name="hybrid_retrieval",
        input={"candidate_count": 20, "cutoff": settings.similarity_cutoff},
    )
    candidates = hybrid_search(
        query=question,
        query_embedding=q_embedding,
        candidate_count=20,
        similarity_cutoff=settings.similarity_cutoff,
        source_types=source_types,
        repo_filter=repo_filter,
    )
    raw_chunks = candidates[:settings.top_k_retrieval]
    retrieval_span.end(output={
        "candidates": len(candidates),
        "chunks_used": len(raw_chunks),
        "top_scores": [round(c.get("similarity", 0), 3) for c in raw_chunks[:3]],
    })

    citations: list[CitationChunk] = [
        CitationChunk(
            chunk_id=c["id"],
            source_type=c["source_type"],
            source_url=c.get("source_url"),
            source_title=c.get("source_title", ""),
            chunk_text=c["chunk_text"],
            similarity_score=round(c.get("similarity", 0.0), 4),
            date_indexed=c.get("date_indexed"),
        )
        for c in raw_chunks
    ]

    context_block = "\n\n".join(
        f"[SOURCE {i + 1}] ({cit.source_type} — {cit.source_title})\n{cit.chunk_text}"
        for i, cit in enumerate(citations)
    )

    messages = _build_messages(
        SYSTEM_PROMPT.format(owner_name=settings.app_owner_name),
        context_block, question, conversation_history,
    )

    # 3. Stream GPT-4o
    generation = trace.generation(
        name="gpt-4o-stream",
        model="gpt-4o",
        model_parameters={"temperature": "0.3", "max_tokens": 1024},
        input=messages,
    )

    full_answer = ""
    prompt_tokens = 0
    completion_tokens = 0

    stream = await oai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
        stream=True,
        stream_options={"include_usage": True},
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            full_answer += delta
            yield f"data: {json.dumps(delta)}\n\n"
        if chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens

    generation.end(
        output=full_answer,
        usage=ModelUsage(input=prompt_tokens, output=completion_tokens, unit="TOKENS"),  # type: ignore[call-overload]
    )

    # 4. Log proof
    proof_id = str(uuid.uuid4())
    proof_span = trace.span(name="proof_log")
    supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)
    supabase.table("proof_records").insert({
        "id": proof_id,
        "question": question,
        "answer": full_answer,
        "citations": [c.model_dump(mode="json") for c in citations],
        "model_used": "gpt-4o",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "visitor_ip_hash": ip_hash,
    }).execute()
    proof_span.end(output={"proof_id": proof_id})

    # 5. Populate semantic cache
    cache_write_span = trace.span(name="cache_populate")
    populate_cache(
        question=question,
        question_embedding=q_embedding,
        answer=full_answer,
        citations=[c.model_dump(mode="json") for c in citations],
        proof_id=proof_id,
        suggestions=[],          # backfilled once suggestions are ready
        source_types=source_types,
        repo_filter=repo_filter,
    )
    cache_write_span.end()

    # 6. Follow-up suggestions (gpt-4o-mini)
    suggestions: list[str] = []
    suggestions_gen = trace.generation(
        name="suggestions",
        model="gpt-4o-mini",
        model_parameters={"temperature": "0.7", "max_tokens": 150},
        input=[
            {"role": "system", "content": "Generate 3 follow-up questions as a JSON array. No other text."},
            {"role": "user", "content": f"Question: {question}\n\nAnswer: {full_answer[:800]}"},
        ],
    )
    try:
        suggestion_resp = await oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You generate follow-up questions a recruiter or visitor might ask after reading a portfolio answer. "
                        "Return exactly 3 short, specific questions as a JSON array of strings. No other text."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nAnswer: {full_answer[:800]}",
                },
            ],
            temperature=0.7,
            max_tokens=150,
        )
        raw = (suggestion_resp.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        suggestions = json.loads(raw)
        if not isinstance(suggestions, list):
            suggestions = []
        suggestions = [s for s in suggestions if isinstance(s, str)][:3]

        s_usage = suggestion_resp.usage
        suggestions_gen.end(
            output=suggestions,
            usage=ModelUsage(  # type: ignore[call-overload]
                input=s_usage.prompt_tokens if s_usage else 0,
                output=s_usage.completion_tokens if s_usage else 0,
                unit="TOKENS",
            ),
        )
    except Exception:
        suggestions = []
        suggestions_gen.end(output=[], metadata={"error": "suggestions_failed"})

    # Backfill suggestions into the cache entry
    if suggestions:
        try:
            supabase.table("semantic_cache") \
                .update({"suggestions": suggestions}) \
                .eq("proof_id", proof_id) \
                .execute()
        except Exception:
            pass

    trace.update(output={
        "from_cache": False,
        "proof_id": proof_id,
        "answer_length": len(full_answer),
        "citations_count": len(citations),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "suggestions_count": len(suggestions),
    })
    lf.flush()  # blocks until queue is drained

    yield f"data: [DONE]\n\n"
    yield f"event: proof\ndata: {json.dumps({'proof_id': proof_id, 'citations': [c.model_dump(mode='json') for c in citations], 'suggestions': suggestions, 'from_cache': False})}\n\n"
