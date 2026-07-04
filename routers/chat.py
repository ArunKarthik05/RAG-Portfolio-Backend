from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from models.schemas import ChatRequest, ChatResponse
from rag.pipeline import stream_rag_chat, run_rag_chat

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/stream")
async def chat_stream(body: ChatRequest, request: Request):
    """
    Streams the answer as Server-Sent Events.
    Final event: `event: proof` with proof_id and citations JSON.
    """
    visitor_ip = request.client.host if request.client else ""
    return StreamingResponse(
        stream_rag_chat(
            body.question,
            visitor_ip,
            source_types=body.source_types,
            repo_filter=body.repo_filter,
            conversation_history=body.conversation_history,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request):
    """Non-streaming version — returns full response at once."""
    visitor_ip = request.client.host if request.client else ""
    return await run_rag_chat(
        body.question,
        visitor_ip,
        source_types=body.source_types,
        repo_filter=body.repo_filter,
        conversation_history=body.conversation_history,
    )
