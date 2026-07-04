"""
Conversations router — stores and retrieves chat history for logged-in users.

Endpoints are public (no admin key) but scoped by user_id supplied by the client.
The frontend sends the NextAuth session user email (or a guest UUID from localStorage).
"""
from fastapi import APIRouter, HTTPException, Query
from supabase import create_client
from config import get_settings
from models.schemas import ConversationCreate, ConversationOut, MessageOut

router = APIRouter(prefix="/conversations", tags=["conversations"])


def _client():
    s = get_settings()
    return create_client(s.supabase_url, s.supabase_service_role_key)


# ── Conversations CRUD ────────────────────────────────────────────────────────

@router.post("/", response_model=ConversationOut)
async def create_conversation(body: ConversationCreate):
    client = _client()
    result = (
        client.table("conversations")
        .insert({
            "user_id": body.user_id,
            "is_guest": body.is_guest,
            "title": body.title,
        })
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create conversation")
    return ConversationOut(**result.data[0])


@router.get("/", response_model=list[ConversationOut])
async def list_conversations(user_id: str = Query(...)):
    client = _client()
    result = (
        client.table("conversations")
        .select("*")
        .eq("user_id", user_id)
        .order("updated_at", desc=True)
        .limit(50)
        .execute()
    )
    return [ConversationOut(**row) for row in result.data or []]


@router.get("/{conversation_id}", response_model=ConversationOut)
async def get_conversation(conversation_id: str):
    client = _client()
    result = (
        client.table("conversations")
        .select("*")
        .eq("id", conversation_id)
        .single()
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationOut(**result.data)


@router.patch("/{conversation_id}/title")
async def update_title(conversation_id: str, title: str = Query(...)):
    client = _client()
    client.table("conversations").update({"title": title}).eq("id", conversation_id).execute()
    return {"ok": True}


@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: str):
    client = _client()
    client.table("conversations").delete().eq("id", conversation_id).execute()
    return {"ok": True}


# ── Messages ─────────────────────────────────────────────────────────────────

@router.get("/{conversation_id}/messages", response_model=list[MessageOut])
async def get_messages(conversation_id: str):
    client = _client()
    result = (
        client.table("conversation_messages")
        .select("*")
        .eq("conversation_id", conversation_id)
        .order("created_at", desc=False)
        .execute()
    )
    return [MessageOut(**row) for row in result.data or []]


@router.post("/{conversation_id}/messages", response_model=MessageOut)
async def append_message(conversation_id: str, body: dict):
    """
    Append a single message. Body: {role, content, citations?, proof_id?}
    Also bumps conversation.updated_at and auto-sets title from first user message.
    """
    client = _client()

    # Insert message
    result = (
        client.table("conversation_messages")
        .insert({
            "conversation_id": conversation_id,
            "role": body["role"],
            "content": body["content"],
            "citations": body.get("citations", []),
            "proof_id": body.get("proof_id"),
        })
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to save message")

    # Bump updated_at
    client.table("conversations").update(
        {"updated_at": "now()"}
    ).eq("id", conversation_id).execute()

    # Auto-title from first user message (if conversation has no title yet)
    if body["role"] == "user":
        conv = client.table("conversations").select("title").eq("id", conversation_id).single().execute()
        if conv.data and not conv.data.get("title"):
            title = body["content"][:60].strip()
            client.table("conversations").update({"title": title}).eq("id", conversation_id).execute()

    return MessageOut(**result.data[0])
