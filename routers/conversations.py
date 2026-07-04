"""
Conversations router — stores and retrieves chat history for logged-in users.

Security model:
  All endpoints require a shared x-internal-key header that only the
  Next.js server knows. The client never calls these routes directly.

  The authenticated user's identity is passed via x-user-id, which is
  set by the Next.js proxy AFTER it has verified the NextAuth session.
  Because the header is only trusted when paired with x-internal-key,
  clients cannot spoof their identity.

  Ownership is enforced on every individual-conversation endpoint:
  the conversation's stored user_id must match x-user-id.
"""
from fastapi import APIRouter, HTTPException, Header, Depends
from typing import Optional
from supabase import create_client
from config import get_settings
from models.schemas import ConversationCreate, ConversationOut, MessageOut

router = APIRouter(prefix="/conversations", tags=["conversations"])


# ── Auth dependency ───────────────────────────────────────────────

def require_internal(x_internal_key: str = Header(...)):
    """Reject any request not originating from the Next.js server."""
    settings = get_settings()
    key = settings.internal_api_key
    if not key or x_internal_key != key:
        raise HTTPException(status_code=403, detail="Forbidden")


def get_caller_user_id(x_user_id: str = Header(...)) -> str:
    """Extract the verified user identity injected by the Next.js proxy."""
    if not x_user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return x_user_id


# ── DB client ─────────────────────────────────────────────────────

def _client():
    s = get_settings()
    return create_client(s.supabase_url, s.supabase_service_role_key)


def _assert_owner(client, conversation_id: str, user_id: str):
    """Raise 403 if the conversation doesn't belong to user_id."""
    result = (
        client.table("conversations")
        .select("user_id")
        .eq("id", conversation_id)
        .single()
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if result.data.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")


# ── Conversations CRUD ────────────────────────────────────────────

@router.post("/", response_model=ConversationOut,
             dependencies=[Depends(require_internal)])
async def create_conversation(
    body: ConversationCreate,
    user_id: str = Depends(get_caller_user_id),
):
    """Create a conversation. user_id is taken from the verified session header."""
    client = _client()
    result = (
        client.table("conversations")
        .insert({
            "user_id": user_id,          # always the verified identity
            "is_guest": body.is_guest,
            "title": body.title,
        })
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create conversation")
    return ConversationOut(**result.data[0])


@router.get("/", response_model=list[ConversationOut],
            dependencies=[Depends(require_internal)])
async def list_conversations(user_id: str = Depends(get_caller_user_id)):
    """List conversations for the authenticated user only."""
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


@router.get("/{conversation_id}", response_model=ConversationOut,
            dependencies=[Depends(require_internal)])
async def get_conversation(
    conversation_id: str,
    user_id: str = Depends(get_caller_user_id),
):
    client = _client()
    _assert_owner(client, conversation_id, user_id)
    result = (
        client.table("conversations")
        .select("*")
        .eq("id", conversation_id)
        .single()
        .execute()
    )
    return ConversationOut(**result.data)


@router.delete("/{conversation_id}",
               dependencies=[Depends(require_internal)])
async def delete_conversation(
    conversation_id: str,
    user_id: str = Depends(get_caller_user_id),
):
    client = _client()
    _assert_owner(client, conversation_id, user_id)
    client.table("conversations").delete().eq("id", conversation_id).execute()
    return {"ok": True}


# ── Messages ──────────────────────────────────────────────────────

@router.get("/{conversation_id}/messages", response_model=list[MessageOut],
            dependencies=[Depends(require_internal)])
async def get_messages(
    conversation_id: str,
    user_id: str = Depends(get_caller_user_id),
):
    client = _client()
    _assert_owner(client, conversation_id, user_id)
    result = (
        client.table("conversation_messages")
        .select("*")
        .eq("conversation_id", conversation_id)
        .order("created_at", desc=False)
        .execute()
    )
    return [MessageOut(**row) for row in result.data or []]


@router.post("/{conversation_id}/messages", response_model=MessageOut,
             dependencies=[Depends(require_internal)])
async def append_message(
    conversation_id: str,
    body: dict,
    user_id: str = Depends(get_caller_user_id),
):
    """
    Append a single message. Body: {role, content, citations?, proof_id?}
    Also bumps conversation.updated_at and auto-sets title from first user message.
    """
    client = _client()
    _assert_owner(client, conversation_id, user_id)

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

    client.table("conversations").update(
        {"updated_at": "now()"}
    ).eq("id", conversation_id).execute()

    if body["role"] == "user":
        conv = (
            client.table("conversations")
            .select("title")
            .eq("id", conversation_id)
            .single()
            .execute()
        )
        if conv.data and not conv.data.get("title"):
            title = body["content"][:60].strip()
            client.table("conversations").update({"title": title}).eq("id", conversation_id).execute()

    return MessageOut(**result.data[0])
