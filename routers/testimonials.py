from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timezone
import uuid
import hmac
import hashlib
import base64
import time

from supabase import create_client
from config import get_settings

router = APIRouter(prefix="/testimonials", tags=["testimonials"])


# ── HMAC helpers ───────────────────────────────────────────────────────────────

def _verify_admin_sig(payload: str, sig: str, ts: int, secret: str) -> bool:
    """
    Verify an HMAC-SHA256 admin signature.

    The frontend signs f"{payload}:{ts}" with the admin secret as key.
    We allow a ±5-minute clock skew window to prevent replay attacks.
    The raw secret is never transmitted — only the signature.
    """
    if not secret:
        return False
    if abs(time.time() - ts) > 300:  # 5-minute window
        return False
    msg = f"{payload}:{ts}".encode()
    expected = hmac.new(secret.encode(), msg, hashlib.sha256).digest()
    try:
        received = base64.b64decode(sig)
        return hmac.compare_digest(expected, received)
    except Exception:
        return False


# ── Models ─────────────────────────────────────────────────────────────────────

class TestimonialCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    role: Optional[str] = Field(None, max_length=100)
    company: Optional[str] = Field(None, max_length=100)
    message: str = Field(..., min_length=10, max_length=1000)
    tags: list[str] = Field(default_factory=list, max_length=3)
    user_id: Optional[str] = None


class TestimonialUpdate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    role: Optional[str] = Field(None, max_length=100)
    company: Optional[str] = Field(None, max_length=100)
    message: str = Field(..., min_length=10, max_length=1000)
    tags: list[str] = Field(default_factory=list, max_length=3)
    user_id: str  # required to verify ownership


class TestimonialOut(BaseModel):
    id: str
    name: str
    role: Optional[str]
    company: Optional[str]
    message: str
    tags: list[str]
    user_id: Optional[str]
    is_guest: bool
    created_at: str


class BulkDeleteRequest(BaseModel):
    ids: list[str]
    sig: str   # HMAC-SHA256 of "bulk:{sorted_ids}:{ts}"
    ts: int    # Unix timestamp (seconds)


# ── DB client ──────────────────────────────────────────────────────────────────

def _client():
    s = get_settings()
    return create_client(s.supabase_url, s.supabase_service_role_key)


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post("/", response_model=TestimonialOut, status_code=201)
async def create_testimonial(body: TestimonialCreate):
    client = _client()
    tags = [t.strip() for t in body.tags if t.strip()][:3]
    record = {
        "id": str(uuid.uuid4()),
        "name": body.name.strip(),
        "role": body.role.strip() if body.role else None,
        "company": body.company.strip() if body.company else None,
        "message": body.message.strip(),
        "tags": tags,
        "user_id": body.user_id,
        "is_guest": body.user_id is None,
        "approved": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    result = client.table("testimonials").insert(record).execute()
    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to save testimonial")
    return _to_out(result.data[0])


@router.get("/", response_model=list[TestimonialOut])
async def list_testimonials():
    client = _client()
    result = (
        client.table("testimonials")
        .select("id,name,role,company,message,tags,user_id,is_guest,created_at")
        .eq("approved", True)
        .order("created_at", desc=True)
        .execute()
    )
    return result.data or []


@router.put("/{testimonial_id}", response_model=TestimonialOut)
async def update_testimonial(testimonial_id: str, body: TestimonialUpdate):
    client = _client()
    existing = client.table("testimonials").select("user_id").eq("id", testimonial_id).execute()
    if not existing.data:
        raise HTTPException(status_code=404, detail="Testimonial not found")
    if existing.data[0].get("user_id") != body.user_id:
        raise HTTPException(status_code=403, detail="Not authorised to edit this testimonial")

    tags = [t.strip() for t in body.tags if t.strip()][:3]
    result = (
        client.table("testimonials")
        .update({
            "name": body.name.strip(),
            "role": body.role.strip() if body.role else None,
            "company": body.company.strip() if body.company else None,
            "message": body.message.strip(),
            "tags": tags,
        })
        .eq("id", testimonial_id)
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=500, detail="Update failed")
    return _to_out(result.data[0])


@router.delete("/{testimonial_id}", status_code=204)
async def delete_testimonial(
    testimonial_id: str,
    # Regular user deletion
    user_id: Optional[str] = Query(default=None),
    # Admin deletion via HMAC — raw secret is NEVER sent
    sig: Optional[str] = Query(default=None),
    ts: Optional[int] = Query(default=None),
):
    client = _client()
    settings = get_settings()

    # Determine if this is a verified admin request
    is_admin = (
        sig is not None
        and ts is not None
        and _verify_admin_sig(testimonial_id, sig, ts, settings.admin_password)
    )

    existing = client.table("testimonials").select("user_id").eq("id", testimonial_id).execute()
    if not existing.data:
        raise HTTPException(status_code=404, detail="Testimonial not found")

    if not is_admin:
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id required")
        if existing.data[0].get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="Not authorised to delete this testimonial")

    client.table("testimonials").delete().eq("id", testimonial_id).execute()


@router.post("/bulk-delete", status_code=200)
async def bulk_delete_testimonials(body: BulkDeleteRequest):
    """
    Bulk delete testimonials. Admin-only via HMAC signature.
    Payload signed by frontend: "bulk:{sorted comma-joined ids}:{ts}"
    """
    settings = get_settings()
    sorted_ids = ",".join(sorted(body.ids))
    payload = f"bulk:{sorted_ids}"
    if not _verify_admin_sig(payload, body.sig, body.ts, settings.admin_password):
        raise HTTPException(status_code=403, detail="Invalid or expired admin signature")
    if not body.ids:
        return {"deleted": 0}
    client = _client()
    client.table("testimonials").delete().in_("id", body.ids).execute()
    return {"deleted": len(body.ids)}


# ── Serialiser ─────────────────────────────────────────────────────────────────

def _to_out(row: dict) -> TestimonialOut:
    return TestimonialOut(
        id=row["id"],
        name=row["name"],
        role=row.get("role"),
        company=row.get("company"),
        message=row["message"],
        tags=row.get("tags") or [],
        user_id=row.get("user_id"),
        is_guest=row.get("is_guest", True),
        created_at=row["created_at"],
    )
