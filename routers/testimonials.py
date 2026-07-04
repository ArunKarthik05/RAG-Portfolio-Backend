from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timezone
import uuid

from supabase import create_client
from config import get_settings

router = APIRouter(prefix="/testimonials", tags=["testimonials"])


class TestimonialCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    role: Optional[str] = Field(None, max_length=100)
    company: Optional[str] = Field(None, max_length=100)
    message: str = Field(..., min_length=10, max_length=1000)
    tags: list[str] = Field(default_factory=list, max_length=3)
    user_id: Optional[str] = None  # email if signed in


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


def _client():
    s = get_settings()
    return create_client(s.supabase_url, s.supabase_service_role_key)


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
    # Verify ownership
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
async def delete_testimonial(testimonial_id: str, user_id: str):
    client = _client()
    existing = client.table("testimonials").select("user_id").eq("id", testimonial_id).execute()
    if not existing.data:
        raise HTTPException(status_code=404, detail="Testimonial not found")
    if existing.data[0].get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not authorised to delete this testimonial")
    client.table("testimonials").delete().eq("id", testimonial_id).execute()


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
