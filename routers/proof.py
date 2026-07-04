from fastapi import APIRouter, HTTPException
from supabase import create_client
from config import get_settings
from models.schemas import ProofRecord

router = APIRouter(prefix="/proof", tags=["proof"])


@router.get("/{proof_id}", response_model=ProofRecord)
async def get_proof(proof_id: str):
    """
    Fetch an immutable proof record by ID.
    Exposed publicly so visitors can verify any answer.
    """
    settings = get_settings()
    client = create_client(settings.supabase_url, settings.supabase_service_role_key)
    result = (
        client.table("proof_records")
        .select("*")
        .eq("id", proof_id)
        .single()
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Proof record not found")
    return ProofRecord(**result.data)
