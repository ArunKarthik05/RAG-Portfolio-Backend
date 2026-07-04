"""
Ingest router — protected by admin API key.
Triggers re-sync of each data source and custom file uploads.
"""
import io
from fastapi import APIRouter, Depends, HTTPException, Header, UploadFile, File, Form
from models.schemas import IngestResponse
from connectors.github import ingest_github
from connectors.linkedin import ingest_linkedin_csv
from connectors.google_calendar import ingest_calendar
from connectors.file_ingestor import ingest_file_bytes
from config import get_settings

router = APIRouter(prefix="/ingest", tags=["ingest"])


def require_admin(x_admin_key: str = Header(...)):
    if x_admin_key != get_settings().admin_api_key:
        raise HTTPException(status_code=403, detail="Forbidden")


@router.post("/github", response_model=IngestResponse, dependencies=[Depends(require_admin)])
async def ingest_github_route():
    added, updated = await ingest_github()
    return IngestResponse(source_type="github", chunks_added=added, chunks_updated=updated, status="ok")


@router.post("/linkedin", response_model=IngestResponse, dependencies=[Depends(require_admin)])
async def ingest_linkedin_route(file: UploadFile = File(...)):
    """
    Upload your LinkedIn data export.
    Accepted formats:
      - Individual CSVs: Profile.csv, Positions.csv, Education.csv, Skills.csv, Certifications.csv
    """
    content = await file.read()
    filename = file.filename or ""
    added, updated = await ingest_linkedin_csv(content, filename)
    return IngestResponse(source_type="linkedin", chunks_added=added, chunks_updated=updated, status="ok")


@router.post("/calendar", response_model=IngestResponse, dependencies=[Depends(require_admin)])
async def ingest_calendar_route():
    added, updated = await ingest_calendar()
    return IngestResponse(source_type="calendar", chunks_added=added, chunks_updated=updated, status="ok")


@router.post("/file", response_model=IngestResponse, dependencies=[Depends(require_admin)])
async def ingest_custom_file(
    file: UploadFile = File(...),
    source_title: str = Form(default=""),
):
    """
    Upload any file (PDF, DOCX, MD, TXT) and ingest it into the vector store.
    source_title overrides the auto-derived title from the filename.
    """
    content = await file.read()
    filename = file.filename or "custom"
    added, updated = await ingest_file_bytes(content, filename, source_title or None)
    return IngestResponse(source_type="custom", chunks_added=added, chunks_updated=updated, status="ok")


@router.get("/sources", dependencies=[Depends(require_admin)])
async def get_sources():
    from rag.store import get_source_stats
    return get_source_stats()
