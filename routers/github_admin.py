"""
GitHub Admin router — list, selectively ingest, and delete GitHub repo chunks.
All routes require x-admin-key header.
"""
from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel
from config import get_settings
from connectors.github import ingest_github, list_github_repos
from rag.store import get_github_repos, delete_repo_chunks

router = APIRouter(prefix="/admin/github", tags=["github-admin"])


def require_admin(x_admin_key: str = Header(...)):
    if x_admin_key != get_settings().admin_api_key:
        raise HTTPException(status_code=403, detail="Forbidden")


class IngestReposRequest(BaseModel):
    repo_names: list[str]  # short repo names to ingest e.g. ["my-project"]


@router.get("/repos", dependencies=[Depends(require_admin)])
async def all_github_repos():
    """
    Returns all public non-fork repos from GitHub API + indexed status from DB.
    Merges the two so the UI can show which repos are already indexed.
    """
    # Live repos from GitHub API
    live_repos = await list_github_repos()

    # Indexed repos from DB
    indexed = {r["repo_name"]: r for r in get_github_repos()}

    # Merge indexed status into live list
    for repo in live_repos:
        db_entry = indexed.get(repo["name"])
        repo["indexed"] = db_entry is not None
        repo["last_indexed"] = db_entry["last_indexed"] if db_entry else None

    return live_repos


@router.post("/ingest", dependencies=[Depends(require_admin)])
async def ingest_selected_repos(body: IngestReposRequest):
    """Ingest (or re-index) specific repos by short name."""
    if not body.repo_names:
        raise HTTPException(status_code=400, detail="repo_names must not be empty")
    added, updated = await ingest_github(repo_names=body.repo_names)
    return {"chunks_added": added, "chunks_updated": updated, "repos": body.repo_names, "status": "ok"}


@router.delete("/repos/{repo_name}", dependencies=[Depends(require_admin)])
async def delete_repo(repo_name: str):
    """Delete all chunks for a given GitHub repo from the vector store."""
    deleted = delete_repo_chunks(repo_name)
    return {"repo_name": repo_name, "chunks_deleted": deleted, "status": "ok"}
