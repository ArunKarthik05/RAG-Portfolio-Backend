from fastapi import APIRouter
from rag.store import get_github_repos, get_source_stats

router = APIRouter(prefix="/sources", tags=["sources"])


@router.get("/github/repos")
async def github_repos():
    """Returns list of distinct GitHub repos that have been ingested."""
    return get_github_repos()


@router.get("/stats")
async def source_stats():
    """Returns chunk counts and last sync time per source type."""
    return get_source_stats()
