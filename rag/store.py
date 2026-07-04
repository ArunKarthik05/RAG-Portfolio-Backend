"""
Supabase pgvector store — chunk upsert and similarity search.

Table: document_chunks
  id            uuid PK
  source_type   text
  source_url    text
  source_title  text
  chunk_text    text
  embedding     vector(512)
  date_indexed  timestamptz
  metadata      jsonb
"""
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from supabase import create_client, Client
from config import get_settings


def _client() -> Client:
    s = get_settings()
    return create_client(s.supabase_url, s.supabase_service_role_key)


def upsert_chunks(chunks: list[dict[str, Any]]) -> tuple[int, int]:
    """
    chunks: list of dicts with keys:
        source_type, source_url, source_title, chunk_text,
        embedding (list[float]), metadata (dict)

    Returns (added, updated) counts.
    """
    client = _client()
    added = updated = 0
    for chunk in chunks:
        chunk_id = str(uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{chunk['source_type']}::{chunk['source_url']}::{chunk['chunk_text'][:64]}"
        ))
        payload = {
            "id": chunk_id,
            "source_type": chunk["source_type"],
            "source_url": chunk.get("source_url", ""),
            "source_title": chunk.get("source_title", ""),
            "chunk_text": chunk["chunk_text"],
            "embedding": chunk["embedding"],
            "date_indexed": datetime.now(timezone.utc).isoformat(),
            "metadata": chunk.get("metadata", {}),
        }
        result = (
            client.table("document_chunks")
            .upsert(payload, on_conflict="id")
            .execute()
        )
        added += 1
    return added, 0


def similarity_search(
    query_embedding: list[float],
    top_k: int = 6,
    similarity_cutoff: float = 0.1,
    source_types: Optional[list[str]] = None,
    repo_filter: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """Dense-only vector search (kept for backward compat). Prefer hybrid_search."""
    client = _client()
    rpc_params: dict[str, Any] = {
        "query_embedding": query_embedding,
        "match_count": top_k,
        "similarity_threshold": similarity_cutoff,
    }
    if source_types:
        rpc_params["source_type_filter"] = source_types
    if repo_filter:
        rpc_params["repo_filter"] = repo_filter

    chunks = client.rpc("match_chunks", rpc_params).execute().data or []
    if not chunks:
        fallback = {**rpc_params, "similarity_threshold": 0.0}
        chunks = client.rpc("match_chunks", fallback).execute().data or []
    return chunks


def fts_search(
    query: str,
    match_count: int = 20,
    source_types: Optional[list[str]] = None,
    repo_filter: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """Sparse full-text search via the fts_chunks Postgres RPC.
    Returns [] gracefully if the function hasn't been created in Supabase yet."""
    import logging
    client = _client()
    params: dict[str, Any] = {"query_text": query, "match_count": match_count}
    if source_types:
        params["source_type_filter"] = source_types
    if repo_filter:
        params["repo_filter"] = repo_filter
    try:
        return client.rpc("fts_chunks", params).execute().data or []
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "fts_chunks RPC unavailable, skipping sparse leg: %s", exc
        )
        return []


def _rrf_fusion(
    dense: list[dict[str, Any]],
    sparse: list[dict[str, Any]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """
    Reciprocal Rank Fusion — merges dense and sparse result lists.
    RRF score = Σ 1 / (k + rank) for each list the document appears in.
    Documents appearing in both lists float to the top.
    """
    scores: dict[str, float] = {}
    index: dict[str, dict[str, Any]] = {}

    for rank, chunk in enumerate(dense):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        index[cid] = chunk

    for rank, chunk in enumerate(sparse):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        index.setdefault(cid, chunk)

    merged = sorted(index.values(), key=lambda c: scores[c["id"]], reverse=True)
    for c in merged:
        c["rrf_score"] = round(scores[c["id"]], 6)
    return merged


def hybrid_search(
    query: str,
    query_embedding: list[float],
    candidate_count: int = 20,
    similarity_cutoff: float = 0.1,
    source_types: Optional[list[str]] = None,
    repo_filter: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """
    Hybrid retrieval: dense pgvector + sparse FTS, fused with RRF.
    Returns up to candidate_count results to feed the reranker.
    """
    client = _client()

    # ── Dense leg ────────────────────────────────────────────────
    dense_params: dict[str, Any] = {
        "query_embedding": query_embedding,
        "match_count": candidate_count,
        "similarity_threshold": similarity_cutoff,
    }
    if source_types:
        dense_params["source_type_filter"] = source_types
    if repo_filter:
        dense_params["repo_filter"] = repo_filter

    dense = client.rpc("match_chunks", dense_params).execute().data or []
    if not dense:
        dense_params["similarity_threshold"] = 0.0
        dense = client.rpc("match_chunks", dense_params).execute().data or []

    # ── Sparse leg ───────────────────────────────────────────────
    sparse = fts_search(query, match_count=candidate_count,
                        source_types=source_types, repo_filter=repo_filter)

    # ── RRF fusion ───────────────────────────────────────────────
    return _rrf_fusion(dense, sparse)


def get_github_repos() -> list[dict[str, Any]]:
    """
    Returns distinct GitHub repos that have been ingested.
    Each entry: {repo_name, source_url, last_indexed}
    """
    client = _client()
    result = (
        client.table("document_chunks")
        .select("source_url, metadata, date_indexed")
        .eq("source_type", "github")
        .execute()
    )

    seen: dict[str, dict] = {}
    for row in result.data or []:
        repo_name = (row.get("metadata") or {}).get("repo", "")
        if not repo_name:
            continue
        if repo_name not in seen or row["date_indexed"] > seen[repo_name]["last_indexed"]:
            seen[repo_name] = {
                "repo_name": repo_name,
                "source_url": row["source_url"],
                "last_indexed": row["date_indexed"],
            }

    return sorted(seen.values(), key=lambda r: r["repo_name"])


def delete_repo_chunks(repo_name: str) -> int:
    """
    Delete all document_chunks for a given GitHub repo (by metadata->>'repo').
    Returns the number of rows deleted.
    """
    client = _client()
    # Supabase doesn't support jsonb filter in .delete() directly, use RPC or raw filter
    result = (
        client.table("document_chunks")
        .delete()
        .eq("source_type", "github")
        .filter("metadata->>repo", "eq", repo_name)
        .execute()
    )
    return len(result.data or [])


def get_source_stats() -> list[dict[str, Any]]:
    client = _client()
    result = (
        client.table("document_chunks")
        .select("source_type, date_indexed")
        .execute()
    )
    stats: dict[str, dict] = {}
    for row in result.data or []:
        st = row["source_type"]
        if st not in stats:
            stats[st] = {"count": 0, "last_synced": None}
        stats[st]["count"] += 1
        d = row["date_indexed"]
        if stats[st]["last_synced"] is None or d > stats[st]["last_synced"]:
            stats[st]["last_synced"] = d
    return [{"source_type": k, **v} for k, v in stats.items()]
