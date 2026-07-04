"""
Semantic caching layer for the RAG pipeline.

Flow:
  1. check_cache()  — embed the question, find a cosine-similar cached entry.
  2. If hit: verify data hasn't changed since the entry was cached (_data_fresh).
  3. If still valid: return the cached answer immediately (no LLM call).
  4. After a real LLM call: populate_cache() stores the result for future hits.

Cache invalidation:
  - TTL: entries older than CACHE_TTL_HOURS are ignored.
  - Data freshness: if any document_chunk was indexed AFTER the cache entry was
    created, the entry is considered stale and a fresh LLM call is made.
"""
import logging
from datetime import datetime, timezone

from supabase import create_client
from config import get_settings

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.70   # cosine similarity to treat as "same question"
CACHE_TTL_HOURS = 24          # ignore entries older than this regardless


def _client():
    s = get_settings()
    return create_client(s.supabase_url, s.supabase_service_role_key)


def check_cache(
    question_embedding: list[float],
    source_types: list[str] | None = None,
    repo_filter: list[str] | None = None,
) -> dict | None:
    """
    Returns the best matching cache entry if:
      - cosine similarity >= SIMILARITY_THRESHOLD
      - entry is within CACHE_TTL_HOURS
      - no document_chunks were indexed after the entry was created

    Returns None if no valid cache hit (caller should do a full LLM call).
    """
    try:
        client = _client()
        result = client.rpc(
            "match_semantic_cache",
            {
                "query_embedding": question_embedding,
                "similarity_threshold": SIMILARITY_THRESHOLD,
                "match_count": 1,
                "source_type_filter": source_types,
                "repo_filter_val": repo_filter,
            },
        ).execute()

        rows = result.data or []
        if not rows:
            return None

        entry = rows[0]

        # --- TTL check ---
        cached_at_str = entry.get("created_at", "")
        if not cached_at_str:
            return None
        cached_at = datetime.fromisoformat(cached_at_str.replace("Z", "+00:00"))
        age_hours = (datetime.now(timezone.utc) - cached_at).total_seconds() / 3600
        if age_hours > CACHE_TTL_HOURS:
            logger.info("Semantic cache MISS — entry expired (%.1fh old)", age_hours)
            return None

        # --- Data freshness check ---
        if _data_updated_since(cached_at, source_types):
            logger.info("Semantic cache MISS — data sources updated after %s", cached_at_str)
            return None

        similarity = entry.get("similarity", 0.0)
        logger.info("Semantic cache HIT (similarity=%.3f, age=%.1fh)", similarity, age_hours)

        # Increment hit counter asynchronously (best-effort)
        try:
            client.rpc("increment_cache_hit", {"entry_id": entry["id"]}).execute()
        except Exception:
            pass

        return entry

    except Exception as exc:
        logger.warning("Cache lookup failed, skipping: %s", exc)
        return None


def _data_updated_since(since: datetime, source_types: list[str] | None) -> bool:
    """True if any document_chunk was indexed after `since`."""
    try:
        client = _client()
        query = (
            client.table("document_chunks")
            .select("date_indexed")
            .order("date_indexed", desc=True)
            .limit(1)
        )
        if source_types:
            query = query.in_("source_type", source_types)
        rows = (query.execute().data or [])
        if not rows:
            return False
        latest_str = rows[0].get("date_indexed", "")
        if not latest_str:
            return False
        latest = datetime.fromisoformat(latest_str.replace("Z", "+00:00"))
        return latest > since
    except Exception as exc:
        logger.warning("Freshness check failed (assuming data is fresh): %s", exc)
        return False   # on error, serve from cache rather than make extra LLM calls


def populate_cache(
    question: str,
    question_embedding: list[float],
    answer: str,
    citations: list[dict],
    proof_id: str,
    suggestions: list[str],
    source_types: list[str] | None = None,
    repo_filter: list[str] | None = None,
) -> None:
    """Store a completed Q&A result in the semantic cache."""
    try:
        client = _client()
        client.table("semantic_cache").insert(
            {
                "question": question,
                "question_embedding": question_embedding,
                "answer": answer,
                "citations": citations,
                "proof_id": proof_id,
                "suggestions": suggestions,
                "source_types": source_types,
                "repo_filter": repo_filter,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        ).execute()
        logger.info("Semantic cache populated for: %.80s", question)
    except Exception as exc:
        logger.warning("Cache population failed: %s", exc)
