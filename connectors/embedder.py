"""
Embed a batch of chunk dicts and upsert them into Supabase.
"""
from rag.embeddings import get_text_embeddings
from rag.store import upsert_chunks


async def embed_and_store(chunks: list[dict]) -> tuple[int, int]:
    """
    Takes raw chunk dicts (without embeddings) → embeds → upserts.
    Returns (added, updated).
    """
    if not chunks:
        return 0, 0

    texts = [c["chunk_text"] for c in chunks]
    embeddings = await get_text_embeddings(texts)

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb

    return upsert_chunks(chunks)
