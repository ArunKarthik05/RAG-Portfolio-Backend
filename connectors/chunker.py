"""
Simple text chunker with overlap.
Chunks are stored with metadata so retrieval has full provenance.
"""
from config import get_settings


def chunk_text(
    text: str,
    source_type: str,
    source_url: str,
    source_title: str,
    extra_metadata: dict | None = None,
) -> list[dict]:
    """
    Split text into overlapping chunks and return list of chunk dicts
    ready for embedding + upsert.
    """
    settings = get_settings()
    size = settings.chunk_size
    overlap = settings.chunk_overlap

    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words).strip()
        if len(chunk_text_str) > 20:  # skip tiny fragments
            chunks.append({
                "source_type": source_type,
                "source_url": source_url,
                "source_title": source_title,
                "chunk_text": chunk_text_str,
                "metadata": extra_metadata or {},
            })
        start += size - overlap
    return chunks
