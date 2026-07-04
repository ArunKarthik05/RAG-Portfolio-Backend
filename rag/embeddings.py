"""
OpenAI embedding model.
Model: text-embedding-3-small — 512-dim (truncated), fast, cheap.
Uses the same OpenAI API key as the chat model — no extra credentials needed.
"""
from openai import AsyncOpenAI
from config import get_settings

_client: AsyncOpenAI | None = None
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 512 


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=get_settings().openai_api_key)
    return _client


async def get_query_embedding(text: str) -> list[float]:
    res = await _get_client().embeddings.create(
        model=EMBED_MODEL,
        input=text,
        dimensions=EMBED_DIMS,
    )
    return res.data[0].embedding


async def get_text_embeddings(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    res = await _get_client().embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        dimensions=EMBED_DIMS,
    )
    return [d.embedding for d in res.data]
