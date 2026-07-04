"""
Generic file ingestor — handles PDF, DOCX, MD, TXT.
Called from /ingest/file for custom uploads (resume, blog posts, etc.)
"""
import io
from pathlib import Path
from connectors.chunker import chunk_text
from connectors.embedder import embed_and_store


def _extract_text(content: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    elif ext == ".docx":
        from docx import Document
        doc = Document(io.BytesIO(content))
        return "\n".join(para.text for para in doc.paragraphs)

    elif ext in (".md", ".txt", ".markdown"):
        return content.decode("utf-8", errors="ignore")

    else:
        # Best-effort decode
        return content.decode("utf-8", errors="ignore")


async def ingest_file_bytes(
    content: bytes,
    filename: str,
    source_title: str | None = None,
) -> tuple[int, int]:
    text = _extract_text(content, filename)
    # Use provided title, otherwise derive from filename
    title = source_title or Path(filename).stem.replace("-", " ").replace("_", " ").title()

    chunks = chunk_text(
        text,
        source_type="custom",
        source_url=f"file://{filename}",
        source_title=title,
        extra_metadata={"filename": filename},
    )
    return await embed_and_store(chunks)
