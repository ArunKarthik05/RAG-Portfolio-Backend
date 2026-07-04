"""
LinkedIn connector — generic CSV ingestion.

Upload any CSV from the LinkedIn data export. Each row is converted into a
natural-language chunk using whatever columns are present — no hardcoded
field names or type detection required.

Supported CSVs (any order, any subset):
  Profile.csv, Positions.csv, Education.csv, Skills.csv, Certifications.csv
"""
import io
import csv
from connectors.embedder import embed_and_store

LINKEDIN_BASE_URL = "https://www.linkedin.com/in/"


def _make_chunk(text: str, source_url: str, source_title: str, metadata: dict) -> dict:
    return {
        "source_type": "linkedin",
        "source_url": source_url,
        "source_title": source_title,
        "chunk_text": text.strip(),
        "metadata": metadata,
    }


def _parse_csv(raw_bytes: bytes) -> tuple[list[str], list[dict]]:
    """Return (fieldnames, rows). Strips BOM, skips empty rows."""
    text = raw_bytes.decode("utf-8-sig", errors="ignore")
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = list(reader.fieldnames or [])
    rows = [row for row in reader if any(v.strip() for v in row.values())]
    return fieldnames, rows


def _row_to_text(row: dict) -> str:
    """Convert a CSV row to a readable key: value string, skipping empty fields."""
    parts = [f"{k}: {v.strip()}" for k, v in row.items() if v.strip()]
    return "\n".join(parts)


async def ingest_linkedin_csv(raw_bytes: bytes, filename: str = "") -> tuple[int, int]:
    fieldnames, rows = _parse_csv(raw_bytes)

    if not rows:
        raise ValueError("The uploaded CSV appears to be empty.")

    # Derive a human-readable section name from the filename (e.g. "Positions")
    section = filename.rsplit(".", 1)[0].strip() if filename else "LinkedIn"
    profile_url = LINKEDIN_BASE_URL

    all_chunks: list[dict] = []

    for i, row in enumerate(rows):
        text = _row_to_text(row)
        if not text:
            continue

        # Use first non-empty value as a short title hint
        first_val = next((v.strip() for v in row.values() if v.strip()), f"Entry {i + 1}")

        all_chunks.append(_make_chunk(
            text,
            profile_url,
            f"LinkedIn — {section}: {first_val}",
            {"section": section.lower(), "row_index": i, "fields": fieldnames},
        ))

    if not all_chunks:
        raise ValueError(
            f"No usable data found in '{filename or 'uploaded CSV'}'. "
            "Check that the file is a valid LinkedIn export CSV."
        )

    return await embed_and_store(all_chunks)
