"""
Google Calendar connector — semantic chunking strategy:
  - 1 chunk per notable past event/talk/conference
  - 1 chunk for overall availability summary
Each chunk is a self-contained, retrievable fact unit.
"""
from datetime import datetime, timezone, timedelta
from google.oauth2 import service_account
from googleapiclient.discovery import build
from config import get_settings
from connectors.embedder import embed_and_store

SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]
CALENDAR_URL = "https://calendar.google.com"


def _get_service():
    settings = get_settings()
    creds = service_account.Credentials.from_service_account_file(
        settings.google_calendar_credentials_json, scopes=SCOPES
    )
    return build("calendar", "v3", credentials=creds)


def _make_chunk(text: str, source_title: str, metadata: dict) -> dict:
    return {
        "source_type": "calendar",
        "source_url": CALENDAR_URL,
        "source_title": source_title,
        "chunk_text": text.strip(),
        "metadata": metadata,
    }


async def ingest_calendar() -> tuple[int, int]:
    settings = get_settings()
    service = _get_service()
    cal_id = settings.google_calendar_id
    now = datetime.now(timezone.utc)

    all_chunks = []

    # ── Past events (1 chunk per notable event) ──────────────────────────
    past_result = (
        service.events()
        .list(
            calendarId=cal_id,
            timeMin=(now - timedelta(days=365)).isoformat(),
            timeMax=now.isoformat(),
            maxResults=100,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )

    for event in past_result.get("items", []):
        title = event.get("summary", "").strip()
        if not title:
            continue
        start = event.get("start", {}).get("dateTime", event.get("start", {}).get("date", ""))
        date_str = start[:10] if start else "unknown date"
        location = event.get("location", "").strip()
        description = event.get("description", "").strip()

        text = f"Event: {title} on {date_str}"
        if location:
            text += f" at {location}"
        if description:
            text += f". {description[:300]}"

        all_chunks.append(_make_chunk(
            text, f"Calendar — {title}",
            {"section": "event", "date": date_str, "title": title},
        ))

    # ── Availability summary (1 chunk) ───────────────────────────────────
    future_result = (
        service.events()
        .list(
            calendarId=cal_id,
            timeMin=now.isoformat(),
            timeMax=(now + timedelta(days=90)).isoformat(),
            maxResults=50,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )
    busy_days = {
        e.get("start", {}).get("dateTime", e.get("start", {}).get("date", ""))[:10]
        for e in future_result.get("items", [])
    }
    busy_days.discard("")

    all_chunks.append(_make_chunk(
        f"{settings.app_owner_name} has {len(busy_days)} busy days in the next 90 days. "
        f"To schedule a meeting or interview, reach out at {settings.app_owner_email}.",
        "Calendar — Availability",
        {"section": "availability"},
    ))

    return await embed_and_store(all_chunks)
