from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # API Keys
    openai_api_key: str
    admin_api_key: str  # secret key to protect /ingest and /admin endpoints
    admin_password: str = ""  # password for the frontend admin panel
    internal_api_key: str = ""  # shared secret between Next.js server and this backend

    # Supabase
    supabase_url: str
    supabase_service_role_key: str
    supabase_db_url: str  # postgres://... (for pgvector direct connection)

    # GitHub connector
    github_token: str
    github_username: str

    # Google Calendar connector
    google_calendar_credentials_json: str  # path to service account JSON file
    google_calendar_id: str = "primary"

    # Observability (optional)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # App settings
    app_owner_name: str = "Arun Karthik"
    app_owner_email: str = "arunkarthik.k@zohocorp.com"
    # Chunking — used only for custom file uploads (PDF, DOCX, TXT).
    # Structured sources (GitHub, LinkedIn, Calendar) use semantic chunking
    # in their connectors directly — one chunk per logical unit.
    chunk_size: int = 120      # ~3-4 sentences, one complete thought
    chunk_overlap: int = 20    # minimal bleed-over for sentence continuity
    top_k_retrieval: int = 6   # more chunks needed since each is smaller
    similarity_cutoff: float = 0.5

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
