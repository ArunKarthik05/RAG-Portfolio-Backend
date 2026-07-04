# RAG Portfolio — Arun Karthik

An AI-powered portfolio site where visitors and HR teams can ask questions and get grounded, cited answers about your background — sourced from GitHub, LinkedIn, Google Calendar, and custom files (resume, etc.).

Every answer includes a **proof record**: the exact source chunks retrieved, with relevance scores, shown inline as expandable references.

---

## RAG Pipeline

```
User Question
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI  /chat/stream                       │
│                                                                 │
│  1. EMBED QUESTION                                              │
│     OpenAI text-embedding-3-small (512-dim)                     │
│                                                                 │
│  2. RETRIEVE CHUNKS  (Supabase pgvector RPC: match_chunks)      │
│     · Cosine similarity search, top-8 chunks                    │
│     · Optional filters:                                         │
│       – source_type_filter  e.g. ["github"]                     │
│       – repo_filter         e.g. ["my-project"]                 │
│     · Falls back to threshold=0.0 if nothing passes cutoff      │
│                                                                 │
│  3. BUILD GROUNDED PROMPT                                       │
│     System: "Answer ONLY from the provided context chunks.      │
│              Cite sources inline as [SOURCE N]."                │
│     Context: top-k chunks labelled [SOURCE 1] … [SOURCE N]     │
│     Question appended                                           │
│                                                                 │
│  4. STREAM ANSWER                                               │
│     GPT-4o, temp=0.3, max_tokens=1024, stream=True             │
│     Chunks yielded as SSE: data: <token>\n\n                   │
│                                                                 │
│  5. LOG PROOF RECORD  (Supabase proof_records table)            │
│     Immutable snapshot: question + answer + full citation JSON  │
│     Final SSE event carries proof_id + citations array          │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
Next.js (Vercel)
  · Streams tokens to UI in real-time
  · Parses final SSE event → renders References panel
  · Source panel lets user filter by connector / specific repos
```

### Chunking strategy

Each data source uses **semantic chunking** — one chunk per logical unit of information, not arbitrary word-count splits.

| Source | Chunks per item |
|--------|----------------|
| GitHub repo | 1 × AI-generated summary (GPT-4o-mini) + 1 × README excerpt (400 words) |
| LinkedIn position | 1 chunk per role |
| LinkedIn education | 1 chunk per degree |
| LinkedIn skills | 1 combined chunk |
| LinkedIn certifications | 1 chunk each |
| Google Calendar event | 1 chunk per past event |
| Google Calendar availability | 1 summary chunk |
| Custom files (PDF/DOCX/TXT) | Word-split: 120 words, 20-word overlap |

### Embedding model

`text-embedding-3-small` with `dimensions=512` — same OpenAI API key as the chat model, no separate embedding provider.

### Proof / audit system

Every answer is stored as an **immutable proof record** in `proof_records`:

```
proof_records
  id                uuid PK
  question          text
  answer            text
  citations         jsonb   ← full chunk snapshot (not FK)
  model_used        text
  prompt_tokens     int
  completion_tokens int
  created_at        timestamptz
  visitor_ip_hash   text    ← SHA-256 first 16 chars only
```

Citations are stored as a full denormalised JSON snapshot, not foreign keys, because `document_chunks` is mutable (re-ingestion updates/deletes chunks). This ensures proofs remain verifiable even after re-indexing.

---

## Architecture

```
Visitor browser
  └── Next.js (Vercel)
        ├── /                    — Chat UI (streaming SSE, source panel)
        ├── /admin               — Admin panel (GitHub CRUD, ingest)
        └── /api/...             — Server-side proxies (inject ADMIN_API_KEY)
              │
              ▼
        FastAPI (AWS Lambda + API Gateway)
              ├── /chat/stream           — RAG + streaming
              ├── /admin/github/repos    — List + indexed status
              ├── /admin/github/ingest   — Selective repo ingest
              ├── /admin/github/repos/{repo} — Delete repo chunks
              ├── /sources/github/repos  — Indexed repos (for source panel)
              ├── /ingest/{source}       — LinkedIn, Calendar, file upload
              └── /proof/{id}            — Fetch proof record
                    │
                    ├── Supabase pgvector  (document_chunks)
                    │     HNSW index, vector(512), cosine similarity
                    └── Supabase Postgres  (proof_records)

Data ingestion flow:
  GitHub API ──────────────┐
  LinkedIn JSON export ────┤
  Google Calendar API ─────┼──► FastAPI connectors
  PDF / DOCX / TXT ────────┘         │
                                      ▼
                              OpenAI embeddings
                                      │
                                      ▼
                            Supabase document_chunks
                            (upsert by deterministic UUID)
```

---

## Setup

### 1. Supabase

1. Create a project at [supabase.com](https://supabase.com)
2. Run `infra/schema.sql` in the SQL editor
3. Copy your project URL and service role key

> **If you previously ran the schema**, the `match_chunks` RPC signature changed (added optional `source_type_filter` and `repo_filter` params). Drop the old function first:
> ```sql
> DROP FUNCTION IF EXISTS match_chunks(vector, integer, double precision);
> ```
> Then re-run `infra/schema.sql`.

### 2. Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Fill in all values in .env
```

**Local dev:**
```bash
uvicorn main:app --reload --port 8000
```

**Deploy to AWS Lambda:**
```bash
# Store secrets in SSM Parameter Store
aws ssm put-parameter --name /rag-portfolio/OPENAI_API_KEY      --value "sk-..."           --type SecureString
aws ssm put-parameter --name /rag-portfolio/ADMIN_API_KEY       --value "your-secret-key"  --type SecureString
aws ssm put-parameter --name /rag-portfolio/SUPABASE_URL        --value "https://..."      --type SecureString
aws ssm put-parameter --name /rag-portfolio/SUPABASE_SERVICE_ROLE_KEY --value "eyJ..."     --type SecureString
aws ssm put-parameter --name /rag-portfolio/SUPABASE_DB_URL     --value "postgresql://..."  --type SecureString
aws ssm put-parameter --name /rag-portfolio/GITHUB_TOKEN        --value "github_pat_..."   --type SecureString
aws ssm put-parameter --name /rag-portfolio/GITHUB_USERNAME     --value "your-username"    --type SecureString

cd infra
sam build && sam deploy --guided
```

### 3. Frontend

```bash
cd frontend
npm install

cp .env.local.example .env.local
# NEXT_PUBLIC_API_URL  = your Lambda API Gateway URL
# ADMIN_API_KEY        = same value as backend (server-side only, never NEXT_PUBLIC_)
# NEXT_PUBLIC_ADMIN_PASSWORD = UI password gate for /admin
```

**Local dev:**
```bash
npm run dev
```

**Deploy to Vercel:**
```bash
npx vercel --prod
# Add all three env vars in Vercel dashboard → Settings → Environment Variables
```

### 4. Google Calendar connector

1. Create a service account in Google Cloud Console
2. Enable the Google Calendar API
3. Download the JSON key file → save as `backend/google-service-account.json`
4. Share your calendar with the service account email (read-only)
5. Set `GOOGLE_CALENDAR_CREDENTIALS_JSON=./google-service-account.json` in `.env`

### 5. Ingest data

Use the **Admin UI at `/admin`** or curl directly:

```bash
BASE=https://your-api.com
KEY=your-admin-key

# List all GitHub repos with indexed status
curl "$BASE/admin/github/repos" -H "x-admin-key: $KEY"

# Index specific repos (generates AI summary per repo)
curl -X POST "$BASE/admin/github/ingest" \
  -H "x-admin-key: $KEY" -H "Content-Type: application/json" \
  -d '{"repo_names": ["my-project", "another-repo"]}'

# Delete a repo from the index
curl -X DELETE "$BASE/admin/github/repos/my-project" -H "x-admin-key: $KEY"

# Sync Google Calendar
curl -X POST "$BASE/ingest/calendar" -H "x-admin-key: $KEY"

# Upload LinkedIn export
curl -X POST "$BASE/ingest/linkedin" \
  -H "x-admin-key: $KEY" -F "file=@Profile.csv"

# Upload resume / custom file
curl -X POST "$BASE/ingest/file" \
  -H "x-admin-key: $KEY" \
  -F "file=@resume.pdf" \
  -F "source_title=Resume · 2025"
```

---

## Admin panel (`/admin`)

| Feature | Description |
|---------|-------------|
| GitHub repo list | Shows all public GitHub repos with indexed / not-indexed status, language, stars, topics |
| Selective indexing | Checkbox-select repos → "Index N repos" generates AI summaries and embeds |
| Delete repo | Trash icon removes all chunks for a repo from the vector store |
| Re-index | Select already-indexed repos and re-run to refresh summaries |
| LinkedIn upload | Upload CSV/JSON export from LinkedIn data download |
| Calendar sync | One-click sync of past events and availability |
| Custom file upload | PDF, DOCX, MD, TXT with optional source title |

---

## Source panel (sidebar)

The chat UI has a left sidebar that lets visitors filter which data source is searched:

| Tab | Behaviour |
|-----|-----------|
| All | Searches across all ingested sources |
| GitHub | Only GitHub chunks; shows indexed repos as checkboxes for further narrowing |
| LinkedIn | Only LinkedIn chunks |
| Calendar | Only Calendar chunks |
| Files | Only custom uploaded file chunks |

The filter is sent as `source_types` and `repo_filter` in the `/chat/stream` request body and applied in the Supabase RPC query.

---

## Key URLs

| URL | Description |
|-----|-------------|
| `/` | Main chat interface |
| `/admin` | Admin panel — GitHub CRUD, connector ingest |
| `/proof/{id}` | Immutable proof record for any answer |

---

## Environment variables

### Backend (`backend/.env`)

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Used for both embeddings (text-embedding-3-small) and chat (GPT-4o) |
| `ADMIN_API_KEY` | Secret key to protect all `/ingest` and `/admin` endpoints |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Service role key (bypasses RLS for writes) |
| `SUPABASE_DB_URL` | Postgres connection string (for direct pgvector ops) |
| `GITHUB_TOKEN` | Personal access token — scopes: `repo`, `read:user` |
| `GITHUB_USERNAME` | Your GitHub username (exact, case-sensitive) |
| `GOOGLE_CALENDAR_CREDENTIALS_JSON` | Path to service account JSON file |
| `GOOGLE_CALENDAR_ID` | Calendar ID (default: `primary`) |

### Frontend (`frontend/.env.local`)

| Variable | Description |
|----------|-------------|
| `NEXT_PUBLIC_API_URL` | Backend URL (Lambda or localhost:8000) |
| `ADMIN_API_KEY` | **Server-side only** — injected by Next.js API proxy routes, never exposed to browser |
| `NEXT_PUBLIC_ADMIN_PASSWORD` | UI gate password for `/admin` (client-side check only) |

---

## Tech stack

| Layer | Tech |
|-------|------|
| Frontend | Next.js 14, Tailwind CSS, React Markdown, Lucide icons |
| Backend | FastAPI, Pydantic v2, Mangum (Lambda adapter) |
| Embeddings | OpenAI `text-embedding-3-small` (512-dim) |
| LLM | OpenAI `GPT-4o` (chat) + `GPT-4o-mini` (repo summaries) |
| Vector DB | Supabase pgvector, HNSW index, cosine similarity |
| Proof store | Supabase Postgres (`proof_records`) |
| Hosting (FE) | Vercel |
| Hosting (BE) | AWS Lambda + API Gateway (SAM) |
| Observability | LangFuse (optional) |
