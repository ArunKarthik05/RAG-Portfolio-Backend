"""
GitHub connector — semantic chunking strategy:
  - 1 chunk per repo: AI-generated summary (GPT-4o-mini)
  - 1 chunk per repo: README excerpt (first 400 words)
Commit history excluded to keep chunks focused on capabilities.
"""
from github import Github, GithubException
from openai import AsyncOpenAI
from config import get_settings
from connectors.embedder import embed_and_store


def _make_chunk(text: str, source_url: str, source_title: str, metadata: dict) -> dict:
    return {
        "source_type": "github",
        "source_url": source_url,
        "source_title": source_title,
        "chunk_text": text.strip(),
        "metadata": metadata,
    }


async def _generate_repo_summary(
    oai: AsyncOpenAI,
    short_name: str,
    description: str,
    language: str,
    topics: str,
    stars: int,
    readme_excerpt: str,
) -> str:
    """GPT-4o-mini: concise recruiter-friendly repo summary."""
    prompt = (
        f"Write a concise 3-4 sentence summary of this GitHub repository for a technical recruiter. "
        f"Focus on what it does, the tech stack, and why it's interesting. Be factual and professional.\n\n"
        f"Repository: {short_name}\n"
        f"Description: {description}\n"
        f"Language: {language}\n"
        f"Topics: {topics}\n"
        f"Stars: {stars}\n"
        f"README excerpt:\n{readme_excerpt[:800]}\n\n"
        f"Write only the summary, no headers or labels."
    )
    completion = await oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200,
    )
    return (completion.choices[0].message.content or "").strip()


async def ingest_github(repo_names: list[str] | None = None) -> tuple[int, int]:
    """
    Ingest GitHub repositories into the vector store.
    repo_names: short names to ingest (e.g. ["my-project"]). None = all public non-fork repos.
    """
    settings = get_settings()
    g = Github(settings.github_token)
    oai = AsyncOpenAI(api_key=settings.openai_api_key)
    user = g.get_user(settings.github_username)
    all_chunks = []

    for repo in user.get_repos(type="public", sort="updated"):
        if repo.fork:
            continue
        if repo_names is not None and repo.name not in repo_names:
            continue

        repo_url = repo.html_url
        repo_full_name = repo.full_name
        short_name = repo.name
        description = repo.description or "No description"
        topics = ", ".join(repo.get_topics()) or "None"
        stars = repo.stargazers_count
        language = repo.language or "Unknown"
        base_meta = {
            "repo": short_name,
            "stars": stars,
            "language": language,
            "topics": topics,
        }

        # ── README excerpt ────────────────────────────────────────────
        readme_excerpt = ""
        try:
            readme = repo.get_readme()
            readme_raw = readme.decoded_content.decode("utf-8", errors="ignore")
            readme_clean = "\n".join(
                line for line in readme_raw.splitlines()
                if not line.startswith("```") and not line.startswith("<!--")
            )
            readme_excerpt = " ".join(readme_clean.split()[:400])
        except GithubException:
            pass

        # ── Chunk 1: AI-generated summary ─────────────────────────────
        try:
            ai_summary = await _generate_repo_summary(
                oai, short_name, description, language, topics, stars, readme_excerpt
            )
            summary_text = (
                f"{short_name} ({language}) — {description}\n"
                f"Topics: {topics} | Stars: {stars}\n\n"
                f"{ai_summary}"
            )
        except Exception:
            summary_text = (
                f"{short_name} is a {language} repository by Arun Karthik.\n"
                f"Description: {description}\nTopics: {topics}\nStars: {stars}"
            )

        all_chunks.append(_make_chunk(
            summary_text, repo_url, repo_full_name,
            {**base_meta, "doc_type": "summary"},
        ))

        # ── Chunk 2: README highlights ────────────────────────────────
        if readme_excerpt.strip():
            all_chunks.append(_make_chunk(
                f"{short_name} README:\n{readme_excerpt}",
                repo_url, f"{repo_full_name} — README",
                {**base_meta, "doc_type": "readme"},
            ))

    return await embed_and_store(all_chunks)


async def list_github_repos() -> list[dict]:
    """Live list of all public non-fork repos from GitHub API (not from DB)."""
    settings = get_settings()
    g = Github(settings.github_token)
    user = g.get_user(settings.github_username)
    repos = []
    for repo in user.get_repos(type="public", sort="updated"):
        if repo.fork:
            continue
        repos.append({
            "name": repo.name,
            "full_name": repo.full_name,
            "description": repo.description or "",
            "language": repo.language or "Unknown",
            "stars": repo.stargazers_count,
            "topics": repo.get_topics(),
            "url": repo.html_url,
            "updated_at": repo.updated_at.isoformat() if repo.updated_at else None,
        })
    return repos
