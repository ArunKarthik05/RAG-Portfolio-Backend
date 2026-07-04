"""
Langfuse observability singleton.

Usage:
    from rag.observability import get_langfuse

    lf = get_langfuse()          # returns Langfuse client or NoOpLangfuse stub
    trace = lf.trace(name="...", input={...})
    span  = trace.span(name="...", input={...})
    span.end(output={...})
    gen   = trace.generation(name="...", model="gpt-4o", input=[...])
    gen.end(output="...", usage={"input": 100, "output": 50})
    lf.flush()                   # always call at end of request

If LANGFUSE_SECRET_KEY / LANGFUSE_PUBLIC_KEY are not set, every call is a
no-op and the pipeline runs normally without crashing.
"""
import logging
from functools import lru_cache

logger = logging.getLogger("observability")


# ── No-op stubs (used when Langfuse is not configured) ───────────────────────

class _NoOpNode:
    """Silently absorbs any span / generation / event call."""

    def span(self, **_):        return self
    def generation(self, **_):  return self
    def event(self, **_):       return self
    def score(self, **_):       return self
    def update(self, **_):      return self
    def end(self, **_):         return self


class _NoOpLangfuse:
    def trace(self, **_):       return _NoOpNode()
    def flush(self):            pass
    def shutdown(self):         pass


# ── Real client factory ───────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_langfuse():
    """
    Return a live Langfuse client if credentials are configured,
    otherwise return a silent no-op stub.
    Cached after first call — safe to call on every request.
    """
    try:
        from config import get_settings
        s = get_settings()

        if not s.langfuse_secret_key or not s.langfuse_public_key:
            logger.info("Langfuse keys not set — observability disabled")
            return _NoOpLangfuse()

        from langfuse import Langfuse
        client = Langfuse(
            public_key=s.langfuse_public_key,
            secret_key=s.langfuse_secret_key,
            host=s.langfuse_host,
        )
        logger.info("Langfuse observability enabled → %s", s.langfuse_host)
        return client

    except Exception as exc:
        logger.warning("Langfuse init failed (%s) — observability disabled", exc)
        return _NoOpLangfuse()
