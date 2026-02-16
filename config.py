import os
from dotenv import load_dotenv

load_dotenv()


def _get_secret(key: str) -> str:
    """Read from Streamlit secrets (cloud) or env vars (local)."""
    try:
        import streamlit as st
        return st.secrets[key]
    except Exception:
        return os.getenv(key)


# API Keys
OPENAI_API_KEY = _get_secret("OPENAI_API_KEY")
ANTHROPIC_API_KEY = _get_secret("ANTHROPIC_API_KEY")
SUPABASE_URL = _get_secret("SUPABASE_URL")
SUPABASE_SERVICE_KEY = _get_secret("SUPABASE_SERVICE_KEY")

# Embedding config
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
EMBEDDING_BATCH_SIZE = 100

# Chunking config
MAX_CHUNK_TOKENS = 1000
MERGE_THRESHOLD_TOKENS = 750
MIN_ARTICLE_BYTES = 100

# Analysis config
DEFAULT_CLUSTER_COUNT = 15
DEFAULT_SIMILAR_CHUNKS = 15
DEFAULT_LINK_SUGGESTIONS = 8

# Claude config
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"

# Newsletter config
SUBSTACK_BASE_URL = "https://www.growth-memo.com"
