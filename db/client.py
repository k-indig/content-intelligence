from supabase import create_client, ClientOptions
from config import SUPABASE_URL, SUPABASE_SERVICE_KEY


def get_client():
    opts = ClientOptions(postgrest_client_timeout=60)
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY, options=opts)


def upsert_article(client, article: dict) -> int:
    """Upsert an article and return its id."""
    result = (
        client.table("articles")
        .upsert(article, on_conflict="post_id")
        .execute()
    )
    return result.data[0]["id"]


def upsert_chunks(client, chunks: list[dict]):
    """Upsert chunks in small batches to avoid timeouts."""
    BATCH = 5
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        client.table("chunks").upsert(
            batch, on_conflict="article_id,chunk_index"
        ).execute()


def get_all_articles(client) -> list[dict]:
    """Fetch all articles (without full_text for listing)."""
    result = (
        client.table("articles")
        .select("id, post_id, title, subtitle, post_date, type, audience, url_slug, word_count")
        .order("post_date", desc=True)
        .execute()
    )
    return result.data


def get_article_by_id(client, article_id: int) -> dict:
    """Fetch a single article with full text."""
    result = (
        client.table("articles")
        .select("*")
        .eq("id", article_id)
        .single()
        .execute()
    )
    return result.data


def get_all_chunk_embeddings(client) -> list[dict]:
    """Fetch all chunks with embeddings for clustering."""
    result = (
        client.table("chunks")
        .select("id, article_id, chunk_text, heading, embedding")
        .execute()
    )
    return result.data


def match_chunks(client, query_embedding: list[float], match_count=15,
                 similarity_threshold=0.5, exclude_article_id=None) -> list[dict]:
    """Call the match_chunks RPC function."""
    params = {
        "query_embedding": query_embedding,
        "match_count": match_count,
        "similarity_threshold": similarity_threshold,
    }
    if exclude_article_id is not None:
        params["exclude_article_id"] = exclude_article_id
    result = client.rpc("match_chunks", params).execute()
    return result.data


def get_article_count(client) -> int:
    result = client.table("articles").select("id", count="exact").execute()
    return result.count


def get_chunk_count(client) -> int:
    result = client.table("chunks").select("id", count="exact").execute()
    return result.count


def upsert_gap_feedback(client, cluster_label: str, suggestion: str, rating: str):
    """Save or update feedback on a gap suggestion."""
    client.table("gap_feedback").upsert(
        {"cluster_label": cluster_label, "suggestion": suggestion, "rating": rating},
        on_conflict="cluster_label,suggestion",
    ).execute()


def get_all_gap_feedback(client) -> list[dict]:
    """Fetch all gap feedback for use in prompts."""
    result = client.table("gap_feedback").select("*").execute()
    return result.data
