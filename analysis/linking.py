import re

import anthropic

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, DEFAULT_LINK_SUGGESTIONS, SUBSTACK_BASE_URL
from db.client import match_chunks
from ingestion.embed import embed_single


def _slug_to_url(slug: str) -> str:
    """Convert a stored slug (possibly with numeric prefix) to a full URL."""
    clean = re.sub(r"^\d+\.", "", slug)
    return f"{SUBSTACK_BASE_URL}/p/{clean}"


def find_similar_chunks(client, text: str, match_count: int = 15,
                        similarity_threshold: float = 0.5,
                        exclude_article_id: int = None) -> list[dict]:
    """Embed input text and find similar chunks via pgvector."""
    embedding = embed_single(text)
    return match_chunks(
        client, embedding,
        match_count=match_count,
        similarity_threshold=similarity_threshold,
        exclude_article_id=exclude_article_id,
    )


def suggest_internal_links(source_title: str, source_text: str,
                           similar_chunks: list[dict],
                           max_suggestions: int = DEFAULT_LINK_SUGGESTIONS) -> str:
    """Use Claude to suggest specific internal links with anchor text."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    chunks_context = []
    for i, chunk in enumerate(similar_chunks):
        url = _slug_to_url(chunk["article_url_slug"])
        chunks_context.append(
            f"[{i+1}] Article: \"{chunk['article_title']}\" "
            f"(URL: {url})\n"
            f"Section: {chunk.get('heading', 'N/A')}\n"
            f"Content: {chunk['chunk_text'][:500]}\n"
            f"Similarity: {chunk['similarity']:.3f}"
        )

    prompt = f"""You are an internal linking specialist for the "Growth Memo" newsletter about SEO, organic growth, and digital marketing.

SOURCE ARTICLE: "{source_title}"
{source_text[:2000]}

SIMILAR CONTENT FROM OTHER ARTICLES:
{chr(10).join(chunks_context)}

Suggest {max_suggestions} specific internal links to add to the source article. For each suggestion:
1. Quote the exact phrase in the source article that should become the anchor text
2. Specify which article to link to with its full URL
3. Explain why this link adds value for the reader

Format each suggestion in markdown:
### Link [number]
**Anchor text:** "[exact phrase from source]"
**Link to:** [article title](full URL)
**Reason:** [brief explanation]
"""

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
