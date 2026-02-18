import re

import streamlit as st

from db.client import get_client, get_all_articles
from analysis.linking import find_similar_chunks
from auth import require_auth
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, SUBSTACK_BASE_URL
import anthropic

st.set_page_config(page_title="Writing Assistant", layout="wide")
require_auth()
st.title("Writing Assistant")
st.caption(
    "Describe a topic or paste an outline. Claude will retrieve your relevant past articles "
    "and write a new draft that builds on — without repeating — what you've already covered."
)


def _slug_to_url(slug: str) -> str:
    clean = re.sub(r"^\d+\.", "", slug)
    return f"{SUBSTACK_BASE_URL}/p/{clean}"


def _get_style_examples(client, n=3) -> list[dict]:
    """Fetch a few recent articles to give Claude a voice reference."""
    result = (
        client.table("articles")
        .select("title, full_text_markdown")
        .order("post_date", desc=True)
        .limit(n)
        .execute()
    )
    return result.data


def draft_article(topic: str, similar_chunks: list[dict], style_examples: list[dict],
                  mode: str, word_count: int) -> str:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Build context from similar past articles
    seen_articles = {}
    for chunk in similar_chunks:
        aid = chunk["article_id"]
        if aid not in seen_articles:
            seen_articles[aid] = {
                "title": chunk["article_title"],
                "url": _slug_to_url(chunk["article_url_slug"]),
                "excerpts": [],
            }
        seen_articles[aid]["excerpts"].append(chunk["chunk_text"][:600])

    past_context = []
    for info in seen_articles.values():
        excerpts = "\n\n".join(info["excerpts"][:2])
        past_context.append(
            f"Article: \"{info['title']}\" ({info['url']})\n{excerpts}"
        )

    # Style examples
    style_block = ""
    if style_examples:
        samples = []
        for ex in style_examples:
            samples.append(f"### {ex['title']}\n{ex['full_text_markdown'][:800]}")
        style_block = "\n\n".join(samples)

    if mode == "Full draft":
        instruction = (
            f"Write a complete Growth Memo newsletter article of approximately {word_count} words on this topic. "
            "Include a compelling intro, structured body with subheadings, and a clear takeaway at the end. "
            "Naturally weave in 2-3 internal links to the related past articles listed above where relevant."
        )
    elif mode == "Outline":
        instruction = (
            "Write a detailed outline for a Growth Memo article on this topic. "
            "Include: working title, angle/hook, 4-6 section headings with 2-3 bullet points each, "
            "and a note on what differentiates this from past coverage."
        )
    else:  # Angle suggestions
        instruction = (
            "Suggest 5 distinct angles or hooks for a Growth Memo article on this topic. "
            "For each angle: give a working title, a 2-sentence pitch, and explain how it differs from or builds on past coverage."
        )

    prompt = f"""You are a writing assistant for Kevin Indig, author of the "Growth Memo" newsletter about SEO, organic growth, and digital marketing.

Kevin's writing style (recent articles for reference):
{style_block}

RELATED PAST ARTICLES (use these as context — build on them, don't repeat them):
{chr(10).join(past_context)}

TOPIC / INPUT FROM KEVIN:
{topic}

TASK:
{instruction}

Important: Match Kevin's voice closely — direct, analytical, data-informed, practitioner-focused. Avoid generic SEO advice."""

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# UI
client = get_client()

col1, col2 = st.columns([3, 1])
with col1:
    topic = st.text_area(
        "Topic, angle, or draft outline",
        placeholder="e.g. 'How AI Overviews are changing the ROI calculation for informational SEO content' or paste a rough outline...",
        height=150,
    )
with col2:
    mode = st.radio("Output mode", ["Full draft", "Outline", "Angle suggestions"])
    word_count = 800
    if mode == "Full draft":
        word_count = st.slider("Target word count", 400, 1600, 800, 100)
    similarity_threshold = st.slider("Context sensitivity", 0.3, 0.8, 0.45, 0.05,
                                     help="Lower = pulls in more loosely related past articles")

if st.button("Generate", disabled=not topic.strip()):
    with st.spinner("Finding relevant past articles..."):
        similar = find_similar_chunks(
            client, topic,
            match_count=20,
            similarity_threshold=similarity_threshold,
        )
        style_examples = _get_style_examples(client, n=3)

    if not similar:
        st.warning("No closely related past articles found. Try lowering Context sensitivity.")
        st.stop()

    # Show which past articles are being used as context
    seen = {}
    for chunk in similar:
        aid = chunk["article_id"]
        if aid not in seen:
            seen[aid] = {"title": chunk["article_title"], "url": _slug_to_url(chunk["article_url_slug"])}

    with st.expander(f"Context: {len(seen)} past articles retrieved"):
        for info in seen.values():
            st.markdown(f"- [{info['title']}]({info['url']})")

    with st.spinner(f"Claude is writing your {mode.lower()}..."):
        result = draft_article(topic, similar, style_examples, mode, word_count)

    st.subheader(mode)
    st.markdown(result)

    # Copy-friendly text area
    with st.expander("Plain text (copy-friendly)"):
        st.text_area("", value=result, height=400, label_visibility="collapsed")
