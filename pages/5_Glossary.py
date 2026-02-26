import re

import anthropic
import streamlit as st

from auth import require_auth
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, SUBSTACK_BASE_URL
from db.client import get_client, match_chunks
from ingestion.embed import embed_single

st.set_page_config(page_title="Glossary Builder", layout="wide")
require_auth()
st.title("Glossary Builder")
st.caption(
    "Enter a term or concept. The app finds every Growth Memo where you've referenced it "
    "and writes a glossary entry in your voice."
)

SYSTEM_PROMPT = """You are a writing assistant for Kevin Indig, author of the Growth Memo newsletter about SEO, organic growth, and AI search.

Your job is to write a glossary entry for a term or concept based on how Kevin has actually used and explained it across his articles. Write in Kevin's voice: direct, analytical, practitioner-focused. No fluff.

The glossary entry must have exactly these sections in this order:

**[Term]**

**What it means**
A clear, concise definition (2–4 sentences). Base this on how Kevin has described or used the term across his articles — not generic textbook definitions.

**Why it matters**
Why this concept is important for SEO practitioners and growth teams (2–4 sentences). Ground it in the practical implications Kevin has written about.

**How to use it**
2–4 concrete, actionable steps or approaches for applying this concept. Frame it for practitioners — what would a growth team or SEO lead actually do with this? Base it on the provided excerpts where possible; use domain knowledge to fill gaps.

**From Growth Memo:**
2–3 direct insights or quotes from Kevin's actual articles below. Use the exact wording where possible, or paraphrase closely. Each insight should be followed by its source in parentheses: (Source: [Article Title])

**Related concepts**
3–5 closely related terms, each with a one-sentence explanation of how it connects. Example format:
- **Content clusters** — the structural method for building topical authority

**Referenced in these Growth Memos**
A bulleted list of every article in the context below that meaningfully references this term, formatted as:
- [Article Title](full URL)

Keep the whole entry under 600 words. Do not invent content for the Growth Memo sections — only use what's in the provided article excerpts. You may use domain knowledge for "How to use it" and "Related concepts".

---

WRITING STYLE RULES — follow these strictly:

HARD BANS (never do these):
- No em dashes (—). Use commas, colons, parentheses, or split into two sentences.
- No emoji in headings or bullets.
- Sentence case for headings, not Title Case.
- Bold sparingly — only for true labels, never for emphasis or decoration.
- No inline-header lists (Label: item, Label: item…). Use prose or a real table.
- No conclusion or recap sections. End when the work is done.
- No didactic disclaimers: "it is important to note", "it is crucial to", "may vary", etc.
- No "Not X, but Y" constructions. State the claim once, plainly.
- No "challenges / future prospects" boilerplate that ends on an upbeat note.
- No "serves as", "stands as", "marks", "represents" as substitutes for "is/are".
- No vague attributions ("experts say", "studies show") — name the source or remove the claim.
- No significance inflation: "pivotal moment", "vital role", "sets the stage", "transformative impact".
- No rule-of-three stacks used repeatedly as a polish device.
- No "From X to Y" ranges that don't define a real scale.

BANNED WORDS — never use:
delve, landscape, evolving, context, insight, nuanced, perspective, paradigm, comprehensive, supercharge, framework, facet, dynamic, intricacies, holistic, iterative, synergy, confluence, pivotal, nuance, robust, transformative, underpinning, spectrum, trajectory, in-depth, tapestry, testament, intrigue, elusive, quintessential, symphony, labyrinth, resonance, embodiment, monumental, ethereal, boundless, mosaic, woven, sculpted, intricate, otherworldly.

BANNED OPENERS AND TRANSITIONS — never use:
"In fact,", "Indeed,", "Absolutely,", "Clearly,", "First and foremost,", "As a result,", "Therefore,", "Consequently,", "In other words,", "To put it simply,", "In summary,", "In conclusion,", "All in all,", "More importantly,", "On one hand / on the other hand,", "The challenge is,", "The key issue is,", "Here's the kicker.", "In a world of."

STYLE:
- Vary sentence lengths. Avoid repeating the same sentence structure across paragraphs.
- Repeat the exact term when needed for clarity — do not cycle synonyms.
- No metaphors, idioms, clichés, or hyperbole.
- No generic advice or invented examples."""


def _slug_to_url(slug: str) -> str:
    clean = re.sub(r"^\d+\.", "", slug)
    return f"{SUBSTACK_BASE_URL}/p/{clean}"


def build_glossary_entry(term: str, chunks: list[dict]) -> str:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Deduplicate articles and build context
    seen_articles = {}
    for chunk in chunks:
        aid = chunk["article_id"]
        if aid not in seen_articles:
            seen_articles[aid] = {
                "title": chunk["article_title"],
                "url": _slug_to_url(chunk["article_url_slug"]),
                "excerpts": [],
            }
        seen_articles[aid]["excerpts"].append(chunk["chunk_text"][:600])

    context_blocks = []
    for info in seen_articles.values():
        excerpts = "\n\n".join(info["excerpts"][:3])
        context_blocks.append(
            f"### {info['title']}\nURL: {info['url']}\n\n{excerpts}"
        )

    context = "\n\n---\n\n".join(context_blocks)

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Write a glossary entry for this term: **{term}**\n\nHere are excerpts from Kevin's Growth Memo articles that reference it:\n\n{context}",
            }
        ],
    )
    return response.content[0].text


# ── UI ──────────────────────────────────────────────────────────────────────

db = get_client()

col1, col2 = st.columns([3, 1])
with col1:
    term = st.text_input(
        "Term or concept",
        placeholder="e.g. zero-click, AEO, content velocity, topical authority…",
    )
with col2:
    match_count = st.slider("Max articles to search", 5, 30, 15)
    threshold = st.slider(
        "Relevance threshold", 0.3, 0.8, 0.45, 0.05,
        help="Higher = stricter match. Lower = casts a wider net."
    )

if st.button("Build glossary entry", disabled=not term.strip()):
    with st.spinner(f"Searching {match_count} most relevant Growth Memo passages…"):
        embedding = embed_single(term)
        chunks = match_chunks(
            db,
            query_embedding=embedding,
            match_count=match_count,
            similarity_threshold=threshold,
        )

    if not chunks:
        st.warning(
            "No relevant passages found. Try lowering the relevance threshold, "
            "or check that the term appears in your Growth Memos."
        )
        st.stop()

    # Show which articles were pulled
    seen = {}
    for c in chunks:
        aid = c["article_id"]
        if aid not in seen:
            seen[aid] = {"title": c["article_title"], "url": _slug_to_url(c["article_url_slug"])}

    with st.expander(f"Found {len(seen)} relevant articles ({len(chunks)} passages)"):
        for info in seen.values():
            st.markdown(f"- [{info['title']}]({info['url']})")

    with st.spinner("Writing glossary entry…"):
        entry = build_glossary_entry(term, chunks)

    st.divider()
    st.markdown(entry)

    with st.expander("Copy-friendly markdown"):
        st.text_area("", value=entry, height=350, label_visibility="collapsed")
