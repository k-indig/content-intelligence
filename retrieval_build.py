"""Build the retrieval manifest for the 41 terms.

Reproduces the exact context + prompt the committed pipeline
(glossary_core.build_glossary_entry) would send to the model, but with a
"ground what's possible" retrieval policy:
  1. primary vector search at 0.45,
  2. broaden to 0.33 if empty,
  3. always merge in chunks from any editor-flagged linked articles.
Only terms with zero grounding after all three are flagged (no_grounding).

Writes retrieval_manifest.json: one object per term with the faithful
user_content string plus grounding diagnostics. SUPABASE_URL is reconstructed
from the service-role JWT ref (the env value is a duplicate JWT, not a URL).
"""
import base64
import json
import os
import re


def _reconstruct_supabase_url():
    key = os.environ.get("SUPABASE_SERVICE_KEY", "")
    url = os.environ.get("SUPABASE_URL", "")
    if url.startswith("http"):
        return url
    for val in (key, url):
        try:
            payload = val.split(".")[1]
            payload += "=" * (-len(payload) % 4)
            claims = json.loads(base64.urlsafe_b64decode(payload))
            if claims.get("ref"):
                return f"https://{claims['ref']}.supabase.co"
        except Exception:
            continue
    raise SystemExit("Could not reconstruct SUPABASE_URL")


os.environ["SUPABASE_URL"] = _reconstruct_supabase_url()

from batch_glossary import load_terms
from db.client import get_client, match_chunks
from ingestion.embed import embed_single
from glossary_core import slug_to_url, _build_direction_block, SYSTEM_PROMPT
from config import GLOSSARY_SECTIONS

TERMS_FILE = "terms_glossary_41.csv"
PRIMARY_THRESHOLD = 0.45
BROADEN_THRESHOLD = 0.33
MATCH_COUNT = 15


def slug_from_url(url: str) -> str:
    m = re.search(r"/p/([^/?#]+)", url.strip())
    return m.group(1) if m else ""


def chunks_for_slug(db, slug: str) -> list[dict]:
    """Fetch all chunks for an article by slug, shaped like match_chunks output.

    Stored slugs carry a numeric post-id prefix (e.g. '105426244.how-to-...'),
    which slug_to_url strips, so match on the suffix rather than equality.
    """
    arts = db.table("articles").select("id,title,url_slug").ilike("url_slug", f"%{slug}").execute().data
    out = []
    for a in arts:
        rows = (
            db.table("chunks").select("chunk_text,chunk_index")
            .eq("article_id", a["id"]).order("chunk_index").execute().data
        )
        for r in rows:
            out.append({
                "article_id": a["id"],
                "article_title": a["title"],
                "article_url_slug": a["url_slug"],
                "chunk_text": r["chunk_text"],
                "similarity": None,
            })
    return out


def build_user_content(term, chunks, angle, notes, links):
    """Identical to glossary_core.build_glossary_entry's prompt assembly."""
    seen = {}
    for ch in chunks:
        aid = ch["article_id"]
        if aid not in seen:
            seen[aid] = {"title": ch["article_title"],
                         "url": slug_to_url(ch["article_url_slug"]), "excerpts": []}
        seen[aid]["excerpts"].append(ch["chunk_text"][:600])
    blocks = []
    for info in seen.values():
        excerpts = "\n\n".join(info["excerpts"][:3])
        blocks.append(f"### {info['title']}\nURL: {info['url']}\n\n{excerpts}")
    context = "\n\n---\n\n".join(blocks)
    sections = ", ".join(GLOSSARY_SECTIONS)
    direction = _build_direction_block(angle, notes, links)
    return (
        f"Write a glossary entry for this term: **{term}**\n\n"
        f"Available sections (pick exactly one for the Suggested section field): {sections}"
        f"{direction}\n\n"
        f"Here are excerpts from Kevin's Growth Memo articles that reference it:\n\n{context}"
    )


def main():
    specs = load_terms(TERMS_FILE)
    db = get_client()
    manifest = []
    for i, spec in enumerate(specs, 1):
        term = spec["term"]
        emb = embed_single(term)
        tier = "primary"
        chunks = match_chunks(db, query_embedding=emb, match_count=MATCH_COUNT,
                              similarity_threshold=PRIMARY_THRESHOLD)
        if not chunks:
            chunks = match_chunks(db, query_embedding=emb, match_count=MATCH_COUNT,
                                  similarity_threshold=BROADEN_THRESHOLD)
            tier = "broadened" if chunks else "none"

        sims = [c["similarity"] for c in chunks if c.get("similarity") is not None]
        max_sim = max(sims) if sims else None

        # Merge editor-flagged linked articles (dedupe by chunk_text).
        seen_text = {c["chunk_text"] for c in chunks}
        link_articles = []
        for url in spec["links"]:
            slug = slug_from_url(url)
            if not slug:
                continue
            for ch in chunks_for_slug(db, slug):
                if ch["chunk_text"] not in seen_text:
                    chunks.append(ch)
                    seen_text.add(ch["chunk_text"])
                if ch["article_title"] not in link_articles:
                    link_articles.append(ch["article_title"])

        grounded = bool(chunks)
        if tier == "none" and link_articles:
            tier = "links_only"

        article_titles = []
        for c in chunks:
            if c["article_title"] not in article_titles:
                article_titles.append(c["article_title"])

        user_content = build_user_content(term, chunks, spec["angle"], spec["notes"], spec["links"]) if grounded else ""

        manifest.append({
            "index": i,
            "term": term,
            "angle": spec["angle"],
            "notes": spec["notes"],
            "links": spec["links"],
            "tier": tier,
            "max_similarity": round(max_sim, 3) if max_sim is not None else None,
            "n_passages": len(chunks),
            "n_articles": len(article_titles),
            "articles": article_titles,
            "no_grounding": not grounded,
            "user_content": user_content,
        })
        flag = "" if grounded else "  <-- NO GROUNDING"
        sim_s = f"{max_sim:.3f}" if max_sim is not None else "n/a"
        print(f"[{i:2}/41] {term[:46]:46} {tier:10} sim={sim_s} passages={len(chunks)}{flag}", flush=True)

    with open("retrieval_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"system_prompt": SYSTEM_PROMPT, "terms": manifest}, f, ensure_ascii=False, indent=2)

    prim = sum(1 for m in manifest if m["tier"] == "primary")
    broad = sum(1 for m in manifest if m["tier"] == "broadened")
    links_only = sum(1 for m in manifest if m["tier"] == "links_only")
    none = sum(1 for m in manifest if m["no_grounding"])
    print(f"\nGrounding: primary(>=0.45)={prim}  broadened(0.33-0.45)={broad}  "
          f"links_only={links_only}  none={none}")
    if none:
        print("No grounding:", ", ".join(m["term"] for m in manifest if m["no_grounding"]))


if __name__ == "__main__":
    main()
