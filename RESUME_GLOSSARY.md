# Resume runbook — create 41 glossary drafts in beehiiv

Temporary handoff note for continuing the "41 glossary terms → beehiiv drafts"
task in a fresh session. Safe to delete once the 41 drafts exist.

## Goal
Generate a glossary entry for each of the 41 terms in `terms_glossary_41.csv`
using the existing content-intelligence pipeline, then create each as a **draft**
post in beehiiv (do NOT publish — drafts only).

## Prerequisites (already set as environment variables)
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`.
Verify (prints only whether set + length, never the value):
```bash
for k in OPENAI_API_KEY ANTHROPIC_API_KEY SUPABASE_URL SUPABASE_SERVICE_KEY; do
  v="${!k}"; [ -n "$v" ] && echo "$k: SET (${#v})" || echo "$k: NOT set"; done
```

## Step 1 — install deps into the interpreter that runs the code
`python` here is /usr/local/bin/python but `pip` may target system python, so use
`python -m pip`. The system `cryptography`/`PyJWT` are broken, so shadow them:
```bash
cd /home/user/content-intelligence
python -m pip install --quiet --ignore-installed PyJWT cffi cryptography
python -m pip install --quiet anthropic openai supabase python-dotenv markdown
python -c "from db.client import get_client, match_chunks; from ingestion.embed import embed_single; import glossary_core, markdown; print('imports OK')"
```

## Step 2 — generate entries + structured manifest
```bash
python gen_glossary_drafts.py     # writes glossary_41.json and glossary_41.md
```
`glossary_41.json` is a list of objects, one per term, each with:
`term, title, suggested_section, subtitle, meta_title, meta_description,
body_markdown, body_html, full_markdown, article_count, passage_count`.
Any term with no vector matches gets `"no_passages": true` — flag those for the
user instead of drafting an empty entry.

## Step 3 — create one beehiiv draft per entry (via the beehiiv MCP)
Publication: **Growth Memo** = `pub_b279cb36-cdbf-4072-bf20-c131e962e0a5`.
For each entry in glossary_41.json call `mcp__beehiiv__save_post` with:
- `publication_id`: pub_b279cb36-cdbf-4072-bf20-c131e962e0a5
- `title`: entry.title
- `subtitle`: entry.subtitle
- `html_content`: entry.body_html
- `content_tags`: [entry.suggested_section]  (one of: "AI Research", "SEO", "Behavior", "Foundations")
- `seo_settings`: { "default_title": entry.meta_title, "default_description": entry.meta_description }

`save_post` creates a DRAFT (publishing stays a human action). The content tag is
what the glossary homepage uses to bucket the term (AI Research→ai, SEO→seo,
Behavior→behavior, Foundations→foundations), so it must be set.

## Step 4 — verify
`mcp__beehiiv__list_posts` with `status: draft` and confirm all 41 are present.
Report the term → suggested_section (category) mapping and any no-passages terms.

## Notes
- Never commit `.env` or the keys; never echo secret values.
- `terms_glossary_41.csv` and `gen_glossary_drafts.py` are already committed.
