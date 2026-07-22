"""Generate glossary entries for the 41 terms and emit a structured manifest.

Runs the same retrieval + generation pipeline as batch_glossary.py (embed term ->
vector-search Growth Memo chunks -> Claude writes the entry), then parses each
entry into structured fields and pre-converts the body to HTML so the beehiiv
drafts can be created directly from glossary_41.json.

Outputs:
  glossary_41.json  - list of {term, title, suggested_section, subtitle,
                       meta_title, meta_description, body_markdown, body_html,
                       full_markdown, article_count, passage_count}
  glossary_41.md    - human-readable combined markdown (same shape batch produces)

Usage:
  python gen_glossary_drafts.py                 # default terms_glossary_41.csv
  python gen_glossary_drafts.py --articles 20 --threshold 0.4
"""
import argparse
import json
import re
import sys
import time

import markdown as md

from batch_glossary import load_terms
from db.client import get_client, match_chunks
from ingestion.embed import embed_single
import glossary_core

DEFAULT_TERMS = "terms_glossary_41.csv"


def _field(text: str, label: str) -> str:
    """Extract a bold-labelled metadata value (value sits on the line(s) below the label)."""
    m = re.search(
        rf"\*\*{re.escape(label)}\*\*\s*\n(.+?)(?=\n\s*\n|\n\*\*|\n##|\Z)",
        text,
        re.S,
    )
    return m.group(1).strip() if m else ""


def parse_entry(entry_md: str, term: str) -> dict:
    """Parse a generated glossary markdown entry into structured fields."""
    title = term
    m = re.search(r"^#\s+(.+)$", entry_md, re.M)
    if m:
        title = m.group(1).strip()

    suggested = _field(entry_md, "Suggested section")
    subtitle = _field(entry_md, "Subtitle")
    meta_title = _field(entry_md, "Meta title")
    meta_desc = _field(entry_md, "Meta description")

    # Body = everything from the first '## ' heading onward.
    bm = re.search(r"(^##\s+.+)\Z", entry_md, re.S | re.M)
    body_md = bm.group(1).strip() if bm else ""

    body_html = md.markdown(body_md, extensions=["sane_lists"]) if body_md else ""

    return {
        "term": term,
        "title": title,
        "suggested_section": suggested,
        "subtitle": subtitle,
        "meta_title": meta_title,
        "meta_description": meta_desc,
        "body_markdown": body_md,
        "body_html": body_html,
        "full_markdown": entry_md.strip(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--terms", default=DEFAULT_TERMS)
    ap.add_argument("--articles", type=int, default=15)
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--json-out", default="glossary_41.json")
    ap.add_argument("--md-out", default="glossary_41.md")
    args = ap.parse_args()

    specs = load_terms(args.terms)
    if not specs:
        print("No terms loaded.", file=sys.stderr)
        return 1
    print(f"Loaded {len(specs)} terms from {args.terms}\n", flush=True)

    db = get_client()
    results = []
    for i, spec in enumerate(specs, 1):
        term = spec["term"]
        print(f"[{i}/{len(specs)}] {term}", flush=True)
        try:
            embedding = embed_single(term)
            chunks = match_chunks(
                db,
                query_embedding=embedding,
                match_count=args.articles,
                similarity_threshold=args.threshold,
            )
        except Exception as e:
            print(f"  retrieval error: {type(e).__name__}: {e}", flush=True)
            results.append({"term": term, "title": term, "error": str(e),
                            "suggested_section": "", "subtitle": "", "meta_title": "",
                            "meta_description": "", "body_markdown": "", "body_html": "",
                            "full_markdown": "", "no_passages": True})
            continue

        if not chunks:
            print("  no relevant passages found", flush=True)
            results.append({"term": term, "title": term, "suggested_section": "",
                            "subtitle": "", "meta_title": "", "meta_description": "",
                            "body_markdown": "", "body_html": "", "full_markdown": "",
                            "no_passages": True})
            continue

        art_count = len({c["article_id"] for c in chunks})
        print(f"  {art_count} articles / {len(chunks)} passages", flush=True)

        entry = glossary_core.build_glossary_entry(
            term, chunks,
            angle=spec["angle"], notes=spec["notes"], source_links=spec["links"],
        )
        parsed = parse_entry(entry, term)
        parsed["article_count"] = art_count
        parsed["passage_count"] = len(chunks)
        # Flag anything that parsed oddly so it can be reviewed before drafting.
        missing = [k for k in ("suggested_section", "subtitle", "meta_title",
                               "meta_description", "body_html") if not parsed[k]]
        if missing:
            print(f"  warning: empty fields after parse: {', '.join(missing)}", flush=True)
        results.append(parsed)
        time.sleep(1)

    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(args.md_out, "w", encoding="utf-8") as f:
        f.write("# Glossary\n\n")
        blocks = [
            r["full_markdown"] or f"## {r['term']}\n\n*No relevant passages found.*"
            for r in results
        ]
        f.write("\n\n---\n\n".join(blocks))
        f.write("\n")

    ok = [r for r in results if not r.get("no_passages")]
    print(f"\nGenerated {len(ok)}/{len(results)} entries.")
    print(f"Wrote {args.json_out} and {args.md_out}")
    missing = [r["term"] for r in results if r.get("no_passages")]
    if missing:
        print(f"\n{len(missing)} term(s) had no passages (need attention):")
        for t in missing:
            print(f"  - {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
