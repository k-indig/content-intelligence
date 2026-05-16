"""Backfill article titles where the current title was derived from the slug.

The old parser fell back to ``slug.replace("-", " ").title()`` whenever
posts.csv lacked a title, producing things like "Googles Ai Mode Seo Impact
Ai Mode" or "I Ve Been Diging Deep". This script finds those rows and
replaces the title with the first H1 from the stored markdown.

Usage:
    python -m ingestion.backfill_titles                 # dry-run (default)
    python -m ingestion.backfill_titles --apply         # actually update rows
    python -m ingestion.backfill_titles --limit 10      # process fewer rows
"""
import argparse
import re

from db.client import get_client


_H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_INLINE_FMT_RE = re.compile(r"[*_`]+")


def slug_titleized(slug: str) -> str:
    """Reproduce the old parse.py fallback exactly: slug.replace('-', ' ').title()."""
    clean = re.sub(r"^\d+\.", "", slug or "")
    return clean.replace("-", " ").title()


def extract_h1(markdown: str) -> str:
    """Return the cleaned text of the first H1 in the markdown, or ''."""
    if not markdown:
        return ""
    match = _H1_RE.search(markdown)
    if not match:
        return ""
    text = match.group(1)
    text = _LINK_RE.sub(r"\1", text)
    text = _INLINE_FMT_RE.sub("", text)
    return text.strip()


def is_linkedin_slug(slug: str) -> bool:
    return (slug or "").startswith("linkedin.") or (slug or "").startswith("linkedin-")


def fetch_candidates(client) -> list[dict]:
    """Fetch articles whose title equals the slug-titleized fallback."""
    result = (
        client.table("articles")
        .select("id, url_slug, title, full_text_markdown")
        .execute()
    )
    candidates = []
    for row in result.data or []:
        slug = row.get("url_slug") or ""
        if is_linkedin_slug(slug):
            continue
        if not row.get("title"):
            continue
        if row["title"].strip() == slug_titleized(slug):
            candidates.append(row)
    return candidates


def main():
    parser = argparse.ArgumentParser(description="Backfill slug-derived article titles.")
    parser.add_argument("--apply", action="store_true",
                        help="Actually write updates (default: dry-run).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N candidates.")
    args = parser.parse_args()

    client = get_client()
    candidates = fetch_candidates(client)
    print(f"Found {len(candidates)} article(s) with slug-derived titles.")

    if args.limit:
        candidates = candidates[:args.limit]

    updated = 0
    skipped = 0
    for row in candidates:
        new_title = extract_h1(row.get("full_text_markdown") or "")
        if not new_title:
            print(f"  [skip] id={row['id']} slug={row['url_slug']!r} — no H1 in markdown")
            skipped += 1
            continue
        if new_title == row["title"]:
            skipped += 1
            continue

        print(f"  id={row['id']} slug={row['url_slug']!r}")
        print(f"    before: {row['title']!r}")
        print(f"    after:  {new_title!r}")

        if args.apply:
            client.table("articles").update({"title": new_title}).eq("id", row["id"]).execute()
        updated += 1

    print()
    if args.apply:
        print(f"Updated {updated} row(s). Skipped {skipped}.")
    else:
        print(f"Dry-run: would update {updated} row(s). Skipped {skipped}.")
        print("Re-run with --apply to write changes.")


if __name__ == "__main__":
    main()
