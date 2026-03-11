"""Batch glossary builder.

Reads terms from a text file (one per line) and generates a glossary entry
for each using the same pipeline as the Streamlit Glossary page.

Usage:
    python batch_glossary.py terms.txt                  # writes glossary.md
    python batch_glossary.py terms.txt -o my_output.md  # custom output file
    python batch_glossary.py terms.txt --articles 20    # search more articles
    python batch_glossary.py terms.txt --threshold 0.4  # looser relevance
"""

import argparse
import sys
import time

from db.client import get_client, match_chunks
from ingestion.embed import embed_single
from pages import glossary_core


def load_terms(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        terms = [line.strip() for line in f if line.strip()]
    return terms


def main():
    parser = argparse.ArgumentParser(description="Batch glossary builder")
    parser.add_argument("terms_file", help="Text file with one term per line")
    parser.add_argument("-o", "--output", default="glossary.md",
                        help="Output markdown file (default: glossary.md)")
    parser.add_argument("--articles", type=int, default=15,
                        help="Max articles to search per term (default: 15)")
    parser.add_argument("--threshold", type=float, default=0.45,
                        help="Similarity threshold (default: 0.45)")
    args = parser.parse_args()

    terms = load_terms(args.terms_file)
    if not terms:
        print("No terms found in file.")
        sys.exit(1)

    print(f"Loaded {len(terms)} terms from {args.terms_file}")
    print(f"Output: {args.output} | articles: {args.articles} | threshold: {args.threshold}\n")

    db = get_client()
    entries = []

    for i, term in enumerate(terms, 1):
        print(f"[{i}/{len(terms)}] {term}...")

        embedding = embed_single(term)
        chunks = match_chunks(
            db,
            query_embedding=embedding,
            match_count=args.articles,
            similarity_threshold=args.threshold,
        )

        if not chunks:
            print(f"  ⚠ No relevant passages found, skipping.")
            entries.append(f"## {term}\n\n*No relevant passages found.*\n")
            continue

        article_count = len({c["article_id"] for c in chunks})
        print(f"  Found {article_count} articles ({len(chunks)} passages)")

        entry = glossary_core.build_glossary_entry(term, chunks)
        entries.append(entry)
        print(f"  ✓ Done")

        # Brief pause between API calls to be respectful
        if i < len(terms):
            time.sleep(1)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("# Glossary\n\n")
        f.write("\n\n---\n\n".join(entries))
        f.write("\n")

    print(f"\nWrote {len(entries)} entries to {args.output}")


if __name__ == "__main__":
    main()
