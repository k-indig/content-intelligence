"""Assemble generated entry files into the structured manifest + combined md.

Reads gen_entries/entry_<index>.md (one per term), applies the SAME
post-processors the committed pipeline applies (_dedup_references,
_capitalize_after_colon), parses each into structured fields, pre-converts the
body to beehiiv-ready HTML, and validates against the entry contract.

Outputs glossary_41.json and glossary_41.md, plus a validation report.
"""
import json
import re
import sys

import markdown as md

from glossary_core import _dedup_references, _capitalize_after_colon
from config import GLOSSARY_SECTIONS

ENTRY_DIR = "gen_entries"


def _field(text: str, label: str) -> str:
    m = re.search(
        rf"\*\*{re.escape(label)}\*\*\s*\n(.+?)(?=\n\s*\n|\n\*\*|\n##|\Z)",
        text, re.S,
    )
    return m.group(1).strip() if m else ""


def _wrap_li_paragraphs(html: str) -> str:
    """beehiiv's listItem contract is <li><p>...</p></li>; Python-Markdown emits
    bare <li>...</li> for tight lists. Wrap inline-only items in <p>, leaving
    items that already hold a block element (<p>, <ul>, <ol>) untouched."""
    def repl(m):
        inner = m.group(1)
        if re.search(r"<(p|ul|ol)\b", inner):
            return m.group(0)
        return f"<li><p>{inner}</p></li>"
    return re.sub(r"<li>(.*?)</li>", repl, html, flags=re.S)


def parse_entry(entry_md: str, term: str) -> dict:
    title = term
    m = re.search(r"^#\s+(.+)$", entry_md, re.M)
    if m:
        title = m.group(1).strip()
    body_m = re.search(r"(^##\s+.+)\Z", entry_md, re.S | re.M)
    body_md = body_m.group(1).strip() if body_m else ""
    body_html = _wrap_li_paragraphs(md.markdown(body_md, extensions=["sane_lists"])) if body_md else ""
    return {
        "term": term,
        "title": title,
        "suggested_section": _field(entry_md, "Suggested section"),
        "subtitle": _field(entry_md, "Subtitle"),
        "meta_title": _field(entry_md, "Meta title"),
        "meta_description": _field(entry_md, "Meta description"),
        "body_markdown": body_md,
        "body_html": body_html,
        "full_markdown": entry_md.strip(),
    }


# Style checks (subset of the hard bans that are easy to detect mechanically).
BANNED_WORDS = {
    "delve", "landscape", "evolving", "nuanced", "paradigm", "comprehensive",
    "supercharge", "framework", "facet", "intricacies", "holistic", "iterative",
    "synergy", "confluence", "pivotal", "robust", "transformative", "underpinning",
    "spectrum", "trajectory", "in-depth", "tapestry", "testament", "quintessential",
    "symphony", "labyrinth", "resonance", "embodiment", "monumental", "mosaic",
    "woven", "sculpted", "intricate",
}


def validate(term, p, grounding):
    issues = []
    if p["suggested_section"] not in GLOSSARY_SECTIONS:
        issues.append(f"suggested_section '{p['suggested_section']}' not in {GLOSSARY_SECTIONS}")
    for fld in ("subtitle", "meta_title", "meta_description", "body_html", "title"):
        if not p[fld]:
            issues.append(f"empty {fld}")
    if len(p["subtitle"]) > 145:
        issues.append(f"subtitle {len(p['subtitle'])}>140")
    if len(p["meta_title"]) > 62:
        issues.append(f"meta_title {len(p['meta_title'])}>60")
    if len(p["meta_description"]) > 160:
        issues.append(f"meta_description {len(p['meta_description'])}>155")
    for h in ("## Why it matters", "## How to use this knowledge",
              "## Related concepts"):
        if h.lower() not in p["body_markdown"].lower():
            issues.append(f"missing section {h}")
    if not re.search(r"## (What it means|The quick answer)", p["body_markdown"]):
        issues.append("missing definition section")
    # Em dash in prose (allowed only in blockquote attributions and related bullets).
    for line in p["body_markdown"].splitlines():
        s = line.strip()
        if "—" in s and not s.startswith(">") and not s.startswith("-") and not s.startswith("—"):
            issues.append(f"em dash in prose: {s[:60]}")
            break
    low = p["body_markdown"].lower()
    hits = sorted({w for w in BANNED_WORDS if re.search(rf"\b{re.escape(w)}\b", low)})
    if hits:
        issues.append("banned words: " + ", ".join(hits))
    return issues


def main():
    manifest = json.load(open("retrieval_manifest.json"))["terms"]
    by_index = {m["index"]: m for m in manifest}

    results, report = [], []
    for idx in range(1, len(manifest) + 1):
        term = by_index[idx]["term"]
        path = f"{ENTRY_DIR}/entry_{idx:02d}.md"
        try:
            raw = open(path, encoding="utf-8").read()
        except FileNotFoundError:
            report.append((idx, term, ["MISSING FILE"]))
            results.append({"term": term, "index": idx, "missing_file": True})
            continue
        raw = _capitalize_after_colon(_dedup_references(raw.strip()))
        p = parse_entry(raw, term)
        p["index"] = idx
        p["tier"] = by_index[idx]["tier"]
        p["max_similarity"] = by_index[idx]["max_similarity"]
        p["n_passages"] = by_index[idx]["n_passages"]
        issues = validate(term, p, by_index[idx])
        p["issues"] = issues
        results.append(p)
        if issues:
            report.append((idx, term, issues))

    results.sort(key=lambda r: r["index"])
    json.dump(results, open("glossary_41.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    with open("glossary_41.md", "w", encoding="utf-8") as f:
        f.write("# Glossary\n\n")
        f.write("\n\n---\n\n".join(
            r.get("full_markdown") or f"## {r['term']}\n\n*missing*" for r in results))
        f.write("\n")

    ok = sum(1 for r in results if not r.get("missing_file") and not r["issues"])
    print(f"Assembled {len(results)} entries -> glossary_41.json / glossary_41.md")
    print(f"Clean: {ok}/{len(results)}")
    if report:
        print("\nEntries needing attention:")
        for idx, term, issues in report:
            print(f"  [{idx:02d}] {term}")
            for i in issues:
                print(f"        - {i}")
    else:
        print("All entries passed validation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
