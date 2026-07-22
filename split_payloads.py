"""Print a review summary and write per-entry beehiiv draft payloads."""
import json
import os
import re
from collections import Counter

data = json.load(open("glossary_41.json"))
os.makedirs("draft_payloads", exist_ok=True)

SECTION_TO_CAT = {"AI Research": "ai", "SEO": "seo", "Behavior": "behavior", "Foundations": "foundations"}

print(f"{'#':>2}  {'title':42} {'section':11} {'sub':>3} {'mt':>2} {'md':>3} {'tier':10} sim")
print("-" * 92)
cats = Counter()
for e in data:
    idx = e["index"]
    title = e["title"]
    sec = e["suggested_section"]
    cats[sec] += 1
    sub, mt, mdd = len(e["subtitle"]), len(e["meta_title"]), len(e["meta_description"])
    sim = e.get("max_similarity")
    sim_s = f"{sim:.3f}" if sim is not None else "link"
    payload = {
        "index": idx,
        "term": e["term"],
        "title": title,
        "subtitle": e["subtitle"],
        "content_tag": sec,
        "html_content": e["body_html"],
        "meta_title": e["meta_title"],
        "meta_description": e["meta_description"],
    }
    json.dump(payload, open(f"draft_payloads/draft_{idx:02d}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    with open(f"draft_payloads/body_{idx:02d}.html", "w", encoding="utf-8") as hf:
        hf.write(e["body_html"])
    warn = ""
    if sec not in SECTION_TO_CAT:
        warn += " BAD-SECTION"
    if sub > 140: warn += " SUB>140"
    if mt > 60: warn += " MT>60"
    if mdd > 155: warn += " MD>155"
    if not e["body_html"].strip(): warn += " EMPTY-HTML"
    print(f"{idx:>2}  {title[:42]:42} {sec:11} {sub:>3} {mt:>2} {mdd:>3} {e['tier']:10} {sim_s}{warn}")

print("\nSection distribution:", dict(cats))
print("Payloads written to draft_payloads/ ->", len(data), "files")
