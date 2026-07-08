# web/

Source of truth for the public **Growth Memo Glossary** page hosted at
<https://library.growth-memo.com/> (the site homepage).

## `glossary.html`

The full HTML/CSS/JS for the glossary. It is deployed by **pasting it into the
custom-code block of the beehiiv Home page** (beehiiv → Website → Home). beehiiv
does not expose that block through its API, so deploying is a manual copy-paste
and this file is the canonical copy.

### How the page works

The page renders a searchable, A–Z, category-filtered index from a `terms`
array hardcoded in the `<script>`. There is **no live connection to beehiiv** —
the array must be kept in sync with the published glossary posts by hand.

Each term is `{ name, url, cat }`:

- `name` — the post title
- `url` — `https://library.growth-memo.com/p/<slug>`
- `cat` — one of `ai | seo | behavior | foundations`, matching the post's
  beehiiv **content tag** (`AI Research → ai`, `SEO → seo`, `Behavior →
  behavior`, `Foundations → foundations`)

### Updating after publishing new glossary posts

1. Add one row to the `terms` array per new post (keep it alphabetical).
2. Set `cat` to match the post's beehiiv content tag. A post with no content
   tag will not appear in any beehiiv tag section and should still be given a
   `cat` here so it shows in the glossary.
3. Update the "Last synced" comment above the array.
4. Copy the whole file into the beehiiv Home page custom-code block and publish.

Last synced: 2026-07-08 — 35 published posts.
