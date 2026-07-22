# Glossary

# Open weight large language model (LLM)

**Suggested section**
Foundations

**Subtitle**
With open weight models, you control cost, privacy, and the data a model learns from, instead of renting all of that from a vendor.

**Meta title**
What is an open weight LLM?

**Meta description**
An open weight LLM publishes its trained weights for anyone to download, run, and fine-tune, instead of locking the model behind a provider's API.

## What it means
An open weight large language model is a model whose trained weights, its learned parameters, are published for anyone to download, run, and fine-tune on their own hardware. This differs from a closed model that you reach only through a provider's API. You decide where an open weight LLM runs and what data it processes.

## Why it matters
With open weight models, you control cost, privacy, and the data a model learns from, instead of renting all of that from a vendor. That control grows more valuable as training data gets scarcer: More sites are blocking AI crawlers and filing lawsuits against model developers over compensation. Hallucination stays your problem on any model, so being able to test and fine-tune one you host is real leverage. Say a mid-market B2B SaaS company hosts an open weight model to classify 50,000 support tickets a month. It cuts per-task cost by roughly 80% against a metered API and keeps customer data on its own servers.

## How to use this knowledge
Reserve hosted open weight models for high-volume, repeatable jobs like classification, extraction, and summarization, where metered API cost or data privacy is the blocker. Fine-tune on proprietary data you would never send to an outside provider. Run your own hallucination check before you trust output in anything customer-facing, since legal queries alone have shown error rates up to 88%. Watch which models keep access to fresh, high-quality training data as more publishers block crawlers.

## Growth Memo guidance
> Training data and computing power are two of the biggest challenges for LLM developers.
> — [#6 - The state of generative AI for SEO (PT.2)](https://www.growth-memo.com/p/the-state-of-generative-ai-pt2)

> In 2025, LLMs have reached the mainstream.
> — [AI Halftime Report H1 2025](https://www.growth-memo.com/p/ai-halftime-report-h1-2025)

## Related concepts
- **Open source large language model (LLM)** — a broader release model where code and sometimes training data are shared, not only the weights.
- **AI tokens** — the units an open weight model processes and bills on, which set the cost of running one yourself.
- **Hallucination** — the convincing but false output you still guard against on a model you host.
- **Training data** — the scarce, contested input that decides how capable any released model can be.

## Referenced in these Growth Memos
- [#6 - The state of generative AI for SEO (PT.2)](https://www.growth-memo.com/p/the-state-of-generative-ai-pt2)
- [AI Halftime Report H1 2025](https://www.growth-memo.com/p/ai-halftime-report-h1-2025)
- [Labeled](https://www.growth-memo.com/p/labeled)

---

# Open source large language model (LLM)

**Suggested section**
Foundations

**Subtitle**
An open source LLM lets you audit how a model was built and adapt it, rather than trusting a model you can only reach through an API.

**Meta title**
What is an open source LLM?

**Meta description**
An open source LLM releases its model code, and often its weights and training details, under a license that lets you inspect, run, and change it.

## What it means
An open source large language model releases its underlying components, the model code, and often the weights and training details, under a license that lets you inspect, run, and change them. The term overlaps with open weight models in everyday use, though open source implies access to more than the trained weights alone.

## Why it matters
An open source LLM lets you audit how a model was built and adapt it, rather than trusting a model you can only reach through an API. That transparency matters most where hallucination carries real cost, because you can test and harden the model against your own cases before it ships. Open source options also reduce vendor lock-in and set a floor on price. Say a DTC skincare brand runs an open source model in-house to draft 2,000 product descriptions a week; it keeps formulation data private and drops per-description cost close to zero after setup.

## How to use this knowledge
Choose open source models when you need to audit behavior for compliance, safety, or reproducibility, beyond the savings on API fees. Fine-tune on your own data and keep that data inside your environment. Build an evaluation set that reflects your real queries, and measure hallucination on it before launch; a Stanford study found legal-query error rates between 69% and 88%. Weigh the engineering cost of hosting and maintaining a model against the control you gain.

## Growth Memo guidance
> The key point is to not trust LLM output blindly, just like we shouldn't trust everything we read on the internet.
> — [#6 - The state of generative AI for SEO (PT.2)](https://www.growth-memo.com/p/the-state-of-generative-ai-pt2)

> For legal queries, a Stanford study found hallucination rates between 69% and 88%.
> — [#6 - The state of generative AI for SEO (PT.2)](https://www.growth-memo.com/p/the-state-of-generative-ai-pt2)

## Related concepts
- **Open weight large language model (LLM)** — the narrower case where the weights are public but the full build may not be.
- **Hallucination** — the failure mode you audit for, and a main reason to want a model you can inspect.
- **AI tokens** — the processing and billing unit that shapes what self-hosting actually saves you.
- **Training data** — the input whose quality and licensing an open source release makes easier to reason about.

## Referenced in these Growth Memos
- [#6 - The state of generative AI for SEO (PT.2)](https://www.growth-memo.com/p/the-state-of-generative-ai-pt2)

---

# AI mention (also referred to as brand mention)

**Suggested section**
SEO

**Subtitle**
Brand mentions shape what buyers trust before they ever click.

**Meta title**
What is an AI mention?

**Meta description**
An AI mention is any time an AI chatbot or AI search feature names your brand in a generated answer, whether or not it links to you.

## What it means
An AI mention is any time an AI chatbot or AI search feature names your brand in a generated answer, whether or not it links to you. It differs from a citation, which is a linked source. Across many tested metrics, brand search volume is the strongest correlate of how often a brand gets mentioned, at a correlation of .334.

## Why it matters
Brand mentions shape what buyers trust before they ever click. In AI Mode studies, brand recognition and the AI's own framing were the top trust drivers, and most users took the first option the AI surfaced. That makes an AI mention a demand-side asset as much as a visibility metric. Say a mid-market project management tool lifts branded search 25% over a 6-month campaign; if the mention-to-brand-search relationship holds, it should surface in more AI answers for its category and pull ahead of louder but less-known rivals.

## How to use this knowledge
Grow branded search demand through PR, partnerships, and off-site presence, the lever most tied to AI mentions. Build trust signals your audience already recognizes, since recognition and AI framing drive the decision in AI Mode. Track mentions by prompt type, such as shopping, comparison, or reputation, and play the role AI assigns your brand instead of fighting for prompts you cannot win. Pair owned assets with third-party coverage, because your own site explains only about 15% of why you get mentioned.

## Growth Memo guidance
> The number of AI Chatbot mentions and brand search volume have a correlation of .334.
> — [What content works well in LLMs?](https://www.growth-memo.com/p/what-content-works-well-in-llms)

> AI framing (how the AI worded its description of a product) and brand recognition were the top 2 trust drivers.
> — [How consumers navigate high-stakes purchases in AI Mode](https://www.growth-memo.com/p/how-consumers-navigate-high-stakes)

> The content on your website accounts for roughly 15% of why you get mentioned or cited in AI responses.
> — [Growth Intelligence Brief #13](https://www.growth-memo.com/p/growth-intelligence-brief-13)

## Related concepts
- **AI citation (also referred to as a brand citation)** — the linked version of appearing in an AI answer, where a mention comes with a source URL.
- **Brand search volume** — the demand signal most correlated with how often AI names your brand.
- **Brand authority** — the trust measure that decides whether users accept an AI answer that names you.
- **AI Mode** — Google's generative search surface where mentions and framing drive most buyer decisions.

## Referenced in these Growth Memos
- [Growth Intelligence Brief #6](https://www.growth-memo.com/p/growth-intelligence-brief-6)
- [Google’s AI Mode SEO Impact | AI Mode User Behavior Study Part 2](https://www.growth-memo.com/p/googles-ai-mode-seo-impact-ai-mode)
- [What content works well in LLMs?](https://www.growth-memo.com/p/what-content-works-well-in-llms)
- [Growth Intelligence Brief #13](https://www.growth-memo.com/p/growth-intelligence-brief-13)
- [How you can track Brand Authority for AI Search](https://www.growth-memo.com/p/how-you-can-track-brand-authority)
- [How consumers navigate high-stakes purchases in AI Mode](https://www.growth-memo.com/p/how-consumers-navigate-high-stakes)
- [Trust Still Lives in Blue Links](https://www.growth-memo.com/p/trust-still-lives-in-blue-links)
- [What Our AI Mode User Behavior Study Reveals about the Future of Search](https://www.growth-memo.com/p/what-our-ai-mode-user-behavior-study)

---

# AI citation (also referred to as a brand citation)

**Suggested section**
SEO

**Subtitle**
A citation is the version of AI visibility that sends traffic and lends authority, because it carries a link back to you.

**Meta title**
What is an AI citation?

**Meta description**
An AI citation is a linked source an AI answer points to when it uses your content, the linked form of a plain brand mention.

## What it means
An AI citation is a linked source that an AI answer points to when it uses your content, the linked form of a plain mention. In AI Overviews (AIOs), ranking means being cited. Analysis of 18,012 verified ChatGPT citations found that 44.2% come from the first 30% of a page.

## Why it matters
A citation is the version of AI visibility that sends traffic and lends authority, because it carries a link back to you. Where AIOs cite is predictable: They read hardest in the first third of a page and rarely past the halfway point. Placement, on top of publishing, decides whether your data gets picked up. Say a mid-market fintech buries its rate comparison table near the bottom of a 3,000-word explainer; moving that table into the first 30% could roughly double how often AI answers cite it, since finance pages land 43.7% of citations in that opening band.

## How to use this knowledge
Put your most citable claims and original data in the first 30% of the page, in every vertical. Frame comparisons explicitly and box your methodology, because AI reaches for benchmarks and citable numbers on "which is best" prompts. Keep the URL stable and the page live, so the source you earn citations from does not move. Target length by vertical and intent instead of a single word count, since very short pages under 1,000 words underperform everywhere.

## Growth Memo guidance
> When we talk about "ranking in AIOs", we mean being cited.
> — [The impact of AI Overviews on SEO - analysis of 19 studies](https://www.growth-memo.com/p/the-impact-of-ai-overviews-on-seo)

> 44.2% of all citations come from the first 30% of a page.
> — [Why proprietary data is your most defensible AI citation asset](https://www.growth-memo.com/p/why-proprietary-data-is-your-most)

> Put your most citable claims and data in the first 30% of the page.
> — [The science of how AI picks its sources](https://www.growth-memo.com/p/the-science-of-how-ai-picks-its-sources)

## Related concepts
- **AI mention (also referred to as brand mention)** — the unlinked form, where AI names you without a source link.
- **Ghost citation** — the case where AI leans on your content but attaches no visible citation.
- **Proprietary data** — the most defensible asset for earning citations, since AI reaches for original benchmarks.
- **AIOs** — the Google surface where being cited is what ranking actually means.

## Referenced in these Growth Memos
- [The impact of AI Overviews on SEO - analysis of 19 studies](https://www.growth-memo.com/p/the-impact-of-ai-overviews-on-seo)
- [Topics matter for third-party authority signals](https://www.growth-memo.com/p/topics-matter-for-third-party-authority)
- [Why most original data never gets cited](https://www.growth-memo.com/p/why-most-original-data-never-gets)
- [AI on Innovation](https://www.growth-memo.com/p/ai-on-innovation)
- [The science of how AI picks its sources](https://www.growth-memo.com/p/the-science-of-how-ai-picks-its-sources)
- [Why proprietary data is your most defensible AI citation asset](https://www.growth-memo.com/p/why-proprietary-data-is-your-most)

---

# Ghost citation

**Suggested section**
AI Research

**Subtitle**
Ghost citations hide much of your real contribution to AI answers, which makes AI visibility hard to measure and easy to undercount.

**Meta title**
What is a ghost citation?

**Meta description**
A ghost citation is when an AI answer draws on your content but shows no visible link or source for it, the gap between influence and attribution.

## What it means
A ghost citation is when an AI answer draws on your content but shows no visible link or source for it. The output is shaped by what the model learned or read, yet you get no clickable credit. It is the gap between influence and attribution.

## Why it matters
Ghost citations hide much of your real contribution to AI answers, which makes AI visibility hard to measure and easy to undercount. If you only track linked citations, you miss the pull your content has when a model uses it silently. That gap also concentrates visible credit in a small set of domains. Say a specialist B2B research firm sees its benchmark numbers repeated across AI answers with no link; its data does the work while a few large domains collect the visible citations, so its dashboard shows a fraction of its true influence.

## How to use this knowledge
Treat linked citations as a floor on your AI visibility, not the whole picture. Watch for your claims, phrases, and data showing up in AI answers without links, and log them as ghost citations. Make your content the easiest version to attribute, with original data, stable URLs, and clear claims in the first third of the page, so models have a reason to link rather than absorb silently. Since about 30 domains own most citations in any topic, invest in the authority and distribution that move you into that set.

## Growth Memo guidance
> ~30 domains own 67% of citations in any topic.
> — [The science of how AI picks its sources](https://www.growth-memo.com/p/the-science-of-how-ai-picks-its-sources)

> I analyzed 1.2 million ChatGPT responses to understand exactly how AI reads a page.
> — [The science of how AI picks its sources](https://www.growth-memo.com/p/the-science-of-how-ai-picks-its-sources)

## Related concepts
- **AI citation (also referred to as a brand citation)** — the visible, linked credit that a ghost citation lacks.
- **AI mention (also referred to as brand mention)** — a named but often unlinked appearance, one step closer to visible than a ghost citation.
- **Attribution** — the link between a source and the answer it shaped, which ghost citations break.
- **Citation concentration** — the pattern where about 30 domains capture most visible citations in a topic.

## Referenced in these Growth Memos
- [The science of how AI picks its sources](https://www.growth-memo.com/p/the-science-of-how-ai-picks-its-sources)

---

# AI tokens

**Suggested section**
Foundations

**Subtitle**
Token cost is what turns AI from a novelty into a line item you can plan around.

**Meta title**
What are AI tokens?

**Meta description**
AI tokens are the pieces of text a language model reads and generates, and the unit AI providers bill on, so token counts set what a task costs.

## What it means
AI tokens are the small pieces of text, roughly words or word fragments, that a language model reads and generates. They are also the unit AI providers bill on, so token counts set what a task costs. Cheap tokens are what let you run large jobs, like categorizing tens of thousands of keywords, for a few dollars.

## Why it matters
Token cost is what turns AI from a novelty into a line item you can plan around. When a task that used to take a day drops to an hour and about $20, the economics of content and analysis change. Cheap tokens push the value of AI toward volume work, while the scarce, human 10% moves up in worth. Say a mid-market ecommerce team generates 5,000 product descriptions for under $50 in tokens; the spend barely registers, so the real question becomes which 10% a person should still handle.

## How to use this knowledge
Price AI work in tokens before you commit, since input plus output tokens times the model's rate gives you the real per-task cost. Send high-volume, repeatable jobs to cheaper models and save premium models for the final quality pass. Cut token waste by trimming prompts, batching inputs, and reusing outputs, because every token is billed. Reinvest the savings into the human 10% that makes content worth reading.

## Growth Memo guidance
> This week, I categorized almost 20,000 keywords into 8 core topics for a client and paid less than $20 in one hour.
> — [Aifficiency](https://www.growth-memo.com/p/aifficiency)

> AI is excellent for drafting up to 90% of content, but the final 10%, which makes it unique, should be human-led.
> — [Live Session October '24 - recap](https://www.growth-memo.com/p/live-session-october-24-recap)

## Related concepts
- **Open weight large language model (LLM)** — a self-hosted model where token cost becomes your own compute cost instead of an API fee.
- **Aifficiency** — the incremental speed and cost gains from AI that cheap tokens make possible.
- **Prompt engineering** — shaping inputs to control how many tokens a task burns.
- **Hallucination** — the quality risk that limits how much of a token-cheap draft you can ship untouched.

## Referenced in these Growth Memos
- [Aifficiency](https://www.growth-memo.com/p/aifficiency)
- [Live Session October '24 - recap](https://www.growth-memo.com/p/live-session-october-24-recap)

---

# Agentic commerce (also referred to as agentic shopping)

**Suggested section**
AI Research

**Subtitle**
Agentic commerce turns organic search from a source of cheap traffic into the mandatory gatekeeper of AI verification.

**Meta title**
What is agentic commerce?

**Meta description**
Agentic commerce is when AI agents complete shopping tasks like inventory checks, cart management, and checkout on AI surfaces instead of your website.

## What it means
Agentic commerce is the practice of AI agents completing shopping tasks like real-time inventory checks, cart management, and checkout directly on AI surfaces such as AI Mode and Gemini, instead of sending you to a website to browse and buy. Kevin frames it as a shift where your website matters less as a destination and more as a database that machines read. The label oversells the near term: Kevin argues that fully autonomous purchasing, handing an agent a credit card and a monthly allowance, is not arriving soon.

## Why it matters
Agentic commerce turns organic search from a source of cheap traffic into the mandatory gatekeeper of AI verification. Marketing arbitrage dies, and accurate product data decides whether an agent surfaces you at all. When the interface is a machine reading feeds, polished copy and clever campaigns count for less than whether your price, stock, and returns are exposed and correct. Imagine a mid-market DTC cookware brand whose product pages read beautifully for humans, but whose shipping speed and stock status sit only in rendered HTML. A shopping agent comparing 5 pans skips it, because it cannot verify availability, and routes the sale to a competitor whose feed exposes the same data cleanly. The brand's conversion from AI surfaces stays near 0% while the competitor captures the sale.

## How to use this knowledge
Audit which product attributes (price, stock, shipping speed, returns) an agent can actually read. If they live only in rendered HTML, you are invisible at the moment of purchase. Shift investment from landing page design for human eyes toward clean, accurate feeds built for machine ingestion. Make product truth a priority: Since agents verify claims against live data, any gap between what you say and what you can prove costs you the sale.

## Growth Memo guidance
> Agentic commerce transforms organic search from a source of cheap traffic into the mandatory gatekeeper of AI verification. Marketing arbitrage dies; product truth wins.
> — [How do you compete in Agentic Commerce?](https://www.growth-memo.com/p/how-do-you-compete-in-agentic-commerce)

> The phrasing "agentic commerce" sets the wrong expectation. Autonomous purchasing, where you give an agent a credit card and monthly allowance to buy on your behalf, is not becoming a reality in the near future.
> — [How do you compete in Agentic Commerce?](https://www.growth-memo.com/p/how-do-you-compete-in-agentic-commerce)

## Related concepts
- **Agentic commerce protocols (ACP, UCP)** — the technical standards that let agents check inventory and check out across AI surfaces.
- **AI verification** — the step where an agent confirms product truth against real-time data before recommending you.
- **Data feeds** — the machine-readable product data agents ingest instead of reading your landing pages.
- **Marketplace SEO** — optimizing a site by its inventory so demand lands when a buyer is deciding.

## Referenced in these Growth Memos
- [How do you compete in Agentic Commerce?](https://www.growth-memo.com/p/how-do-you-compete-in-agentic-commerce)
- [Growth Intelligence Brief #13](https://www.growth-memo.com/p/growth-intelligence-brief-13)
- [Growth Intelligence Brief #10](https://www.growth-memo.com/p/growth-intelligence-brief-10)

---

# Agentic commerce protocols (ACP, UCP)

**Suggested section**
AI Research

**Subtitle**
The shift from crawling to protocols collapses the legacy 14-click funnel into just 2 interactions.

**Meta title**
What are agentic commerce protocols (ACP, UCP)?

**Meta description**
Agentic commerce protocols are open standards, like Google's UCP, that let AI agents check inventory, manage carts, and check out across AI surfaces.

## What it means
Agentic commerce protocols are open standards that let AI agents complete shopping tasks like real-time inventory checks, cart management, and secure checkout across AI surfaces. Google's Universal Commerce Protocol (UCP) is one such standard, and it enables "Business Agents" that act as virtual sales associates directly on the SERP. OpenAI's agentic commerce protocol (ACP) sits behind "Buy it in ChatGPT." Kevin describes these protocols as narrowing where you can compete while still leaving room for marketing.

## Why it matters
The shift from crawling to protocols collapses the legacy 14-click funnel of search, browse, click, and checkout into just 2 interactions: The model parses intent by matching expert reviews against real-time inventory, then you buy in a single click. That compression moves the decision away from your landing page and onto your data. If your shipping speed, inventory status, or return policy is not accessible via API, you are invisible to the agent. Say a mid-market outdoor gear retailer supports UCP so its live stock and shipping data flow to an agent. When a shopper asks an assistant for a 2-person tent in stock under $300, the agent matches reviews against that retailer's real-time inventory and completes checkout in one click, turning a multi-step browse into 2 interactions while a rival with no feed never enters the comparison.

## How to use this knowledge
Check whether your commerce stack can support the emerging protocols (UCP, ACP) and expose price, stock, shipping speed, and returns through an API. Optimize those feeds for machine ingestion with the same rigor you once gave landing page design. Because the funnel now depends on the model matching reviews against live inventory, keep your product specifications accurate and your review data structured so you win that match.

## Growth Memo guidance
> Google introduced the Universal Commerce Protocol (UCP), an open-source standard that allows AI agents to perform complex shopping tasks like real-time inventory checks, cart management, and secure checkout across different surfaces.
> — [Growth Intelligence Brief #13](https://www.growth-memo.com/p/growth-intelligence-brief-13)

> If your shipping speed, inventory status, or return policy isn't accessible via API, you are invisible to the agent.
> — [How do you compete in Agentic Commerce?](https://www.growth-memo.com/p/how-do-you-compete-in-agentic-commerce)

## Related concepts
- **Agentic commerce** — the broader shift where agents shop on your behalf, which these protocols make operational.
- **AI verification** — the step where an agent confirms product truth against real-time data, standardized by the protocols.
- **Data feeds** — the machine-readable product data the protocols transmit, replacing the landing page as the interface.
- **Business Agents** — virtual sales associates that act directly on the SERP once a protocol like UCP is in place.

## Referenced in these Growth Memos
- [How do you compete in Agentic Commerce?](https://www.growth-memo.com/p/how-do-you-compete-in-agentic-commerce)
- [Growth Intelligence Brief #13](https://www.growth-memo.com/p/growth-intelligence-brief-13)

---

# Inventory indexation

**Suggested section**
SEO

**Subtitle**
You don't always need more pages to grow; often you win by controlling which existing pages get indexed.

**Meta title**
What is inventory indexation?

**Meta description**
Inventory indexation is the process of deciding which product or listing pages get crawled and included in search indexes, and which stay out.

## What it means
Inventory indexation is the process of deciding which of your product or listing pages should be crawled and included in search indexes, and which should be left out. For sites with large catalogs, like marketplaces and e-commerce stores, not every page earns a place in the index. Google indexes only about a third of the pages it finds, so choosing which pages you submit protects crawl budget and keeps thin or expired listings from diluting quality.

## Why it matters
You don't always need more pages to grow; often you win by controlling which existing pages get indexed. On a large catalog, letting every listing into the index spends crawl budget on dead pages and slows discovery of the ones that convert. Imagine a mid-market marketplace with 500,000 listings, half of them expired. When all of them stay indexable, Googlebot wastes crawl budget on dead inventory and new, high-demand listings get discovered slowly. After the team noindexes expired pages and submits only active, in-stock URLs, time-to-index for new listings drops from weeks to days.

## How to use this knowledge
Segment your page inventory by demand and quality, then noindex or consolidate the thin, expired, and duplicate pages so crawl attention flows to pages that earn it. Use XML sitemaps and, where available, indexing APIs to push the pages you want indexed rather than waiting to be crawled. Monitor coverage over time: Google indexes only about a third of what it finds and averages 27 days to index a page unless the domain is highly authoritative, so track which listings make it in and which drop out.

## Growth Memo guidance
> You do not always need to add more pages to grow. Often, you can improve your existing page inventory, but you need a monitoring system to figure this out in the first place.
> — [SEOzempic](https://www.growth-memo.com/p/seozempic)

> Google is very choosy and indexes only 1/3 of the pages it finds, but since 2023 it has been indexing more and more.
> — [March news](https://www.growth-memo.com/p/march-25-trends-and-news-roundup)

## Related concepts
- **Crawl budget** — the finite attention search engines spend on your site, which inventory indexation protects.
- **Index selection** — Google's choice to include only a share of the pages it discovers.
- **Indexing APIs and IndexNow** — the push model where you submit the pages you want indexed instead of waiting for a crawl.
- **Marketplace SEO** — optimizing a site by its inventory, where indexation decisions scale to millions of pages.
- **Page inventory** — the full set of pages you could index, which you prune and prioritize.

## Referenced in these Growth Memos
- [15 experts reveal the trends for Technical SEO in 2020](https://www.growth-memo.com/p/15-experts-reveal-the-trends-for-technical-seo-in-2020)
- [IndexNow and the future of web crawling](https://www.growth-memo.com/p/indexnow-and-the-future-of-web-crawling)
- [March news](https://www.growth-memo.com/p/march-25-trends-and-news-roundup)
- [SEOzempic](https://www.growth-memo.com/p/seozempic)
- [The end of crawling and the beginning of API indexing](https://www.growth-memo.com/p/the-end-of-crawling-and-the-beginning-of-api-indexing)
- [Google's index is smaller than we think - and might not grow at all](https://www.growth-memo.com/p/googles-index-is-smaller-than-we-think-and-might-not-grow-at-all)
- [Google's latest updates leave no room for low-quality content](https://www.growth-memo.com/p/googles-latest-updates)
- [Effective Marketplace SEO is more like Product Growth](https://www.growth-memo.com/p/effective-marketplace-seo-is-more)

---

# LLM citation

**Suggested section**
AI Research

**Subtitle**
Citation rates stay low even when models read relevant sources: Gemini shows no clickable citation in 92% of answers.

**Meta title**
What is an LLM citation?

**Meta description**
An LLM citation is the inline source attribution an AI engine shows next to a generated answer, crediting the pages it pulled in to produce it.

## What it means
An LLM citation is the inline source attribution an AI engine shows next to a generated answer. It identifies the pages the retriever pulled in and the generator chose to credit. A model can only cite what it actually reads at answer time, so live retrieval, which attaches a URL, is what makes attribution possible. Anything a model only absorbed during pre-training usually goes uncredited.

## Why it matters
Citations are how you earn visibility and referral clicks in AI search, and the rates stay low even when models read relevant sources: Gemini shows no clickable citation in 92% of answers, and Perplexity cites only 3 to 4 of the roughly 10 pages it visits per query. To get cited you first have to enter the candidate pool, which for many engines means ranking in Google's top 10 for the query. Say a mid-market B2B analytics vendor publishes a strong benchmark study, but it ranks on page 2 for the questions buyers actually ask. It never enters the candidate pool, so the engine cites a thinner competitor post that ranks top 5. After the vendor earns top-10 rankings for 30 question-based variations, its citation share in AI answers climbs from near 0 to a steady presence, and referral clicks follow.

## How to use this knowledge
Rank in Google's top 10 for many fan-out and question variations around your topic, not only head terms, because pages that rank for varied long-tail queries have a higher citation probability. Make each page easy to lift and credit: Put the answer up front, keep structure clean, and hold the URL stable. Track citations by engine and expect low rates and heavy concentration, then diversify the queries and engines you target rather than relying on any single source.

## Growth Memo guidance
> Many LLMs use search engines as retrieval sources. Higher organic rankings increase the probability of entering the LLM's candidate pool and receiving citations.
> — [State of AI Search Optimization 2026](https://www.growth-memo.com/p/state-of-ai-search-optimization-2026)

> The router favors non-reasoning models, which show fewer citations and send less traffic out.
> — [LLM traffic is shrinking](https://www.growth-memo.com/p/llm-traffic-is-shrinking)

## Related concepts
- **Candidate pool** — the set of pages an LLM considers before choosing what to cite; top-10 rankings raise your odds of entering it.
- **Retrieval (RAG)** — the fetch step that adds a URL and makes attribution possible.
- **Grounding** — anchoring an answer in retrieved pages, which varies by model and decides whether you get credited.
- **LLM referral traffic** — the clicks a citation sends, which shrink when models cite less.

## Referenced in these Growth Memos
- [State of AI Search Optimization 2026](https://www.growth-memo.com/p/state-of-ai-search-optimization-2026)
- [LLM traffic is shrinking](https://www.growth-memo.com/p/llm-traffic-is-shrinking)
- [How much can we influence AI responses?](https://www.growth-memo.com/p/how-much-can-we-influence-ai-responses)

---

# LLM experience gain

**Suggested section**
AI Research

**Subtitle**
As AI engines summarize your static pages, an interactive on-page experience is one of the few things a summary can't replace.

**Meta title**
What is LLM experience gain?

**Meta description**
LLM experience gain is the practice of building interactive AI features into your site so visitors get a better experience than static pages give.

## What it means
LLM experience gain is the practice of building new, interactive AI features into your content and website so visitors get a better experience than static pages provide. The term comes from Zack Notes. Rather than only optimizing to be cited by outside AI engines, you use LLMs on your own site: Answer widgets, guided tools, or on-demand explanations that respond to what a visitor asks.

## Why it matters
As AI engines summarize your static pages, an interactive on-page experience is one of the few things a summary can't replace. Kevin's argument is that you should plan content quality for both LLMs and actual humans, auditing thin boilerplate and enriching it with real user data. An AI feature turns a page a visitor would skim, or let an engine summarize for them, into something they act on instead. Imagine a DTC skincare brand whose ingredient pages are dense text most shoppers skim. It adds a feature that lets a shopper enter their skin concern and get a routine built from the brand's own product data. Time on page and add-to-cart rate rise, and the tool delivers a first-hand interaction a generic summary cannot reproduce.

## How to use this knowledge
Find the pages where visitors bounce or where an AI summary would fully substitute for your content, and treat those as candidates for an interactive feature. Ground each feature in data only you hold, like product specifications, usage data, or original research, so the experience is not something a competitor or an engine can copy. Plan quality for both machines and people: Audit low-value boilerplate and replace it with real user data and visualizations. Measure the lift in engagement and conversion, not rankings alone.

## Growth Memo guidance
> Plan content quality for both LLMs and actual humans, and enrich low-value boilerplate with real user data and visualizations.
> — [Effective Marketplace SEO is more like Product Growth](https://www.growth-memo.com/p/effective-marketplace-seo-is-more)

> When your site deeply addresses a topic, you not only become more useful to your audience, but you also are more visible to search engines and LLMs.
> — [Operationalizing your topic-first SEO strategy](https://www.growth-memo.com/p/operationalizing-your-topic-first)

## Related concepts
- **Answer Engine Optimization (AEO)** — optimizing to be surfaced by outside AI engines, the counterpart to building AI experiences on your own site.
- **Content differentiation** — original data or interaction an AI summary can't reproduce.
- **Topical authority** — deep topic coverage that makes your brand more useful to readers and more visible to LLMs.
- **Engagement metrics** — time on page and task completion, the signals an on-site AI experience aims to lift.

## Referenced in these Growth Memos
- [Effective Marketplace SEO is more like Product Growth](https://www.growth-memo.com/p/effective-marketplace-seo-is-more)
- [Operationalizing your topic-first SEO strategy](https://www.growth-memo.com/p/operationalizing-your-topic-first)

---

# LLM leaderboards

**Suggested section**
AI Research

**Subtitle**
Your model choice shapes the quality, cost, and speed of every AI feature you ship, and leaderboards give you a shortlist.

**Meta title**
What are LLM leaderboards?

**Meta description**
LLM leaderboards are public rankings that compare AI models on standardized benchmarks so you can choose which model to build with.

## What it means
LLM leaderboards are public rankings that compare AI models on standardized benchmarks and head-to-head evaluations, so you can see which model performs best on tasks like reasoning, coding, or retrieval. Builders use them to decide which model should power a feature or product. Some score models on shared academic tests; others rank them by blind user preference votes on real prompts.

## Why it matters
Your model choice shapes the quality, cost, and speed of every AI feature you ship, and leaderboards give you a shortlist before you run your own tests. Models differ enough that the right pick changes what your product can do and what it costs to run. For a company in growth mode building an LLM into its product, a leaderboard is the fastest way to narrow dozens of options down to a few worth evaluating. Say a mid-market SaaS team is adding an AI support assistant. Instead of defaulting to the best-known model, they check leaderboards for the top performers on retrieval and instruction-following, shortlist 3, and test them on real tickets. They find a cheaper model resolves 85% of the tickets the flagship handles, at roughly a third of the cost, and ship that one.

## How to use this knowledge
Use leaderboards to build a shortlist matched to the task you are solving, whether that is reasoning, coding, or cost-sensitive retrieval, rather than defaulting to the single top overall model. Treat the ranking as a starting point and validate it with your own evaluations on your real data, since benchmark scores rarely match your exact use case. Re-check before any major model commitment, because rankings shift as new models ship. Weigh cost and latency next to quality, since the top-ranked model is often the wrong production choice.

## Growth Memo guidance
> Many LLMs use search engines as retrieval sources. Higher organic rankings increase the probability of entering the LLM's candidate pool and receiving citations.
> — [State of AI Search Optimization 2026](https://www.growth-memo.com/p/state-of-ai-search-optimization-2026)

## Related concepts
- **Model evaluation (evals)** — your own tests on real data that confirm what a leaderboard only suggests.
- **Benchmark** — the standardized task a leaderboard scores models against.
- **LLM experience gain** — building AI features into your site, where the model you pick from a leaderboard shows up.
- **Cost per token and latency** — production constraints leaderboards rarely capture, which often decide the real pick.

## Referenced in these Growth Memos
- [State of AI Search Optimization 2026](https://www.growth-memo.com/p/state-of-ai-search-optimization-2026)

---

# LLMs

**Suggested section**
AI Research

**Subtitle**
Because an LLM is frozen at training, every "AI SEO" tactic that tries to change the model itself is misdirected effort.

**Meta title**
What are LLMs?

**Meta description**
LLMs, or large language models, are the systems behind tools like ChatGPT, with knowledge fixed at training and updated only through retrieval.

## What it means
LLMs, or large language models, are the systems that generate answers in tools like ChatGPT and Google's AI Overviews (AIOs). Their knowledge is fixed at the moment of training, so a model cannot learn anything new about your brand after its cutoff. What an LLM says about you at query time depends on what gets retrieved and fed into the prompt, rather than on what the model already memorized.

## Why it matters
Because an LLM is frozen at training, every "AI SEO" tactic that tries to change the model itself is misdirected effort. Many LLMs pull from search engines as retrieval sources, so your leverage sits in ranking for the queries that trigger retrieval. Say a mid-market B2B SaaS company spends a quarter publishing keyword-stuffed pages about itself to "teach" ChatGPT to recommend it. Nothing moves, because the model was frozen months before those pages existed. When the same team instead earns top-10 rankings for 40 question-based variations of its category, its citation rate in AI answers climbs from near zero to a measurable share, because retrieval now surfaces its pages.

## How to use this knowledge
Stop optimizing for the model and start optimizing for retrieval: Rank in the top 10 for fan-out query variations around your core topics, reaching beyond head terms. Strengthen the entity relationships in your content through strategic internal linking, since search engines and LLMs judge authority by how entities connect. Skip hacky LLM visibility tricks and invest in durable topical authority, because grounding differs by model and short-term manipulation does not last.

## Growth Memo guidance
> Many LLMs use search engines as retrieval sources. Higher organic rankings increase the probability of entering the LLM's candidate pool and receiving citations.
> — [State of AI Search Optimization 2026](https://www.growth-memo.com/p/state-of-ai-search-optimization-2026)

> Search engines and LLMs are mapping relationships between entities and judging your brand's authority accordingly.
> — [Internal Linking Grows Up: Evolving from Link Juice to Entity Maps](https://www.growth-memo.com/p/internal-linking-grows-up-evolving)

> Grounding varies from model to model and not all LLMs prioritize pages ranking at the top of Google search.
> — [How much can we influence AI responses?](https://www.growth-memo.com/p/how-much-can-we-influence-ai-responses)

## Related concepts
- **Retrieval-augmented generation (RAG)** — the mechanism that feeds live retrieved pages into an LLM at query time, where your real leverage sits.
- **AIOs** — Google's answers generated by an LLM, drawing on ranked search results as retrieval sources.
- **Topical authority** — the signal LLMs use to decide whether your brand deserves a citation on a topic.
- **Fan-out queries** — the varied reformulations an LLM issues, which you rank for to enter its candidate pool.

## Referenced in these Growth Memos
- [State of AI Search Optimization 2026](https://www.growth-memo.com/p/state-of-ai-search-optimization-2026)
- [Internal Linking Grows Up: Evolving from Link Juice to Entity Maps](https://www.growth-memo.com/p/internal-linking-grows-up-evolving)
- [How much can we influence AI responses?](https://www.growth-memo.com/p/how-much-can-we-influence-ai-responses)
- [Operationalizing your topic-first SEO strategy](https://www.growth-memo.com/p/operationalizing-your-topic-first)

---

# Long-tail queries

**Suggested section**
SEO

**Subtitle**
Users are searching with more words and full questions, so long-tail queries make up a growing share of demand.

**Meta title**
What are long-tail queries?

**Meta description**
Long-tail queries are specific, lower-volume searches of several words that together add up to a market bigger than head terms.

## What it means
Long-tail queries are specific, lower-volume searches made up of more words than head terms. Each one gets few searches on its own, but combined they add up to a market bigger than the popular head terms. The idea comes from Chris Anderson's 2004 "The Long Tail" essay, which showed that more than half of Amazon's book sales came from titles outside its top 130,000. The internet is vast and easy to search, so it can serve almost any specific need, which is what makes the long tail work.

## Why it matters
Users are searching with more words and full questions, so long-tail queries make up a growing share of demand. Conversational AI tools push this further, because prompts are longer and more varied than the keywords people once typed. Pages that answer many long-tail and question variations signal that you cover a topic completely, which feeds topical authority. Imagine a DTC skincare brand that ranks only for a head term like "vitamin C serum." It captures a thin slice of demand. When it builds pages that answer specific long-tail queries such as "can I use vitamin C serum with retinol at night," it starts capturing dozens of low-volume searches that together drive more qualified traffic than the single head term, lifting non-branded sessions by 40% over 2 quarters.

## How to use this knowledge
Map the long-tail and question variations around each parent topic, then build content that covers the related entities and questions completely. Track rankings for those variations, not only your head terms, since AI prompts are varied and reward depth. Where the volume justifies it, generate pages programmatically so you can cover long-tail demand at scale without writing each page by hand.

## Growth Memo guidance
> The internet is so vast and easy to search that it has content and products for anyone, no matter their taste or need.
> — [Here is the longtail](https://www.growth-memo.com/p/here-is-the-longtail)

> Topical relevance, how completely a site covers related entities and questions, is a real and important factor in ranking.
> — [How to measure topical authority](https://www.growth-memo.com/p/how-to-measure-topical-authority)

## Related concepts
- **Head terms** — the high-volume, generic queries that sit at the opposite end from long-tail queries.
- **Topical authority** — the depth signal you build by covering the entities and questions behind long-tail queries.
- **Fan-out queries** — the many conversational reformulations AI search issues, most of which are long-tail.
- **Programmatic SEO** — the method for producing pages at scale to cover long-tail demand.
- **Search intent** — the specific need behind a long-tail query, usually clearer than a head term's.

## Referenced in these Growth Memos
- [Youtube's hashtag pages and the long-tail](https://www.growth-memo.com/p/youtubes-hashtag-pages-and-the-long-tail)
- [Here is the longtail](https://www.growth-memo.com/p/here-is-the-longtail)
- [How to measure topical authority](https://www.growth-memo.com/p/how-to-measure-topical-authority)

---

# Marketplace SEO

**Suggested section**
SEO

**Subtitle**
For marketplaces, around 80% of new users can come from organic search, which makes SEO the primary growth engine.

**Meta title**
What is Marketplace SEO?

**Meta description**
Marketplace SEO optimizes a two- or three-sided platform around its inventory so buyers land directly on it at the moment of decision.

## What it means
Marketplace SEO is the practice of optimizing a two- or three-sided platform around its inventory (supply) so that buyers (demand) land directly on the marketplace when they are about to make a decision, like booking a flight or scheduling an appointment. It works at the scale of hundreds of thousands or millions of pages generated from that inventory. Kevin treats it as a product-led SEO function of the business, not website optimization, which is why he calls marketplaces "SEO Aggregators."

## Why it matters
For marketplaces, around 80% of new users can come from organic search, which makes SEO the primary growth engine. Their advantage is scale: Every unit of supply can spawn its own page targeting a specific query, so growth is tied to product and inventory decisions rather than editorial output. Say a two-sided home services marketplace connects plumbers with homeowners, and each new provider profile and service-city combination generates its own page. Growing verified supply from 5,000 to 20,000 providers can quadruple indexable pages and lift organic sessions well beyond what content alone could produce, because each page answers a specific "plumber in [city]" query.

## How to use this knowledge
Run marketplace SEO as product growth: Align with product, UX, and supply teams instead of treating it as a content channel. Prioritize technical SEO at scale, since crawl efficiency, page quality, and internal linking decide whether millions of pages get indexed and ranked. Use reviews and ratings to give programmatic pages the trust and relevance that generic templates lack. Because AI threatens the traffic moat, diversify beyond core category and product keywords into new page types and page quality bets.

## Growth Memo guidance
> Good SEO is the result of Product Growth, not just website optimization.
> — [Effective Marketplace SEO is more like Product Growth](https://www.growth-memo.com/p/effective-marketplace-seo-is-more)

> Marketplace SEO is deeply intertwined with product evolution, UX, and brand strategy.
> — [Marketplace Deep Dive - Q2](https://www.growth-memo.com/p/marketplace-deep-dive-q2)

## Related concepts
- **Product-led SEO (PLSEO)** — the broader strategy marketplace SEO belongs to, where pages come from the product and supply rather than writers.
- **SEO Aggregators** — Kevin's term for marketplaces that win through massive programmatic scale.
- **Programmatic SEO** — the technique of generating large volumes of pages from structured inventory data.
- **User-generated content (UGC)** — reviews and ratings that give marketplace pages trust and relevance.
- **Crawl efficiency** — the technical priority when a site holds hundreds of thousands of pages.

## Referenced in these Growth Memos
- [Effective Marketplace SEO is more like Product Growth](https://www.growth-memo.com/p/effective-marketplace-seo-is-more)
- [Marketplace Deep Dive - Q1 (Case Study: Zillow)](https://www.growth-memo.com/p/marketplace-deep-dive-q1-case-study)
- [Marketplace Deep Dive - Q2](https://www.growth-memo.com/p/marketplace-deep-dive-q2)
- [3 examples of product-led SEO](https://www.growth-memo.com/p/5-examples-of-product-led-seo)

---

# Microagents

**Suggested section**
AI Research

**Subtitle**
Microagents let a small growth team 10x output by automating repetitive analysis and chaining narrow agents together.

**Meta title**
What are microagents?

**Meta description**
Microagents are small, single-purpose AI agents you build to automate one narrow task and chain into larger workflows.

## What it means
Microagents are small, single-purpose AI agents you build to handle one narrow task, like extracting data from Google Search Console and generating action items, or scraping and analyzing 2 pages. You connect these simple building blocks to assemble more complex workflows over time. The point is leverage: Custom agents that multiply what a small team can produce.

## Why it matters
Microagents let a small growth team 10x output by automating repetitive analysis and chaining narrow agents together. Rather than build one large, brittle automation, you build narrow agents for specific tasks and connect them, which is easier to debug and reuse. Say a 3-person SEO team spends 6 hours a week pulling Search Console data, flagging pages that dropped, and drafting fix lists. A microagent that extracts the GSC data and generates action items cuts that to 30 minutes, and a second microagent that scrapes and compares the flagged pages hands the team a prioritized queue. Two narrow agents, chained, free up most of a workday every week.

## How to use this knowledge
Start with one simple use case, such as extracting GSC data and generating action items, or scraping and analyzing 2 pages. Build each agent as a narrow building block, then connect the blocks into larger workflows as you gain confidence. Experiment with agent platforms to find where a custom build beats an off-the-shelf tool for your team's specific tasks.

## Growth Memo guidance
> When getting started, define a simple use case. Start with tasks like extracting data from Google Search Console and generating action items. By connecting these simple building blocks, you can create more complex agents over time.
> — [Live Session recap - 1/31/25](https://www.growth-memo.com/p/live-session-recap-13125)

> Agents will likely use Multi-Source Corroboration, where one AI agent retrieves the info and another agent acts as a "judge" to verify it.
> — [10 SEO, marketing, and tech predictions for 2026](https://www.growth-memo.com/p/10-seo-marketing-and-tech-predictions)

## Related concepts
- **AI agents** — the broader category; microagents are the small, narrow-task version you chain together.
- **Multi-Agent Debate** — a setup where one agent retrieves and another verifies, showing agents specializing by role.
- **Agentic Web** — the emerging infrastructure where agents act on the open web.
- **Automation** — the outcome microagents deliver by handling repetitive data and analysis work.

## Referenced in these Growth Memos
- [Live Session recap - 1/31/25](https://www.growth-memo.com/p/live-session-recap-13125)
- [10 SEO, marketing, and tech predictions for 2026](https://www.growth-memo.com/p/10-seo-marketing-and-tech-predictions)

---

# NEG SEO

**Suggested section**
SEO

**Subtitle**
Your rankings can drop for reasons that have nothing to do with your own work.

**Meta title**
What is NEG SEO?

**Meta description**
NEG SEO, or negative SEO, is the practice of deliberately trying to hurt another site's search rankings with spammy links, scraping, or hacking.

## What it means
NEG SEO, short for negative SEO, is the practice of deliberately trying to hurt another site's search rankings. Common tactics include pointing spammy or toxic backlinks at a competitor, scraping and duplicating their content, creating fake profiles or reviews, or hacking pages to inject links. The goal is to trigger the negative signals that search engines penalize.

## Why it matters
Your rankings can drop for reasons that have nothing to do with your own work. Google measures some negative factors against thresholds, so an attack that pushes your site past a limit can cause real damage even when your own content and links are sound. Say a mid-market B2B SaaS company loses 30% of its non-branded rankings in a month, and its team finds thousands of new spammy backlinks from unrelated gambling and adult sites pointing at its money pages. By disavowing the toxic domains and auditing their backlink profile weekly, they halt the damage and recover most positions over the next 2 quarters.

## How to use this knowledge
Monitor your backlink profile continuously and watch for sudden spikes of toxic links, then disavow the worst domains before they accumulate. Track your positions and SERP Features closely so you spot an unexplained drop fast and can diagnose whether it is an attack or an algorithm change. Keep the site secure with patched software and monitoring for scraped content or unauthorized edits. Watch on-site negative signals like 404s and duplicates too, since Google reinforces consequences once they pass a threshold.

## Growth Memo guidance
> Google also seems to measure negative factors with thresholds: A few 404s won't hurt, but after a certain percentage Google seems to reinforce negative consequences.
> — [The 10 SEO ranking factors we know to be true](https://www.growth-memo.com/p/the-10-seo-ranking-factors-we-know-to-be-true)

## Related concepts
- **Backlink profile** — the set of inbound links an attacker tries to poison with toxic domains.
- **Disavow file** — the tool you submit to Google to neutralize spammy links pointing at your site.
- **Toxic links** — low-quality or spammy backlinks that can drag down rankings.
- **Ranking factors** — the positive and negative signals negative SEO tries to manipulate.
- **Technical SEO** — the monitoring and maintenance discipline that keeps your site resilient to attacks.

## Referenced in these Growth Memos
- [The 10 SEO ranking factors we know to be true](https://www.growth-memo.com/p/the-10-seo-ranking-factors-we-know-to-be-true)

---

# NLPs

**Suggested section**
AI Research

**Subtitle**
NLP is why search stopped rewarding exact-match keywords and started rewarding meaning.

**Meta title**
What are NLPs?

**Meta description**
NLP, or natural language processing, is the field of AI that teaches machines to read, interpret, and generate human language.

## What it means
NLP, or natural language processing, is the field of AI that teaches machines to read, interpret, and generate human language. Search engines use NLP to understand what a query means and how well a page answers it, and large language models are built on the same foundation. When people write "NLPs," they usually mean the specific models or systems that do this work.

## Why it matters
NLP is why search stopped rewarding exact-match keywords and started rewarding meaning. It is also the source of generative AI's biggest weakness: Models built on NLP produce fluent text that can be confidently wrong. Imagine a DTC skincare brand that lets an NLP model auto-generate 500 product descriptions, and the model invents an ingredient benefit that is not true. Because these models generate convincing but sometimes false text, one unchecked pass ships a compliance problem across every page. A human review step on generated copy catches it before publication.

## How to use this knowledge
Write for meaning and entities rather than exact-match keywords, since NLP interprets intent instead of matching strings. Structure content so these systems can extract a clear answer: State the question, define the entity, and answer plainly near the top. Never trust model output blindly, and add a human review step for factual claims, especially in regulated topics like health, legal, and finance where hallucination rates run high.

## Growth Memo guidance
> LLMs can make things up in a very convincing way. For legal queries, a Stanford study found hallucination rates between 69% and 88%.
> — [#6 - The state of generative AI for SEO (PT.2)](https://www.growth-memo.com/p/the-state-of-generative-ai-pt2)

> The key point is to not trust LLM output blindly, just like we shouldn't trust everything we read on the internet.
> — [#6 - The state of generative AI for SEO (PT.2)](https://www.growth-memo.com/p/the-state-of-generative-ai-pt2)

## Related concepts
- **Large language models (LLMs)** — generative systems built on NLP that produce human-like text and can hallucinate.
- **Hallucination** — the failure mode where an NLP model generates convincing but false output.
- **Entities** — the real-world things NLP extracts and links to understand meaning.
- **Semantic search** — search that uses NLP to match meaning rather than exact keywords.

## Referenced in these Growth Memos
- [#6 - The state of generative AI for SEO (PT.2)](https://www.growth-memo.com/p/the-state-of-generative-ai-pt2)

---

# PR SEO

**Suggested section**
SEO

**Subtitle**
Offsite signals drive 85% of brand visibility in LLM answers, which turns digital PR into a ranking lever.

**Meta title**
What is PR SEO?

**Meta description**
PR SEO combines public relations with search optimization to build the third-party signals that drive brand visibility in search and AI answers.

## What it means
PR SEO combines public relations with search optimization to build the third-party signals that decide how visible your brand is. Instead of relying only on content you publish, you earn coverage, mentions, and citations on sites that search engines and LLMs already trust. Digital PR is the working core of it: Placing your brand where authority is borrowed rather than self-declared.

## Why it matters
Third-party signals now carry much of how brands surface in AI search. Kevin cites data that offsite signals drive 85% of brand visibility in LLM answers, which turns digital PR into a ranking lever rather than an awareness line item. When you budget SEO by capacity, PR becomes a share you allocate against, next to technical work and content.

Say a mid-market B2B SaaS company shifts 40% of its SEO capacity to digital PR for 2 quarters. It earns mentions on 15 industry sites where it had no presence before, and its citation rate in ChatGPT answers for category prompts climbs from near zero to roughly 25% of responses.

## How to use this knowledge
Allocate capacity to digital PR on purpose: Decide what share of your fixed SEO capacity goes to PR versus technical work and content, and model the visibility you expect from each. Target sources that both search engines and LLMs already cite, prioritizing placements on sites that rank and get pulled into answers for your category prompts. Pair PR with prompt monitoring so you can see where your brand shows up across relevant ChatGPT prompts, then syndicate and repurpose that content to fill the gaps.

## Growth Memo guidance
> Third-party signals drive 85% of brand visibility in LLMs.
> — [Budget SEO for capacity, not output](https://www.growth-memo.com/p/budget-seo-for-capacity-not-output)

> Invest in monitoring and optimizing your visibility across relevant ChatGPT prompts with targeted content, PR campaigns, content syndication, and content repurposing.
> — [How is answer engine optimization different from SEO?](https://www.growth-memo.com/p/is-geoaeo-the-same-as-seo)

## Related concepts
- **Digital PR** — the tactical core of PR SEO, earning third-party mentions and links.
- **Brand authority** — what PR SEO builds, and a close cousin of topical authority in AI search.
- **Answer engine optimization** — the discipline PR SEO feeds, since LLMs lean on offsite signals.
- **Zero-click search** — the setting that makes offsite visibility matter more than raw sessions.

## Referenced in these Growth Memos
- [Budget SEO for capacity, not output](https://www.growth-memo.com/p/budget-seo-for-capacity-not-output)
- [How is answer engine optimization different from SEO?](https://www.growth-memo.com/p/is-geoaeo-the-same-as-seo)

---

# Query relevance

**Suggested section**
AI Research

**Subtitle**
Relevance stopped being about hitting a keyword and became about covering the meaning around it.

**Meta title**
What is query relevance?

**Meta description**
Query relevance is how closely a page's meaning matches a query, measured by semantic similarity between query and content rather than keyword overlap.

## What it means
Query relevance is how closely a page's meaning matches what a query is actually asking. Search engines and LLMs convert both the query and your content into embeddings, then measure the distance between them. The closer your page sits to the intent behind the query, the higher your odds of ranking or being cited. It is search intent, scored by similarity rather than by exact keyword match.

## Why it matters
LLM prompts are conversational and varied, so a single topic splinters into many phrasings. Google's AI Mode takes a prompt and fans it out into a swarm of sub-queries, each aimed at a different part of what the user wants. Pages that answer many long-tail and question-based variations, not only the head term, carry higher citation probability. Relevance stopped being about hitting a keyword and became about covering the meaning around it.

Say a DTC skincare brand runs one page for "vitamin C serum". A competitor also answers "is vitamin C serum good for oily skin", "when to apply vitamin C serum", and "vitamin C serum vs retinol". When AI Mode fans out a prompt on the topic, the competitor matches 4 sub-queries while the brand matches only one, so the competitor earns the citation.

## How to use this knowledge
Map the query variations around each core topic by listing the sub-questions an LLM would generate from your head term, then check which ones your page actually answers. Write for semantic coverage: Address the related entities, questions, and comparisons that sit close to your topic in meaning, so your embeddings land near more queries. Since AI systems pull from search results, aim to rank in the top 10 for those fan-out variations, because broad ranking across long-tail phrasings raises how often you get cited.

## Growth Memo guidance
> Rank in Google's top 10 for fan-out query variations around your core topics, not just head terms. Since LLM prompts are conversational and varied, pages ranking for many long-tail and question-based variations have higher citation probability.
> — [State of AI Search Optimization 2026](https://www.growth-memo.com/p/state-of-ai-search-optimization-2026)

> In AI Mode, Gemini explodes your prompt into a swarm of sub-queries.
> — [Query fan-out](https://www.growth-memo.com/p/query-fan-out)

## Related concepts
- **Search intent** — the user goal query relevance tries to satisfy.
- **Query fan-out** — the mechanism that multiplies one prompt into many relevance checks.
- **Embedding similarity** — the math behind semantic relevance scoring.
- **Topical authority** — relevance across a whole topic instead of a single query.

## Referenced in these Growth Memos
- [State of AI Search Optimization 2026](https://www.growth-memo.com/p/state-of-ai-search-optimization-2026)
- [Query fan-out](https://www.growth-memo.com/p/query-fan-out)
- [AI on Innovation](https://www.growth-memo.com/p/ai-on-innovation)

---

# Query share

**Suggested section**
SEO

**Subtitle**
Query share turns the abstract idea of topical authority into a number you can track.

**Meta title**
What is query share?

**Meta description**
Query share is the portion of total search demand for a topic that your site captures through its rankings, used as a proxy for topical authority.

## What it means
Query share is the portion of total search demand for a topic that your site captures through rankings. Map every query that belongs to a topic, add up the demand behind them, and query share is the slice you actually rank for and win. Kevin treats it as a proxy for topical authority: The more of a topic's queries you cover, the more completely you own that topic in Google's eyes.

## Why it matters
Google rewards sites that cover a topic deeply by comparing how well they cover the relevant entities and questions. Query share turns that abstract idea into a number you can track. It answers the question your boss keeps asking, how do you prove topical authority is real and growing, with evidence instead of assertion. A rising query share across a topic is direct proof you cover more of it than you did before.

Say a mid-market B2B SaaS company maps 400 queries in its core topic and ranks in the top 10 for 60 of them in January. After a content push, it ranks for 180. Query share roughly triples, and its citation rate in AI answers for that topic climbs alongside it.

## How to use this knowledge
Define the topic's query set first, because you cannot measure share without the denominator. Map the full set of queries that belong to the topic, then track what share of it you hold in the top 10 and watch the trend rather than the snapshot. Turn the gaps into a content roadmap: Queries in the set where you do not yet rank become your next briefs, ordered by demand and business value.

## Growth Memo guidance
> Topical relevance, how completely a site covers related entities and questions, is a real and important factor in ranking.
> — [How to measure topical authority](https://www.growth-memo.com/p/how-to-measure-topical-authority)

> Brand authority, a close cousin of topical authority, can be the difference between earning the click or being buried.
> — [How to measure topical authority](https://www.growth-memo.com/p/how-to-measure-topical-authority)

## Related concepts
- **Topical authority** — what query share is a proxy for.
- **Query universe** — the full query set that forms the denominator of query share.
- **Share of voice** — the visibility-weighted cousin of query share across a market.
- **Brand authority** — the close cousin Kevin ties to earning clicks in AI search.

## Referenced in these Growth Memos
- [How to measure topical authority](https://www.growth-memo.com/p/how-to-measure-topical-authority)

---

# Query universe

**Suggested section**
SEO

**Subtitle**
Mapping the query universe is the first step to measuring and optimizing topical authority.

**Meta title**
What is a query universe?

**Meta description**
A query universe is the full set of search queries that relate to a topic, mapped in one place so you can prioritize and cover them over time.

## What it means
A query universe is the full set of search queries that relate to a topic, gathered in one place. Kevin built the idea at Shopify as a keyword universe: A large pool of the language your audience uses to search, held in a spreadsheet or database and sorted so the most important queries rise to the top. You build it once and work through it over time instead of running a fresh keyword sprint every quarter.

## Why it matters
Mapping the query universe is the first step to measuring and optimizing topical authority. You cannot know how much of a topic you cover until you have listed everything the topic contains. Gathering queries is the easy part, since anyone can generate a long list. The value comes from the filters and sorting that decide what you work on first.

Say a DTC skincare brand mines 3,000 queries across its categories into one universe, then sorts by intent and conversion potential instead of raw search volume. It surfaces 200 high-intent queries with no page assigned. That sorted gap becomes a 2-quarter content roadmap instead of another spreadsheet collecting dust.

## How to use this knowledge
Mine broadly, then centralize: Pull seed keywords through rank trackers and fold your old keyword exports into one universe in a spreadsheet or database like BigQuery. Sort by more than volume, since high-volume queries often convert worse or lose traffic to AI Overviews (AIOs), so weight by intent and business value instead. Treat the result as a living prioritization system you keep exploring and conquering across your site over time, rather than a research task you run once a quarter.

## Growth Memo guidance
> A Keyword Universe is a big pool of language your target audience uses when they search that will help them find you.
> — [Keywords are dead. But the Keyword Universe Isn't.](https://www.growth-memo.com/p/universe)

> Anyone can create a large list of keywords, but creating strong filters and sorting mechanisms is hard.
> — [Keywords are dead. But the Keyword Universe Isn't.](https://www.growth-memo.com/p/universe)

> At Shopify, we built the initial concept of a "keyword universe" when I was there between 2020 and 2022.
> — [AI changed my work. And yours, too.](https://www.growth-memo.com/p/ai-changed-my-work-and-yours-too)

## Related concepts
- **Topical authority** — what the query universe lets you measure and grow.
- **Query share** — the portion of the universe you actually rank for.
- **Keyword mining** — the workflow that populates the universe.
- **Search intent** — the sorting dimension that beats raw search volume.

## Referenced in these Growth Memos
- [Keywords are dead. But the Keyword Universe Isn't.](https://www.growth-memo.com/p/universe)
- [Keywords Are Dead But The Keyword](https://www.growth-memo.com/p/keywords-are-dead-but-the-keyword)
- [AI changed my work. And yours, too.](https://www.growth-memo.com/p/ai-changed-my-work-and-yours-too)
- [How to measure topical authority](https://www.growth-memo.com/p/how-to-measure-topical-authority)

---

# Ranking overlap

**Suggested section**
SEO

**Subtitle**
Overlap tells you where authority is shared and where it is winner-take-all.

**Meta title**
What is ranking overlap?

**Meta description**
Ranking overlap is the share of queries or topics where more than one site ranks, used to gauge topical competitiveness and shared authority.

## What it means
Ranking overlap is the share of queries or topics where more than one site ranks. Measured across competitors, it shows how contested a topic is and how concentrated authority is. Measured across search and AI engines, it shows how often the same pages surface in Google, ChatGPT, and Perplexity. Low overlap means each player, or each engine, is working from its own retrieval logic.

## Why it matters
Overlap tells you where authority is shared and where it is winner-take-all. New data on 3.7 million citations shows cross-engine overlap is small and varies by page type: Guides and tutorials overlap most, homepages least. That pushes against the instinct that high-intent queries are where shared authority shows up. Each engine trusts different sources and prefers different formats, and that does most of the work.

Say a mid-market B2B SaaS company checks 500 category queries and finds it ranks in the top 10 for 300, but shares that top 10 with the same 2 competitors on 220 of them. High overlap means the topic is crowded, so you win a citation with a sharper angle rather than more volume.

## How to use this knowledge
Measure overlap before you invest, because heavy competitor overlap on a topic signals a crowded field where you need an angle the incumbents lack. Lead with explanatory content, which travels across engines better than brand or product pages, so guides and tutorials raise your odds of shared citations. Track overlap per engine too, and do not assume a page cited in one AI engine is cited in the others: Check each and close the gaps you find.

## Growth Memo guidance
> Most teams talk about "AI visibility" like it's one thing. New data on 3.7 million citations across ChatGPT, Perplexity, and Google AI Overviews suggests it isn't.
> — [The Consensus Gap](https://www.growth-memo.com/p/the-consensus-gap)

> Guides and tutorials have the highest cross-engine overlap at 2.3%, followed by blogs at 1.8%, category pages at 1.6%, product pages at 1.2%, and homepages at 1.1%.
> — [The Consensus Gap](https://www.growth-memo.com/p/the-consensus-gap)

> Higher organic rankings increase the probability of entering the LLM's candidate pool and receiving citations.
> — [State of AI Search Optimization 2026](https://www.growth-memo.com/p/state-of-ai-search-optimization-2026)

## Related concepts
- **Topical authority** — high overlap on a topic raises the bar for proving it.
- **Query share** — your slice of a topic, which overlap tells you how hard it is to grow.
- **Cross-engine visibility** — the AI-era reading of ranking overlap across ChatGPT, Perplexity, and Google.
- **Differentiation** — the way to win when overlap is high and volume alone will not.

## Referenced in these Growth Memos
- [The Consensus Gap](https://www.growth-memo.com/p/the-consensus-gap)
- [State of AI Search Optimization 2026](https://www.growth-memo.com/p/state-of-ai-search-optimization-2026)
- [How to measure topical authority](https://www.growth-memo.com/p/how-to-measure-topical-authority)

---

# Relevance engineering

**Suggested section**
SEO

**Subtitle**
As search shifts from clicks to citations, the job is getting your point of view into an AI's answer with the authority to be recommended.

**Meta title**
What is relevance engineering?

**Meta description**
Relevance engineering is the practice of shaping how well your brand and content match what search engines and AI systems retrieve, ranked by authority.

## What it means
Relevance engineering is the practice of shaping how well your brand and content match what search engines and AI systems retrieve, judged by authority more than by volume. It reframes SEO around a harder question: When everyone can produce the same content cheaply, what makes a machine choose you? The work moves closer to brand and authority than to keywords and links.

## Why it matters
LLMs erode the advantage of content volume and pressure you to make things they cannot. Google already weighs customer demand, measured through brand search, and customer experience, measured through user signals, more heavily than thin optimization. As search shifts from clicks to citations, the job is getting your data and point of view into an AI's answer with enough authority to be recommended. Kevin argues the SEO label may not survive this shift, and that what replaces it sits closer to brand marketing.

Say a mid-market B2B SaaS company stops measuring only rankings and traffic and starts tracking brand search volume, citations in AI answers, and conversion influence. Over 3 quarters, its raw clicks fall 15% while its citation rate in category prompts doubles and organic pipeline holds steady. Relevance engineering is what moved the second set of numbers.

## How to use this knowledge
Optimize for ingestion by structuring your data and content so an AI can lift them into an answer with your brand attached. Build differentiation machines cannot copy, since original data, first-hand experience, and a clear point of view beat out-executing competitors on content volume. Change what you measure, adding citations, brand mentions, and conversion influence to your reporting through a layering approach so success reflects influence alongside traffic. And keep investing in brand, because Google rewards customer demand and experience, which makes brand search and user signals relevance inputs you can build.

## Growth Memo guidance
> LLMs erode the advantage of volume and create pressure to create content of intense value. The unlock for SEO is figuring out what type of content LLMs cannot create.
> — [Eroding moats](https://www.growth-memo.com/p/eroding-moats)

> SEO shifts from optimizing for clicks to optimizing for ingestion.
> — [How do you compete in Agentic Commerce?](https://www.growth-memo.com/p/how-do-you-compete-in-agentic-commerce)

> With a "new SEO" (SEO 2.0?) comes an opportunity to detach from performance marketing and get closer to brand marketing.
> — [Growth Intelligence Brief #3](https://www.growth-memo.com/p/growth-intelligence-brief-3)

## Related concepts
- **Answer engine optimization** — the AI-search slice of relevance engineering.
- **Topical authority** — the authority half of what relevance engineering builds.
- **Differentiation** — the content edge that keeps you relevant when everyone has the same text.
- **Information gain** — original data and analysis as a relevance signal machines reward.
- **Brand marketing** — where relevance engineering pulls the discipline, per Kevin.

## Referenced in these Growth Memos
- [Eroding moats](https://www.growth-memo.com/p/eroding-moats)
- [How do you compete in Agentic Commerce?](https://www.growth-memo.com/p/how-do-you-compete-in-agentic-commerce)
- [Growth Intelligence Brief #3](https://www.growth-memo.com/p/growth-intelligence-brief-3)
- [Differentiation](https://www.growth-memo.com/p/differentiation)
- [Information Gainz](https://www.growth-memo.com/p/information-gainz)
- [The impact of AI Mode on SEO -analysis of 10 studies](https://www.growth-memo.com/p/the-impact-of-ai-mode-on-seo-analysis)

---

# Retrieval augmented generation (RAG) LLMs

**Suggested section**
AI Research

**Subtitle**
Hallucination is the biggest risk of generative AI and the biggest factor holding back wider adoption.

**Meta title**
What is retrieval augmented generation (RAG)?

**Meta description**
Retrieval augmented generation (RAG) is a method where an LLM retrieves external documents before generating an answer to reduce hallucination.

## What it means
Retrieval augmented generation (RAG) is a method where a large language model (LLM) fetches relevant documents from an external source before it generates an answer, then grounds its response in that retrieved material. Instead of answering only from what it memorized during training, a RAG system pulls in current, specific information and writes the reply from it. The point is to reduce hallucination and tie output to sources you can check.

## Why it matters
Hallucination is the biggest risk of generative AI and the biggest factor holding back wider adoption. LLMs can make things up in a very convincing way, and the failure rate is not small: A Stanford study found hallucination rates between 69% and 88% for legal queries. RAG matters to you because it constrains the model to retrieved evidence, which is how AI search products like AI Overviews (AIOs) and chat assistants keep answers grounded in real pages, including yours. If your page is the source a RAG system retrieves, you get quoted and cited. If it is not retrievable or not trusted, you are absent from the answer.

Say a mid-market B2B SaaS company publishes a pricing explainer with clear, structured facts. When an AI assistant using RAG answers "how much does this cost," it retrieves that page, quotes the numbers, and cites the company. A competitor that buries the same facts inside a gated PDF gets skipped, and its citation share on pricing prompts falls from 40% to near 0.

## How to use this knowledge
Make your core facts retrievable: Keep them on indexable pages in plain text, not locked inside images or gated files. Add structured data and clear headings so a retrieval system can lift an exact answer without guessing. Treat any LLM output you rely on as a draft to verify, since the model can state a wrong fact with full confidence.

## Growth Memo guidance
> The biggest risk of using generative AI and, at the same time, the biggest factor holding even wider adoption back is hallucination. LLMs can make things up in a very convincing way.
> — [#6 - The state of generative AI for SEO (PT.2)](https://www.growth-memo.com/p/the-state-of-generative-ai-pt2)

> The key point is to not trust LLM output blindly, just like we shouldn't trust everything we read on the internet.
> — [#6 - The state of generative AI for SEO (PT.2)](https://www.growth-memo.com/p/the-state-of-generative-ai-pt2)

## Related concepts
- **Hallucination** — the failure mode retrieval augmented generation reduces by grounding answers in real documents.
- **Grounding** — tying an AI answer to retrieved sources so it can be verified.
- **AI Overviews** — a search product that retrieves and synthesizes web pages, so being retrievable decides whether you appear.
- **Retrieval surface optimization** — the work of making your content easy for these systems to fetch and cite.

## Referenced in these Growth Memos
- [#6 - The state of generative AI for SEO (PT.2)](https://www.growth-memo.com/p/the-state-of-generative-ai-pt2)

---

# Retrieval surface optimization

**Suggested section**
SEO

**Subtitle**
A results page is no longer a single ranked list, and tracking one blue-link position undercounts both your visibility and your risk.

**Meta title**
What is retrieval surface optimization?

**Meta description**
Retrieval surface optimization is making your content selectable across every SERP feature and AI grounding source that retrieves and displays it.

## What it means
Retrieval surface optimization is making your content visible and selectable across every place a search or AI system can pull it from. That includes classic SERP features like image and product carousels and People Also Ask, schema-enhanced listings, and the grounding sources an AI answer cites. It widens the number of surfaces you can appear on and the number of ways a system can lift your content into a result.

## Why it matters
A results page is no longer a single ranked list, and tracking one blue-link position undercounts both your visibility and your risk. Google augments results with SERP features, and AI systems now retrieve and synthesize sources rather than just point to them. In 2025, images are the most visible SERP feature in product-related queries, so a page that ranks well but never enters a carousel loses real estate to weaker pages that do. When an AI answer grounds itself in retrieved pages, the surface you optimize for shifts from a link to the facts a system will quote.

Say a DTC skincare brand adds ItemList schema to its category pages. Google renders product carousels for those pages, and carousel eligibility climbs from 20% to 60% of the brand's category keywords, capturing image slots it never held before.

## How to use this knowledge
Add ItemList schema on category pages so Google can render product carousels, which take more vertical space than a single listing. Make filter and sort options visible and internally linked so they become eligible for sitelinks in the SERP preview. Mine query refinements and autosuggest for query variations, and track ranks by SERP feature and by refinement to decide which filtered category pages you let Google index. Monitor branded prompts so you know what LLMs return about you, the same way you would defend a branded keyword.

## Growth Memo guidance
> Grounding an AI-generated answer introduces a fundamentally different constraint: The system is no longer just pointing to information, it is using it. The goal shifts from "fetch the best documents" to "fetch the best information to synthesize into a reliable, verifiable answer."
> — [What to do now that AIOs turned search into reading sessions](https://www.growth-memo.com/p/traditional-intents-can-no-longer)

> In 2025, images are now the most visible SERP feature in product-related queries.
> — [Google E-commerce SERP Features 2025 vs 2024](https://www.growth-memo.com/p/google-e-commerce-serp-features-2025)

## Related concepts
- **SERP features** — the carousels, PAA boxes, and listings that are the surfaces you optimize for retrieval.
- **Query fan-out** — the AI Mode process that pulls many sources, widening the surfaces you can be retrieved from.
- **Grounding** — how an AI answer selects and uses retrieved sources, which retrieval surface optimization targets.
- **Topical authority** — the depth signal that makes a system more likely to retrieve and trust your pages.

## Referenced in these Growth Memos
- [The state of ecommerce SERP Features](https://www.growth-memo.com/p/the-state-of-ecommerce-serp-features)
- [Google E-commerce SERP Features 2025 vs 2024](https://www.growth-memo.com/p/google-e-commerce-serp-features-2025)
- [Query refinements](https://www.growth-memo.com/p/query-refinements)
- [What to do now that AIOs turned search into reading sessions](https://www.growth-memo.com/p/traditional-intents-can-no-longer)
- [Query fan-out](https://www.growth-memo.com/p/query-fan-out)

---

# SEO AI agents

**Suggested section**
SEO

**Subtitle**
SEO shifts from optimizing for clicks to optimizing for ingestion, getting your data into the answer with enough authority to win the slot.

**Meta title**
What are SEO AI agents?

**Meta description**
SEO AI agents covers both the agents you build to automate SEO work and the agents that act on the web, which your site must be readable by.

## What it means
SEO AI agents covers two connected shifts. The first is the custom agents and workflows you build to automate your own SEO work, from cannibalization detection to reporting. The second is the AI agents that act on the web for a user, booking, buying, and clicking buttons, which your site now has to be readable by. In both, the unit of work moves from a page a person reads to structured data and actions a machine executes.

## Why it matters
SEO shifts from optimizing for clicks to optimizing for ingestion. Agentic commerce moves you from infinite shelf space, the 10 blue links and endless pagination, to constrained shelf space, a few recommendation slots inside an AI response. If your site sits behind CAPTCHAs or lacks action schema, high-value agents fail to transact and you lose revenue as the user moves on. On the build side, the SEOs who learn to build agents automate the grunt work and free up time for strategy.

Say a mid-market B2B SaaS company builds a custom agent that reads Search Console data each morning, flags true keyword cannibalization, and drafts consolidation tickets. A monthly analysis that used to take an analyst 2 days now runs as a 10-minute review.

## How to use this knowledge
Build agents and workflows for the repetitive parts of SEO first, like cannibalization alerts and reporting, where the logic is clear and the volume is high. Make your site machine-readable with clean semantic HTML and labeled buttons, since cluttered layouts and vague CTAs like "Click Here" cause agent task failures. Keep your product feeds accurate and complete so agents can ingest and act on them, and treat feed integrity as the new technical SEO.

## Growth Memo guidance
> SEOs should learn how to build AI agents and workflows that automate tasks. AI changes the way search works but also the way SEOs work.
> — [System builders](https://www.growth-memo.com/p/system-builders)

> In this environment, SEO shifts from optimizing for clicks to optimizing for ingestion.
> — [How do you compete in Agentic Commerce?](https://www.growth-memo.com/p/how-do-you-compete-in-agentic-commerce)

> Cluttered layouts or unlabeled buttons (e.g., vague CTAs like "Click Here") confuse AI agents, leading to task failures.
> — [February '25 trends and news roundup](https://www.growth-memo.com/p/february-25-trends-and-news-roundup)

## Related concepts
- **Agentic commerce** — the shift to constrained shelf space where agents pick a few products, raising the stakes for clean data.
- **Large Action Models (LAMs)** — models built to act on the web and click buttons, the engines behind transacting agents.
- **Feed integrity** — the accuracy of the product data agents ingest, which becomes the new technical SEO.
- **Query fan-out** — the multi-query process AI Mode and agents use, which rewards broad topic coverage.

## Referenced in these Growth Memos
- [10 SEO, marketing, and tech predictions for 2026](https://www.growth-memo.com/p/10-seo-marketing-and-tech-predictions)
- [February '25 trends and news roundup](https://www.growth-memo.com/p/february-25-trends-and-news-roundup)
- [How do you compete in Agentic Commerce?](https://www.growth-memo.com/p/how-do-you-compete-in-agentic-commerce)
- [System builders](https://www.growth-memo.com/p/system-builders)
- [The New Normal](https://www.growth-memo.com/p/the-new-normal)

---

# SEO moat

**Suggested section**
SEO

**Subtitle**
Organic search is a non-linear system where content, links, and user experience compound, so a lead built over years is hard to catch.

**Meta title**
What is an SEO moat?

**Meta description**
An SEO moat is a durable advantage that makes your rankings hard to displace, built from topical authority, brand strength, and compounding factors.

## What it means
An SEO moat is a durable competitive advantage that makes your search rankings hard for competitors to displace without heavy investment in content depth and breadth. It comes from things rivals cannot copy quickly: Deep topical coverage, brand strength, a product that generates its own content and links, and the compounding effect of content, links, and user experience working together. A moat is why some sites keep ranking through algorithm updates while others get wiped out.

## Why it matters
Organic search is a non-linear system where content, links, and user experience compound, so a lead built over years is hard to catch. Without a moat you are exposed, because what Google gives it can take away in a single update. Topical relevance and brand authority increasingly decide whether you earn the click or get buried beneath AI Overviews (AIOs), which makes depth of coverage a defense, not a vanity metric.

Say a DTC skincare brand spends 3 years building complete coverage of every ingredient, routine, and skin concern. A better-funded competitor still cannot rank overnight, because the brand's topical authority and base of returning visitors compound. When a broad core update lands, the brand holds its rankings while thinner competitors drop 30%.

## How to use this knowledge
Build topical authority by covering the entities and questions of your topic completely, so a system reads your site as the fuller source. Invest in brand, since strong brands drive direct traffic and brand authority that ranking factors alone cannot replicate. Reduce dependency on any single keyword or page, because a moat is also insurance against Google reshaping a class of results. Use product-led SEO where you can, so the product itself creates pages and links that are hard to copy.

## Growth Memo guidance
> Topical relevance, ie, how completely a site covers related entities and questions, is a real and important factor in ranking.
> — [How to measure topical authority](https://www.growth-memo.com/p/how-to-measure-topical-authority)

> Organic Search is a non-linear system, meaning the whole is greater than the sum of its parts. Some factors seem to compound, others seem to be driven by thresholds.
> — [The 10 SEO ranking factors we know to be true](https://www.growth-memo.com/p/the-10-seo-ranking-factors-we-know-to-be-true)

## Related concepts
- **Topical authority** — the depth of coverage that forms the core of an SEO moat.
- **Brand authority** — brand strength that drives direct traffic and trust, a close cousin of topical authority.
- **Product-led SEO** — building a product that generates content and links competitors cannot easily copy.
- **Non-linear ranking system** — the compounding of content, links, and user experience that makes a lead hard to catch.

## Referenced in these Growth Memos
- [How to measure topical authority](https://www.growth-memo.com/p/how-to-measure-topical-authority)
- [The 10 SEO ranking factors we know to be true](https://www.growth-memo.com/p/the-10-seo-ranking-factors-we-know-to-be-true)
- [Search On 2021 - The rise of visual indexing](https://www.growth-memo.com/p/search-on-2021-the-rise-of-visual-indexing)
- [Follow-up: David vs. Goliath](https://www.growth-memo.com/p/follow-up-david-vs-goliath)
- [5 fundamental problems in SEO](https://www.growth-memo.com/p/5-fundamental-problems-in-seo)

---

# SEO RFP

**Suggested section**
Foundations

**Subtitle**
Most companies buy SEO without a clear scope, then judge it on the wrong metrics.

**Meta title**
What is an SEO RFP?

**Meta description**
An SEO RFP is a request for proposal a company sends to agencies or consultants to solicit and compare bids for an SEO engagement.

## What it means
An SEO RFP (request for proposal) is a document a company sends to SEO agencies or consultants to solicit bids for a project or ongoing engagement. It spells out the business goals, scope of work, deliverables, timeline, budget range, and how proposals will be judged, so every vendor answers the same brief and you can compare them fairly. It is the formal front door to hiring outside SEO help.

## Why it matters
Most companies buy SEO without a clear scope, then judge it on the wrong metrics. A good SEO RFP forces you to state goals in business terms, revenue, market share, and growth, before you spend, and it gives vendors enough to propose a real plan instead of a generic pitch. Because SEO returns are hard to predict, the RFP is where you set scope, budget logic, and success metrics up front rather than arguing about them six months in.

Say a mid-market B2B SaaS company issues an SEO RFP that names a target of 25% more qualified organic pipeline in 12 months, a defined budget band, and required monthly reporting. Three agencies respond with comparable plans instead of vague promises, and the company picks the one whose scenario-based plan fits its budget. Without the RFP it would have signed a retainer priced on clicks with no way to compare offers.

## How to use this knowledge
Define the goal in executive terms first, since revenue, market share, and growth are what leadership actually buys. Scope the work by capacity and scenario, not by a traffic forecast, so the budget maps to what the team can deliver. Ask each vendor to show how they predict returns and what they will not promise, which separates operators from sales decks. Standardize your evaluation criteria before proposals arrive so you compare plans on the same terms.

## Growth Memo guidance
> Your budget planning must be scenario-based, not traffic-forecasted.
> — [Budget SEO for capacity, not output](https://www.growth-memo.com/p/budget-seo-for-capacity-not-output)

> One of the most fundamental challenges of SEO is predicting returns. Investing without a guaranteed ROI is called gambling.
> — [The ROI of SEO - how to predict traffic and revenue](https://www.growth-memo.com/p/the-roi-of-seo-how-to-predict-traffic-and-revenue)

## Related concepts
- **SEO ROI** — the return an RFP asks vendors to forecast and defend.
- **Scenario-based budgeting** — the capacity-first way to size the scope inside an RFP.
- **SEO business case** — the internal argument an RFP formalizes for buying SEO.
- **In-house vs. agency SEO** — the staffing decision an RFP process usually settles.

## Referenced in these Growth Memos
- [Budget SEO for capacity, not output](https://www.growth-memo.com/p/budget-seo-for-capacity-not-output)
- [The ROI of SEO - how to predict traffic and revenue](https://www.growth-memo.com/p/the-roi-of-seo-how-to-predict-traffic-and-revenue)
- [How to explain the value of SEO to executives](https://www.growth-memo.com/p/how-to-explain-the-value-of-seo-to)

---

# SERP coverage

**Suggested section**
SEO

**Subtitle**
A results page is no longer a single ranked list, so tracking only your blue-link position undercounts both your visibility and your risk.

**Meta title**
What is SERP coverage?

**Meta description**
SERP coverage is how much of the search results you occupy for a topic, across the queries you rank for and the features you appear in.

## What it means
SERP coverage is how much of the search results you occupy for a topic, measured two ways: The range of queries you rank for, and the number of elements you appear in on each results page. It counts classic rankings plus SERP features like image and product carousels, People Also Ask, and featured snippets. Broad SERP coverage means you show up for more queries in a topic and in more slots on each of those queries.

## Why it matters
A results page is no longer a single ranked list, so tracking only your blue-link position undercounts both your visibility and your risk. Google augments results with SERP features that can capture the click or make a whole class of keywords redundant for classic SEO. SERP features are also unstable, since Google constantly tests them, which means your coverage can shift without any change on your side. Coverage ties directly to topical authority: Covering a topic completely is how you rank for more of its queries and qualify for more of its features.

Say a DTC skincare brand ranks 3rd for a head term but also appears in the image carousel, a People Also Ask answer, and a product listing on the same page. A competitor sitting at position 1 with no feature presence still loses visible space. Measured by SERP coverage, the brand holds 4 slots to the competitor's 1 and captures more clicks despite the lower classic rank.

## How to use this knowledge
Track SERP features, not just positions, and learn how your rank tracker names and reports each feature so your data stays comparable. Add structured data like ItemList and FAQ markup to qualify for carousels and People Also Ask. Build topical coverage so you rank for more queries across a topic rather than a handful of head terms. Watch which features Google switches on and off, since those toggles reveal how much traffic a feature actually moves.

## Growth Memo guidance
> SERP Features are Google's way of augmenting search results with potentially helpful direct answers. The risk is that Google sends out less traffic. In some cases, SERP Features can make a whole class of keywords redundant for SEO.
> — [The impact of SERP Features on traffic](https://www.growth-memo.com/p/augmentation)

> Topical relevance, ie, how completely a site covers related entities and questions, is a real and important factor in ranking.
> — [How to measure topical authority](https://www.growth-memo.com/p/how-to-measure-topical-authority)

## Related concepts
- **SERP features** — the carousels, PAA boxes, and snippets that expand what SERP coverage measures.
- **Topical authority** — completeness of topic coverage that drives how many queries you rank for.
- **Share of voice** — the aggregate visibility metric that SERP coverage feeds into.
- **Zero-click search** — the outcome when a SERP feature answers the query, which raises the stakes of coverage.

## Referenced in these Growth Memos
- [The impact of SERP Features on traffic](https://www.growth-memo.com/p/augmentation)
- [Google E-commerce SERP Features 2025 vs 2024](https://www.growth-memo.com/p/google-e-commerce-serp-features-2025)
- [How to measure topical authority](https://www.growth-memo.com/p/how-to-measure-topical-authority)

---

# Static pretrained LLMs

**Suggested section**
AI Research

**Subtitle**
The models behind AI search are frozen at their training cutoff, so they can repeat outdated facts or invent new ones.

**Meta title**
What are static pretrained LLMs?

**Meta description**
A static pretrained LLM is a language model whose knowledge is frozen at its training cutoff, so it cannot see newer information and can hallucinate.

## What it means
A static pretrained LLM is a large language model whose knowledge is fixed at the point its training ends. It learns from a snapshot of text collected up to a cutoff date, then stops learning. It cannot see anything published after that cutoff unless a retrieval layer feeds it fresh data. Because it predicts text from what it already absorbed, it can state wrong facts with full confidence, which is called hallucination.

## Why it matters
The models behind AI search are frozen at their training cutoff, so they can repeat outdated facts or invent new ones. A Stanford study found hallucination rates between 69% and 88% for legal queries, and even newer models hallucinate around 3% of the time. Say a mid-market B2B SaaS company reprices its plans: A static pretrained LLM keeps quoting last year's number until fresh data reaches it, and prospects who ask an assistant get a wrong price with no warning.

## How to use this knowledge
Treat any AI answer about your brand, pricing, or recent events as unverified until you check it against source data. Keep your first-party facts current and machine-readable so a retrieval layer can override stale pretrained knowledge. When you evaluate an AI tool, ask what its knowledge cutoff is and whether it retrieves live sources or answers from memory alone. Do not trust output blindly.

## Growth Memo guidance
> The key point is to not trust LLM output blindly, just like we shouldn't trust everything we read on the internet.
> — [The state of generative AI for SEO (PT.2)](https://www.growth-memo.com/p/the-state-of-generative-ai-pt2)

> LLMs can make things up in a very convincing way.
> — [The state of generative AI for SEO (PT.2)](https://www.growth-memo.com/p/the-state-of-generative-ai-pt2)

## Related concepts
- **Hallucination** — the tendency of a static pretrained LLM to state false information confidently because it predicts text rather than retrieves verified facts.
- **Retrieval-augmented generation (RAG)** — the method that feeds live data into a model to compensate for its frozen training knowledge.
- **Knowledge cutoff** — the date after which a static pretrained LLM has seen no new information.
- **AI Overviews (AIOs)** — Google's generative answers, which lean on models whose base knowledge is static and must be grounded with live retrieval.

## Referenced in these Growth Memos
- [The state of generative AI for SEO (PT.2)](https://www.growth-memo.com/p/the-state-of-generative-ai-pt2)

---

# Supply and demand (in SEO context)

**Suggested section**
SEO

**Subtitle**
When supply and demand are mismatched, you hold inventory nobody searches for, or you miss demand you have no pages for.

**Meta title**
What is supply and demand in SEO?

**Meta description**
In marketplace SEO, supply is your inventory of listings and demand is buyer search interest, and strategy connects the two at decision time.

## What it means
In marketplace and product-led SEO, supply is the inventory your platform holds, the listings, profiles, and product data that generate pages, and demand is the search interest of buyers looking to make a decision. SEO strategy connects the two: It turns your supply of inventory into pages that intercept buyer demand at the moment they decide to buy or book. The product itself supplies the data, and search quantifies the demand behind real problems.

## Why it matters
Marketplaces win in search because their supply of inventory auto-generates pages that match a wide range of buyer demand. When supply and demand are mismatched, you hold inventory nobody searches for, or you miss demand you have no pages for. Say a DTC marketplace for used instruments holds 40,000 listings but only publishes category pages for its 200 highest-volume queries. Building one page per model to meet long-tail demand could grow its indexed landing pages from 200 to several thousand and lift non-brand organic sessions 30%.

## How to use this knowledge
Map your supply of inventory to the queries buyers actually search, and flag models or categories that have demand but no page. Build page templates that turn supply into demand-matched pages at scale, backed by the reviews and technical signals that earn trust. Size the opportunity by search demand and conversion value so you build where buyer interest is highest first. Report on topic and inventory coverage as a whole, not page by page.

## Growth Memo guidance
> Marketplace SEO is the practice of optimizing a site based on its inventory (supply) so that potential buyers (demand) land directly on your marketplace when they're about to make a decision, like a software purchase, booking a flight, scheduling a medical appointment, etc.
> — [Effective Marketplace SEO is more like Product Growth](https://www.growth-memo.com/p/effective-marketplace-seo-is-more)

> For supply-driven product-led SEO, remember: The product itself "supplies" data. That's the content that produces pages for optimization.
> — [3 examples of product-led SEO](https://www.growth-memo.com/p/5-examples-of-product-led-seo)

> SEO quantifies customer needs because it's based on search demand a.k.a. problems real people have.
> — [Great product features are key to SEO traffic for large sites](https://www.growth-memo.com/p/great-product-features-are-key-to-seo-traffic-for-large-sites)

## Related concepts
- **Marketplace SEO** — the practice of matching your inventory supply to buyer demand so searchers land on your platform at decision time.
- **Product-led SEO** — the model where your product data supplies the content that becomes demand-matched pages.
- **Opportunity sizing** — the method for ranking which demand to chase by estimating revenue impact.
- **Search demand** — the volume of queries that signals real problems buyers want solved.

## Referenced in these Growth Memos
- [Effective Marketplace SEO is more like Product Growth](https://www.growth-memo.com/p/effective-marketplace-seo-is-more)
- [3 examples of product-led SEO](https://www.growth-memo.com/p/5-examples-of-product-led-seo)
- [Great product features are key to SEO traffic for large sites](https://www.growth-memo.com/p/great-product-features-are-key-to-seo-traffic-for-large-sites)

---

# Time to value

**Suggested section**
Behavior

**Subtitle**
With AI answers and fewer clicks, the site that buries its answer or loads slowest loses the user before value lands.

**Meta title**
What is time to value?

**Meta description**
Time to value is the elapsed time between a user starting a task and getting the value they came for, in product or in search.

## What it means
Time to value is the elapsed time between a user starting a task and getting the value they came for. It comes from product and UX, where it is the time to reach the "Aha Moment," the point where a user realizes a product's value and retains. In search, time to value is the path from a query, through an organic result or an answer engine, to the moment the user actually gets their answer on the site they land on.

## Why it matters
With AI answers and fewer clicks, the site that buries its answer or loads slowest loses the user before value lands. A searcher who does not reach value fast does not convert or come back, and answer engines increasingly deliver the value before any click happens. Say a mid-market B2B SaaS company hides its pricing behind 3 clicks and a sign-up wall. An answer engine skips it and summarizes a rival instead, so exposing the number on the landing page could cut time to value to seconds and recover the qualified visits it was losing.

## How to use this knowledge
Define the moment a user gets what they came for, then measure how long and how many steps it takes to get there. Front-load that answer so both searchers and answer engines reach it in the first screen, not after scrolling or a gate. Strip friction between the click and the value: Fast load, no interstitials, the answer above the fold. In the era of fewer clicks, make the value you are known for easy to extract so answer engines send you users who already trust you.

## Growth Memo guidance
> Your job as a Growth Marketer is getting users to the "Aha Moment" as quickly as possible.
> — [Why product market-fit is so important for Growth Marketing](https://www.growth-memo.com/p/why-product-market-fit-is-so-important-for-growth-marketing)

> Outcomes trump insights. In 2025, the value of AI is getting things done.
> — [The Alpha is not LLM monitoring](https://www.growth-memo.com/p/the-alpha-is-not-llm-monitoring)

## Related concepts
- **Aha Moment** — the point where a user first realizes a product's value, and the target that time to value races toward.
- **Activation** — the funnel stage where a new user reaches value for the first time.
- **Product-market fit** — the state you reach when enough users hit value fast and retain.
- **Zero-click search** — the pattern where answer engines deliver value on the results page, compressing time to value before any click.

## Referenced in these Growth Memos
- [Why product market-fit is so important for Growth Marketing](https://www.growth-memo.com/p/why-product-market-fit-is-so-important-for-growth-marketing)
- [The Alpha is not LLM monitoring](https://www.growth-memo.com/p/the-alpha-is-not-llm-monitoring)

---

# Topic graph

**Suggested section**
SEO

**Subtitle**
A topic graph makes your pages work together, building the topical relevance Google rewards and giving AI models entities to cite.

**Meta title**
What is a topic graph?

**Meta description**
A topic graph is the connected structure of the topics, subtopics, and entities your content covers and how they relate to one another.

## What it means
A topic graph is the connected structure of the topics, subtopics, and entities your content covers and how they relate to one another. It maps how completely and coherently your content covers a subject area, which is what Google and AI models assess when they decide whether you are a relevant source. Kevin operationalizes the topic graph as a topic map or topic matrix: The parent topics, subtopics, personas, and formats that make up your coverage. In topic-first SEO, topics replace keywords as the atomic unit.

## Why it matters
Keyword-first SEO treats each page in isolation. A topic graph makes your pages work together, building the topical relevance Google rewards and giving AI models entities to cite. Without it, coverage is patchy and your pages compete with each other instead of reinforcing the topic. Say a mid-market B2B SaaS company has 80 posts targeting individual keywords with no structure. Mapping them into a topic graph of 6 parent topics and their subtopics exposes 3 thin areas and 12 overlapping posts, and consolidating and filling the gaps could raise the share of its subtopics ranking in the top 10.

## How to use this knowledge
Build a topic map: Lock your parent topics, then fan out subtopics by persona, funnel stage, and the problems you pull from sales calls and community threads. Audit your coverage against the graph to find subtopics with no page, thin areas, and pages that overlap, then consolidate or fill. Measure and report on topics as a whole so you know which lever to pull when a topic underperforms. Use the graph as the semantic backbone for AI search, where complete entity coverage raises your odds of a citation.

## Growth Memo guidance
> Keywords are no longer the atomic unit of SEO. But topics are.
> — [Topic-first SEO: The smarter way to scale authority](https://www.growth-memo.com/p/topic-first-seo-the-smarter-way-to)

> Use your topic map as the semantic backbone for all qualitative synthesis.
> — [Personas are critical for AI search](https://www.growth-memo.com/p/personas-are-critical-for-ai-search)

## Related concepts
- **Topic clusters** — the grouping of related pages around a parent topic that gives the graph its structure.
- **Topical relevance** — how completely your topic graph covers the entities and questions Google ties to a subject.
- **Topic map** — the working document that lays out your parent topics, subtopics, and formats.
- **Entities** — the people, places, and concepts whose coverage across your graph signals depth to search and AI.
- **Information architecture** — the site structure that mirrors your topic graph, managed as one thought with SEO.

## Referenced in these Growth Memos
- [Topic-first SEO: The smarter way to scale authority](https://www.growth-memo.com/p/topic-first-seo-the-smarter-way-to)
- [Operationalizing your topic-first SEO strategy](https://www.growth-memo.com/p/operationalizing-your-topic-first)
- [Personas are critical for AI search](https://www.growth-memo.com/p/personas-are-critical-for-ai-search)
- [How to measure topical authority](https://www.growth-memo.com/p/how-to-measure-topical-authority)
- [IBM](https://www.growth-memo.com/p/ibm)
- [A better approach to keyword research for content marketing](https://www.growth-memo.com/p/a-better-approach-to-keyword-research-for-content-marketing)

---

# Topical relevance

**Suggested section**
SEO

**Subtitle**
Topical authority was dismissed as an SEO ghost concept until Google's leaked docs confirmed topical relevance as a real ranking factor.

**Meta title**
What is topical relevance?

**Meta description**
Topical relevance is how completely your site covers the entities and questions tied to a subject, the term Google's docs use for topical authority.

## What it means
Topical relevance is how completely your site covers the entities and questions related to a subject. It is the term Google's internal documents use for what the SEO industry has long called topical authority. Google measures it by comparing how well your site covers relevant entities against its own model of the topic. The two names point at the same thing: Cover a subject completely and you earn a ranking advantage.

## Why it matters
Topical authority was dismissed as an SEO ghost concept until Google's leaked docs confirmed topical relevance as a real ranking factor. That changes how you argue for content investment: Depth of coverage across a whole topic moves rankings, not just links or single-page tweaks, and in AI search, complete coverage raises your citation odds. Say a DTC skincare brand sits on page 2 for its main category. Instead of chasing backlinks, it publishes 15 pages covering the ingredients, routines, and questions Google ties to the topic, and as its topical relevance climbs, the category page moves into the top 5.

## How to use this knowledge
Stop pitching topical authority as a vague idea. Point to the Google documentation leaks that name topical relevance as a real ranking factor when you make the case for content budget. Measure how completely you cover the entities and questions Google associates with your topic, compare it against competitors, and fill the gaps. Prioritize entity and question coverage over one-keyword-per-page thinking, and reuse the same work for AI search, where complete coverage raises citation likelihood.

## Growth Memo guidance
> Internal docs leaks and public signals from Google show that topical relevance, ie, how completely a site covers related entities and questions, is a real and important factor in ranking.
> — [How to measure topical authority](https://www.growth-memo.com/p/how-to-measure-topical-authority)

> I used to dismiss topical authority as an SEO ghost concept. But back in 2022, I was wrong: It's far from a ghost.
> — [How to measure topical authority](https://www.growth-memo.com/p/how-to-measure-topical-authority)

## Related concepts
- **Topical authority** — the SEO industry's name for the same concept Google's documents call topical relevance.
- **Entities** — the people, places, and concepts whose coverage Google measures to score topical relevance.
- **Topic graph** — the structured coverage of topics and entities that builds topical relevance.
- **Brand authority** — a close cousin Kevin ties to topical relevance in the era of AI Overviews (AIOs).
- **Information gain** — the additive coverage that deepens topical relevance beyond what competitors already say.

## Referenced in these Growth Memos
- [How to measure topical authority](https://www.growth-memo.com/p/how-to-measure-topical-authority)
- [Topical Authority: myth or reality?](https://www.growth-memo.com/p/topical-authority-myth-or-reality)

---

# Traffic cannibalization

**Suggested section**
SEO

**Subtitle**
AI answers now cannibalize the click itself, so even a page that ranks well can lose the visit.

**Meta title**
What is traffic cannibalization?

**Meta description**
Traffic cannibalization is when one source, a competing page of your own or an AI answer, eats traffic that would otherwise reach your site.

## What it means
Traffic cannibalization is when one source of traffic eats into another that would otherwise reach you. The classic case is two or more of your own pages competing for the same user intent, so none of them ranks well. Kevin's correction: Cannibalization happens at the user intent level rather than the keyword level. The same word now describes AI search surfaces, where AI Overviews (AIOs) and chatbots absorb clicks that used to land on your site.

## Why it matters
Cannibalization drags down organic traffic and revenue, and it is hard to pin down: It changes over time and comes in degrees. When Google updates its understanding of intent in a core update, two pages that were fine can suddenly cannibalize each other. AI answers now cannibalize the click itself, so even a page that ranks well can lose the visit. Say a mid-market B2B SaaS company runs 4 posts aimed at the same buyer intent. Merging them into one could move the page from position 8 to 3, while AIOs that answer the query inline still cut its click-through rate from 20% to 8%.

## How to use this knowledge
Detect cannibalization at the intent level: Group pages by the user intent they serve, then measure content similarity, where a cosine similarity above 0.7 is high and below 0.5 is low. Consolidate or redirect near-duplicate pages, or re-scope them to distinct intents. Re-audit after every core update, since Google's shifting read of intent creates fresh overlap. Account for AI cannibalization too: Track how AIOs and chat answers cut your clicks even where you rank, and favor content that earns the click or the citation.

## Growth Memo guidance
> We all need to stop thinking about this concept as keyword cannibalization and instead as content cannibalization based on user intent.
> — [Solving keyword cannibalization with AI {free workflow}](https://www.growth-memo.com/p/an-ai-powered-workflow-to-solve-content)

> Adding an AI chatbot to Search would mean Google would cannibalize itself, which only a few companies in history successfully accomplished.
> — [How to cannibalize your own product well](https://www.growth-memo.com/p/why-product-cannibalization-can-be-a-good-idea-and-why-not)

> Product cannibalization, or market cannibalization, is often seen as something bad, but it can be good or even necessary.
> — [Standby as Google cannibalizes itself (while also devouring all of us)](https://www.growth-memo.com/p/standby-as-google-cannibalizes-itself)

## Related concepts
- **Content cannibalization** — Kevin's reframing of the problem at the user intent level instead of the keyword level.
- **User intent** — the level at which cannibalization actually happens.
- **Domain quality** — the site-level signal that heavy cannibalization can drag down.
- **AI Overviews** — the AI answer surface that cannibalizes clicks even from pages that rank.
- **Product cannibalization** — the parent idea: Replacing your own traffic or revenue source with a newer one, sometimes on purpose.

## Referenced in these Growth Memos
- [Standby as Google cannibalizes itself (while also devouring all of us)](https://www.growth-memo.com/p/standby-as-google-cannibalizes-itself)
- [Solving keyword cannibalization with AI {free workflow}](https://www.growth-memo.com/p/an-ai-powered-workflow-to-solve-content)
- [How to cannibalize your own product well](https://www.growth-memo.com/p/why-product-cannibalization-can-be-a-good-idea-and-why-not)

---

# Traffic per query

**Suggested section**
SEO

**Subtitle**
You can't treat a query that drives 5,000 visits the same as one that drives 5.

**Meta title**
What is traffic per query?

**Meta description**
Traffic per query estimates how much organic traffic a single query sends to your site, so you can weight coverage and calculate query share.

## What it means
Traffic per query is an estimate of how much organic traffic a single query or keyword sends to your site. You attach a traffic figure to each query you rank for, then use those figures to judge how valuable your coverage of a topic actually is. It turns a raw count of ranked queries into a measure of demand you can prioritize against.

## Why it matters
A list of queries you rank for tells you nothing about their worth until you know what each one is likely to bring. Traffic per query lets you calculate query share, the portion of a topic's total demand you capture, and decide whether covering more queries is worth the engineering and writing time. Without it, you treat a query that drives 5,000 visits the same as one that drives 5.

Say a mid-market B2B SaaS company ranks for 400 queries in a topic and calls that strong coverage. Once you weight each query by its estimated traffic, you find 12 queries drive 80% of the topic's demand and the company ranks for only 3 of them. Query share by value sits at 18%, well below the 60% the raw count implied.

## How to use this knowledge
Pull estimated traffic for every query in a topic from Search Console clicks or a rank tracker, then group the queries by the topic or page template they belong to. Weight each query by its traffic to see which ones carry the demand, and prioritize the high-value queries you don't yet cover. When traffic drops, segment before and after by query type so you can tell whether a valuable cluster declined rather than the whole site.

## Growth Memo guidance
> If we can't promise a specific outcome, we can at least narrow the impact down and attach outcome probabilities.
> — [The ROI of SEO - how to predict traffic and revenue](https://www.growth-memo.com/p/the-roi-of-seo-how-to-predict-traffic-and-revenue)

> The most important thing to do in any case you lose traffic or plateau is to look at the data before and after the drop. Often, a certain type of queries or a page template declined.
> — [12 reasons your SEO traffic is plateauing and how to fix it](https://www.growth-memo.com/p/12-reasons-your-seo-traffic-is-plateauing-and-how-to-fix-it)

## Related concepts
- **Query share** — the portion of a topic's total query demand you capture, which traffic per query lets you weight by value
- **Topical authority** — how completely you cover a topic's entities and questions, measured more honestly when queries are weighted by traffic
- **Search volume** — the raw demand estimate for a query, which traffic per query converts into expected visits at your ranking position
- **Traffic prediction** — narrowing expected SEO outcomes so you can build a business case for the work

## Referenced in these Growth Memos
- [The ROI of SEO - how to predict traffic and revenue](https://www.growth-memo.com/p/the-roi-of-seo-how-to-predict-traffic-and-revenue)
- [12 reasons your SEO traffic is plateauing and how to fix it](https://www.growth-memo.com/p/12-reasons-your-seo-traffic-is-plateauing-and-how-to-fix-it)
- [How to measure topical authority](https://www.growth-memo.com/p/how-to-measure-topical-authority)

---

# User intent alignment

**Suggested section**
SEO

**Subtitle**
When AI systems rarely repeat your exact query, content that only matches keywords gets skipped.

**Meta title**
What is user intent alignment?

**Meta description**
User intent alignment is matching your content to the goal behind a query rather than its exact words, which is what search engines and LLMs reward.

## What it means
User intent alignment is matching your content to the goal behind a query rather than the exact words in it. Google has treated meeting user intent as a core ranking requirement since Hummingbird and RankBrain, and LLM-powered results raise the stakes because they rarely echo the query verbatim. Aligning to intent means you satisfy what the searcher is trying to do, not just the phrase they typed.

## Why it matters
Intent alignment decides whether you rank at all and whether AI systems surface your page. Since AI Mode and AI Overviews (AIOs) break a query into sub-questions and rarely repeat the exact wording, content that only matches keywords gets skipped. Aligning to intent is how you stay visible when the phrase you optimized for is no longer the phrase being answered.

Say a DTC skincare brand ranks a page for "best moisturizer for dry skin" but writes it as a thin product list. An AI Mode query fans out into winter dryness, fragrance-free options, and dermatologist recommendations, and the page gets cited for none of them because it answers the phrase rather than the underlying intent. After you rewrite it to cover those sub-questions, the page starts appearing in AIO citations for 9 queries it previously missed.

## How to use this knowledge
Study the top-ranking results and SERP features for a query to read the intent Google rewards, then shape your content to top it. Map each query and its intent to buyer-journey stages so you show up at every step, not only the final one. Use content tuning: Publish your best version, watch which queries Google tries to rank it for, and expand from there. For AI visibility, answer the sub-questions a query fans out into instead of the exact phrase alone.

## Growth Memo guidance
> Meeting User Intent is one of the most critical factors for ranking high in Google Search. Intent is not static, so we must understand and monitor it for the keywords we want to rank for.
> — [What is User Intent? How to optimize for it like a pro](https://www.growth-memo.com/p/what-is-user-intent-how-to-optimize-for-it-like-a-pro)

> Without fulfilling the right user intent, a page won't rank.
> — [User intent mapping on steroids](https://www.growth-memo.com/p/user-intent-mapping-steroids)

> Only 6% of AIOs contain the search query, which makes meeting user intent in the content much more important than we might have assumed.
> — [AI on Innovation - part 2](https://www.growth-memo.com/p/ai-on-innovation-part-2)

## Related concepts
- **Query fan-out** — how AI Mode expands one query into many, which is why aligning to intent beats matching keywords
- **User intent mapping** — assigning intent to thousands of queries at scale so you can build content around real goals
- **Content tuning** — publishing then expanding a page based on the queries Google tries to rank it for
- **SERP features** — signals you read to infer the intent Google is rewarding for a query

## Referenced in these Growth Memos
- [What is User Intent? How to optimize for it like a pro](https://www.growth-memo.com/p/what-is-user-intent-how-to-optimize-for-it-like-a-pro)
- [User intent mapping on steroids](https://www.growth-memo.com/p/user-intent-mapping-steroids)
- [AI on Innovation - part 2](https://www.growth-memo.com/p/ai-on-innovation-part-2)
- [Query fan-out](https://www.growth-memo.com/p/query-fan-out)
- [Butterfly Effect](https://www.growth-memo.com/p/butterfly-effect)
- [Trust Still Lives in Blue Links](https://www.growth-memo.com/p/trust-still-lives-in-blue-links)

---

# User-generated content (UGC)

**Suggested section**
SEO

**Subtitle**
Without UGC, thousands of near-identical marketplace pages stay thin, and Google has little reason to rank one over another.

**Meta title**
What is user-generated content (UGC)?

**Meta description**
User-generated content (UGC) is content your users create, like reviews, ratings, and seller descriptions, that makes thin marketplace pages unique.

## What it means
User-generated content (UGC) is content your users create rather than your team: Reviews, ratings, comments, profiles, seller descriptions, and uploaded templates. On marketplaces and aggregators, UGC is what makes otherwise thin or near-duplicate inventory pages unique enough to rank. It doubles as a scaling mechanism, since users add content faster than any content team could.

## Why it matters
Marketplaces spin up pages from a database, and many differ only by a product name or a city. Without UGC, thousands of near-identical marketplace pages stay thin and interchangeable, and Google has little reason to rank one over another. Reviews and commentary give each page unique text, freshness, and trust signals, which is why user contributions become the primary fuel for marketplace and product-led SEO.

Say a marketplace for home-service pros has 20,000 location pages that differ only by city name, and duplicate-content dilution keeps most out of the top 10. After it adds customer reviews and a Q&A section to each page, the pages carry unique text, and the share ranking on page one climbs to 38%, up from 12% over 2 quarters.

## How to use this knowledge
Build product mechanisms that prompt users to leave reviews, ratings, and questions, so your supply generates its own content. Surface that UGC on category and inventory pages to differentiate templated pages and keep them fresh. Add review and rating structured data so the content is machine-readable in search. Treat UGC generation as a product growth function, not a one-off content task.

## Growth Memo guidance
> With UGC-based PLSEO, user contributions (templates, profiles, reviews) become the primary SEO fuel.
> — [3 examples of product-led SEO](https://www.growth-memo.com/p/5-examples-of-product-led-seo)

> Marketplaces often rely on user-generated reviews and commentary (plus the right technical signals) to build trust and relevance.
> — [Effective Marketplace SEO is more like Product Growth](https://www.growth-memo.com/p/effective-marketplace-seo-is-more)

## Related concepts
- **Product-led SEO** — using the product itself, including user contributions, as the engine for organic growth
- **Marketplace SEO** — optimizing inventory pages where UGC supplies the differentiation and trust
- **Thin content** — near-duplicate pages that UGC turns into unique, rankable ones
- **Structured data** — review and rating markup that makes UGC machine-readable in search

## Referenced in these Growth Memos
- [3 examples of product-led SEO](https://www.growth-memo.com/p/5-examples-of-product-led-seo)
- [Effective Marketplace SEO is more like Product Growth](https://www.growth-memo.com/p/effective-marketplace-seo-is-more)

---

# Zero volume search queries

**Suggested section**
SEO

**Subtitle**
If you filter your keyword list to volume above zero, you delete the exact queries that build topical authority.

**Meta title**
What are zero volume search queries?

**Meta description**
Zero volume search queries are searches that SEO tools report as having no measurable volume even though people actually search them.

## What it means
Zero volume search queries are searches that SEO tools report as having no measurable search volume, even though people actually run them. They surface as demand your tools can't see: Long-tail phrasings, query refinements, and the sub-questions AI systems generate. Zero volume rarely means zero searches; it means the query is too specific or too new for a tool's sample to register.

## Why it matters
Tools like Ahrefs and Semrush estimate volume from samples, so they miss or round rare queries down to zero. As short-head demand fragments into long-tail and AI systems fan queries out, more of your real demand hides in these queries. If you filter your keyword list to volume above zero, you delete the exact queries that build topical authority and earn AI citations.

Say a mid-market fintech filters its research to queries with 50 or more monthly searches and keeps 300 keywords. A competitor targets the 4,000 zero-volume long-tail questions its buyers actually ask. Twelve months later the competitor captures 3x the non-brand clicks, despite chasing keywords the tools said had no demand.

## How to use this knowledge
Stop filtering keyword lists by a minimum volume threshold, and pull queries from Search Console, People Also Ask, query refinements, and AI Mode fan-outs where zero-volume demand lives. Prioritize fringe and zero-volume questions on your topic map, since they deepen authority and often resolve buyer hesitations. Track clicks and impressions in Search Console rather than tool volume to see the demand estimates miss.

## Growth Memo guidance
> Query refinements lead to search queries without search volume.
> — [Query refinements](https://www.growth-memo.com/p/query-refinements)

> Demand did not disappear, it atomized into thousands of specific queries.
> — [The Great Decoupling](https://www.growth-memo.com/p/the-great-decoupling)

> Search volume is a double-edged sword: It guides our strategy, yet it's so fundamentally flawed that we should handle it with extreme care.
> — [The inaccuracy and flaws of search volume](https://www.growth-memo.com/p/the-inaccuracy-and-flaws-of-search-volume)

## Related concepts
- **Long-tail keywords** — specific, low-frequency queries where most zero-volume demand lives
- **Query fan-out** — AI Mode's expansion of one query into many, generating queries tools never record
- **Query refinements** — filter pills that create searches without recorded volume
- **Topical authority** — coverage depth built partly by answering zero-volume, fringe questions
- **Search volume** — the tool estimate that misses or zeroes out rare queries

## Referenced in these Growth Memos
- [Query refinements](https://www.growth-memo.com/p/query-refinements)
- [The Great Decoupling](https://www.growth-memo.com/p/the-great-decoupling)
- [The inaccuracy and flaws of search volume](https://www.growth-memo.com/p/the-inaccuracy-and-flaws-of-search-volume)
- [Query fan-out](https://www.growth-memo.com/p/query-fan-out)
- [How to measure topical authority](https://www.growth-memo.com/p/how-to-measure-topical-authority)

---

# Agent readiness

**Suggested section**
AI Research

**Subtitle**
If an agent can't extract your pricing, it cites a competitor instead and you lose the sale without ever seeing the traffic.

**Meta title**
What is agent readiness?

**Meta description**
Agent readiness is how well your website lets AI agents retrieve, read, and act on your content instead of serving only human visitors.

## What it means
Agent readiness is how well your website works for AI agents that retrieve, read, and act on your content for a user, rather than for human visitors alone. An agent-ready site exposes its information in a form software can parse, like clean HTML, structured data, and machine-readable pricing and specs, and lets an agent finish a task without hitting a wall. It is the site-level counterpart to the Agentic Web, where software does the browsing.

## Why it matters
As users hand research, shopping, and booking to AI agents, the agent becomes your visitor. If an agent can't extract your pricing, it cites a competitor instead and you lose the sale without ever seeing the traffic. This is the site-level stake of the Agentic Web: As it inverts traditional internet economics, pages built for human eyes risk being invisible to the software doing the work.

Say a B2B SaaS company keeps its pricing in an image and hides specs behind a JavaScript tab. An agent comparing vendors can read neither, so it pulls stale numbers from a third-party review site and recommends a competitor. After the company moves pricing into plain HTML with structured data, agents extract it correctly and it starts showing up in far more agent-generated comparisons.

## How to use this knowledge
Audit whether an agent can retrieve, extract, and cite the answers your key pages hold, like pricing, integrations, and specs, without rendering JavaScript or clearing blockers. Put decision-critical facts in server-rendered HTML and structured data so agents parse them cheaply. Cut the time and tokens an agent spends to reach an answer, because slow or bloated pages get abandoned. Test your pages the way an agent reads them, not just how they look in a browser.

## Growth Memo guidance
> This week, we're looking at the infrastructure of the Agentic Web and the inversion of traditional internet economics.
> — [Growth Intelligence Brief #14](https://www.growth-memo.com/p/growth-intelligence-brief-14)

## Related concepts
- **Agentic Web** — the shift to software agents browsing and buying in place of people, which agent readiness prepares your site for
- **Structured data** — machine-readable markup that lets agents extract pricing and specs without guessing
- **AI Overviews (AIOs)** — generated answers that extract from your pages the way agents do, rewarding the same machine-readable content
- **First-party sufficiency** — whether an agent can answer from your own page instead of citing a third party

## Referenced in these Growth Memos
- [Growth Intelligence Brief #14](https://www.growth-memo.com/p/growth-intelligence-brief-14)
