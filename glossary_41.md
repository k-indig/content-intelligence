# Glossary

# Open weight large language model (LLM)

**Suggested section**
AI Research

**Subtitle**
Open weight models give you direct control over LLM capability, but the same hallucination and citation risks still apply.

**Meta title**
What is an open weight LLM?

**Meta description**
An open weight LLM is a model whose trained parameters are public, so you can download, run, fine-tune, or self-host it instead of calling an API.

## What it means
An open weight large language model (LLM) is a model whose trained parameters are released publicly, so you can download, run, fine-tune, or self-host it instead of only reaching it through a vendor's API. The weights are open, but the training data and code often are not, which separates open weight from fully open source. Llama, Mistral, and DeepSeek are common examples.

## Why it matters
Open weight models give you direct control over LLM capability, but the same hallucination and citation risks still apply. When you can self-host a model, you are not locked into one vendor's pricing, rate limits, or data policy, and you can fine-tune on your own data. The reliability problems Kevin flags for any LLM do not disappear: Models make things up convincingly, and different models pull from different citation sources, so output quality varies. Say a mid-market B2B SaaS company self-hosts an open weight model to summarize support tickets. It cuts per-query cost from $0.02 on a proprietary API to near zero at scale, but a 5% hallucination rate means answers still need human review before they reach customers.

## How to use this knowledge
Decide per use case whether control (self-hosting, fine-tuning, data privacy) outweighs the quality edge of a frontier API model. Benchmark an open weight model on your actual tasks, not public leaderboards, and measure the hallucination rate you can tolerate. Keep a human in the loop for anything customer-facing. Track which models cite which sources, since that shapes whether your content gets surfaced.

## Growth Memo guidance
> The biggest risk of using generative AI and, at the same time, the biggest factor holding even wider adoption back is hallucination. LLMs can make things up in a very convincing way.
> — [#6 - The state of generative AI for SEO (PT.2)](https://www.growth-memo.com/p/the-state-of-generative-ai-pt2)

> Important to note is that LLMs rely on different citation sources to varying degrees.
> — [AI Halftime Report H1 2025](https://www.growth-memo.com/p/ai-halftime-report-h1-2025)

> If content is the new oil, social networks are oil rigs.
> — [Labeled](https://www.growth-memo.com/p/labeled)

## Related concepts
- **Large language model (LLM)** — the broader model class that open weight and closed models both belong to.
- **Hallucination** — the reliability risk that persists whether a model is open weight or proprietary.
- **Fine-tuning** — the main reason to pick an open weight model, since you can retrain it on your own data.
- **Training data** — open weights ship the parameters but not the data the model learned from.

## Referenced in these Growth Memos
- [#6 - The state of generative AI for SEO (PT.2)](https://www.growth-memo.com/p/the-state-of-generative-ai-pt2)
- [AI Halftime Report H1 2025](https://www.growth-memo.com/p/ai-halftime-report-h1-2025)
- [Labeled](https://www.growth-memo.com/p/labeled)
- [How much can we influence AI responses?](https://www.growth-memo.com/p/how-much-can-we-influence-ai-responses)

---

# Open source Large Language Model (LLM)

**Suggested section**
AI Research

**Subtitle**
Self-host an open source LLM and the hallucination risk becomes yours to measure and contain.

**Meta title**
What is an open source LLM?

**Meta description**
An open source LLM is a large language model with public weights you can download, self-host, and fine-tune instead of using a provider's API.

## What it means

An open source Large Language Model (LLM) is a model whose weights are released publicly, so you can download it, run it on your own infrastructure, and fine-tune it on your own data. It is the alternative to a closed model you reach only through a provider's API. The tradeoff is control: You decide where the model runs and what it trains on, and you also carry responsibility for the quality of what it produces.

## Why it matters

Hosting your own model does not remove the core risk Kevin flags with any LLM: Hallucination. Models make things up in a convincing way, and older or smaller ones do it more often. A Stanford study found hallucination rates between 69% and 88% for legal queries, and broad estimates run 15 to 20% for an older model like GPT-3 versus about 3% for newer ones. When you self-host an open source model, that error rate is yours to measure and contain.

Say a mid-market B2B SaaS company self-hosts an open source LLM to draft first-line support replies. It looks cheaper than an API until a spot check shows 16% of answers contain a fabricated detail. After adding retrieval grounding and human review on billing topics, the fabrication rate on sampled replies drops from 16% to 4%, and the model is safe to ship.

## How to use this knowledge

Pick an open source model only when control, cost at scale, or data residency justify the operational load of running it yourself. Benchmark it against a closed API on your own prompts before committing, and measure hallucination on a labeled sample rather than trusting vendor claims. Ground outputs in retrieval and keep a human in the loop for high-stakes answers, because a self-hosted model moves the accuracy burden onto you.

## Growth Memo guidance

> The biggest risk of using generative AI and, at the same time, the biggest factor holding even wider adoption back is hallucination. LLMs can make things up in a very convincing way.
> — [#6 - The state of generative AI for SEO (PT.2)](https://www.growth-memo.com/p/the-state-of-generative-ai-pt2)

> The key point is to not trust LLM output blindly, just like we shouldn't trust everything we read on the internet.
> — [#6 - The state of generative AI for SEO (PT.2)](https://www.growth-memo.com/p/the-state-of-generative-ai-pt2)

## Related concepts

- **Hallucination** — the fabricated-output risk you inherit and have to measure when you self-host a model.
- **Fine-tuning** — the main reason to choose open weights, since you can train the model on your own data.
- **Retrieval-augmented generation (RAG)** — grounds a model's answers in your sources to cut hallucination on factual queries.
- **Proprietary LLM** — the closed, API-only alternative you access through a provider instead of hosting yourself.

## Referenced in these Growth Memos

- [#6 - The state of generative AI for SEO (PT.2)](https://www.growth-memo.com/p/the-state-of-generative-ai-pt2)

---

# AI mention (also referred to as brand mention)

**Suggested section**
AI Research

**Subtitle**
Brand search volume is the single biggest predictor of how often AI engines name you, ahead of anything on your site.

**Meta title**
What is an AI mention or brand mention?

**Meta description**
An AI mention is when an LLM names your brand in a generated answer, with or without a clickable citation linking back to you.

## What it means

An AI mention is your brand or product named inside an AI-generated answer, whether the model links to you or not. It differs from a citation, which attaches a clickable URL. A model can name you from pre-training without crediting a source, so a mention can carry no traffic. Across LLMs, brands tend to appear alongside competing brands rather than alone.

## Why it matters

Brand search volume is the single biggest predictor of how often AI engines name you, ahead of anything on your site. Kevin matched many metrics against AI Chatbot visibility and found brand search volume correlated most strongly with mentions, at .334. Early research suggests a mention's value is high, especially at purchase intent. Your content is only part of the picture: Roughly 15% of why you get mentioned or cited comes from your own site.

Say a mid-market B2B SaaS company tracks how often ChatGPT names it in category prompts. It sits at 4% share of mentions while 3 competitors split most of the rest. Instead of publishing more content, the team runs brand campaigns that lift branded search 30% over 2 quarters, and its mention rate climbs to 12%.

## How to use this knowledge

Track how often each AI engine names you in the prompts your buyers use, and measure your share against the competitors named beside you. Treat branded search volume as a lever that demand generation and PR can move, since it directly affects your mention rate. Because output is inconsistent between engines and runs, sample each prompt several times, not once. Build the trust and authority signals that make a model comfortable recommending you.

## Growth Memo guidance

> I found one factor that stands out more than anything else: Brand search volume.
> — [What content works well in LLMs?](https://www.growth-memo.com/p/what-content-works-well-in-llms)

> Your goal is to ensure your brand is (1) trusted by your target audience and (2) visible in AI Mode output text.
> — [Google’s AI Mode SEO Impact | AI Mode User Behavior Study Part 2](https://www.growth-memo.com/p/googles-ai-mode-seo-impact-ai-mode)

> The content on your website accounts for roughly 15% of why you get mentioned or cited in AI responses.
> — [Growth Intelligence Brief #13](https://www.growth-memo.com/p/growth-intelligence-brief-13)

## Related concepts

- **Brand authority** — the trust and recognition signal that raises how often AI engines name you.
- **LLM citation** — the linked version of a mention, attaching a clickable URL to your brand.
- **Brand search volume** — the demand metric most correlated with AI mention frequency.
- **Share of voice** — how your mention count compares to the competitors named in the same answers.

## Referenced in these Growth Memos

- [Growth Intelligence Brief #6](https://www.growth-memo.com/p/growth-intelligence-brief-6)
- [Google’s AI Mode SEO Impact | AI Mode User Behavior Study Part 2](https://www.growth-memo.com/p/googles-ai-mode-seo-impact-ai-mode)
- [What content works well in LLMs?](https://www.growth-memo.com/p/what-content-works-well-in-llms)
- [Growth Intelligence Brief #13](https://www.growth-memo.com/p/growth-intelligence-brief-13)
- [How you can track Brand Authority for AI Search](https://www.growth-memo.com/p/how-you-can-track-brand-authority)
- [How consumers navigate high-stakes purchases in AI Mode](https://www.growth-memo.com/p/how-consumers-navigate-high-stakes)
- [Trust Still Lives in Blue Links](https://www.growth-memo.com/p/trust-still-lives-in-blue-links)
- [What Our AI Mode User Behavior Study Reveals about the Future of Search](https://www.growth-memo.com/p/what-our-ai-mode-user-behavior-study)
- [Topics matter for third-party authority signals](https://www.growth-memo.com/p/topics-matter-for-third-party-authority)
- [State of AI Search Optimization 2026](https://www.growth-memo.com/p/state-of-ai-search-optimization-2026)

---

# AI citation (also referred to as a brand citation)

**Suggested section**
AI Research

**Subtitle**
If your page isn't cited, you're invisible in the AI answer even when you rank in classic results.

**Meta title**
What is an AI citation?

**Meta description**
An AI citation is when an AI Overview or chatbot references your page or brand as a source in the answer it generates for a query.

## What it means

An AI citation, also called a brand citation, is when an AI system references your page or brand as a source in the answer it generates. In Kevin's words, when you talk about ranking in AI Overviews (AIOs), you mean being cited. AIOs and chatbots build an answer first, then attach sources that fit, so a citation is not the same as a keyword-matched blue link.

## Why it matters

Citation is the unit of visibility in AI search. If your page is not cited, you are invisible in the AI answer even when you rank in classic results. Kevin's analysis of 18,012 verified ChatGPT citations found a ski-ramp distribution: 44.2% of citations come from the first 30% of a page, and content buried deep is roughly 2.5x less likely to be cited. Where you place a claim decides whether it gets cited.

Say a mid-market B2B SaaS company buries its pricing benchmark in the conclusion of a 3,000-word post. It ranks on page 1 but earns no AIO citations. Move that benchmark into the first 30% of the page, box the methodology, and the comparison starts getting cited.

## How to use this knowledge

Put your most citable claims and data in the first 30% of the page, in every vertical. The first 10% is usually navigation and intro filler, so the 10-20% band is where AI reads hardest. Lead with the result: For a comparison, the headline finding goes first, then method, then the caveats. Box your methodology (sample, time window, what was measured, how), because attribution confidence is part of what makes a number citable. Publish expert content under a named author with a byline and a clear timestamp, not just a brand.

## Growth Memo guidance

> When we talk about "ranking in AIOs", we mean being cited.
> — [The impact of AI Overviews on SEO - analysis of 19 studies](https://www.growth-memo.com/p/the-impact-of-ai-overviews-on-seo)

> 44.2% of all citations come from the first 30% of a page ... content buried deep in a long post is roughly 2.5x less likely to be cited.
> — [Why proprietary data is your most defensible AI citation asset](https://www.growth-memo.com/p/why-proprietary-data-is-your-most)

> Attribution confidence is part of what makes a number citable.
> — [Why most original data never gets cited](https://www.growth-memo.com/p/why-most-original-data-never-gets)

## Related concepts

- **AIO citations** — the Google answer box where being cited, rather than ranking, decides whether you appear.
- **Ski-ramp distribution** — the front-loaded citation pattern that dictates where citable claims belong on the page.
- **Proprietary data** — original numbers that are hard to substitute, which raises your citation odds.
- **Third-party authority** — off-site signals and named authorship that make your brand a more citable source.

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
Counting only the citations you can see undercounts how often AI systems actually use your material.

**Meta title**
What is a ghost citation?

**Meta description**
A ghost citation is when an AI answer uses your content to shape its response but never shows a visible, clickable link back to your site.

## What it means
A ghost citation is when an AI answer draws on your content to build its response but never shows a visible, clickable citation back to your site. The influence is real, but the attribution and the click are missing. It sits behind the citation concentration found across AI answers, where a small set of domains earns most of the visible credit while other sources feed the answer without ever surfacing.

## Why it matters
Counting only the citations you can see undercounts how often AI systems actually use your material. If you measure AI visibility by visible links alone, you miss every time your content shaped an answer without credit. I analyzed 1.2 million ChatGPT responses and found that roughly 30 domains own 67% of citations in any topic, so most sites compete for a thin slice of visible credit while their content may still feed answers invisibly.

Say a mid-market B2B SaaS company tracks 12 visible ChatGPT citations a month and calls that its AI footprint. When it audits the actual answers, its documentation is paraphrased in 40 more responses with no link. The real influence is 4x the tracked number, which changes how it values the content.

## How to use this knowledge
Audit the full text of AI answers for your priority prompts, not just the visible citation list, so you catch paraphrases and mentions that never link back. Track branded prompts as diligently as product prompts to see where your material informs answers without credit. Then treat the handful of domains that dominate citations in your topic as digital PR and syndication targets, since that is where visible attribution concentrates.

## Growth Memo guidance
> This tells you which pages AI routinely considers.
> — [The science of how AI picks its sources](https://www.growth-memo.com/p/the-science-of-how-ai-picks-its-sources)

> Roughly 30 domains own 67% of citations in any topic.
> — [The science of how AI picks its sources](https://www.growth-memo.com/p/the-science-of-how-ai-picks-its-sources)

## Related concepts
- **Citation concentration** — the pattern where a few domains earn most visible AI citations, leaving the rest invisible.
- **Answer engine optimization** — earning visibility inside AI answers, where ghost citations are the uncredited version of that visibility.
- **Brand mentions** — unlinked references that, like ghost citations, carry influence without a click.
- **AI visibility tracking** — monitoring that miscounts your real footprint when it logs only visible citations.

## Referenced in these Growth Memos
- [The science of how AI picks its sources](https://www.growth-memo.com/p/the-science-of-how-ai-picks-its-sources)

---

# AI tokens

**Suggested section**
Foundations

**Subtitle**
When you can categorize 20,000 keywords for under $20, the low unit cost of a token is the reason.

**Meta title**
What are AI tokens?

**Meta description**
AI tokens are the units of text that language models read and write, and the meter that sets what running AI actually costs.

## What it means

AI tokens are the units of text that a language model reads and writes. A token is roughly 4 characters, or about 3/4 of a word. Every model prices its input and output by the token, so tokens are the meter that decides what running AI actually costs and how much text you can feed a model at once.

## Why it matters

Token pricing is what makes large-scale AI work economical. When you can categorize 20,000 keywords for under $20, the low unit cost of a token is the reason. As prices per token fall, jobs that were once too expensive to automate become routine, which is the practical engine behind most of the productivity gains you hear about.

Say a mid-market B2B SaaS company wants to classify 100,000 support tickets by theme. At older per-token prices the job might have cost $2,000; after prices dropped, the same job runs for $150. That moves it from a once-a-quarter project to a weekly habit.

## How to use this knowledge

Estimate token cost before you commit to an AI workflow: Count the input plus output tokens per run, then multiply by the model's per-token price. Batch and trim inputs to cut spend on repetitive jobs. Track cost per output (per keyword categorized, per draft written) so you can compare models and re-price a workflow as token rates keep falling.

## Growth Memo guidance

> Aifficiencies are incremental improvements from AI. Instead of doing new things, the biggest value add from AI so far is doing things faster and better.
> — [Aifficiency](https://www.growth-memo.com/p/aifficiency)

> This week, I categorized almost 20,000 keywords into 8 core topics for a client and paid less than $20 in one hour. AI is NOS for no-code.
> — [Aifficiency](https://www.growth-memo.com/p/aifficiency)

## Related concepts

- **Aifficiency** — Kevin's term for the speed and cost gains from AI, which cheap tokens make possible.
- **Token limit** — the cap on how many tokens a model can read and generate in a single request.
- **Programmatic content** — large-scale AI generation where token cost per page decides whether the project is worth running.
- **AI Overviews (AIOs)** — Google's AI answers, produced by processing tokens across many retrieved pages.

## Referenced in these Growth Memos

- [Aifficiency](https://www.growth-memo.com/p/aifficiency)
- [#6 - The state of generative AI for SEO (PT.2)](https://www.growth-memo.com/p/the-state-of-generative-ai-pt2)

---

# Agentic commerce (also referred to as agentic shopping)

**Suggested section**
AI Research

**Subtitle**
Agentic commerce transforms organic search from a source of cheap traffic into the mandatory gatekeeper of AI verification.

**Meta title**
What is agentic commerce?

**Meta description**
Agentic commerce is shopping performed by AI agents on your behalf: Inventory checks, cart management, and secure checkout across AI surfaces.

## What it means

Agentic commerce (also called agentic shopping) is shopping carried out by AI agents on your behalf: Real-time inventory checks, cart management, and secure checkout completed across surfaces like AI Mode and chat assistants. Google's Universal Commerce Protocol and OpenAI's "Buy it in ChatGPT" are the infrastructure behind it. Kevin's read is that it turns organic search from a source of cheap traffic into the gatekeeper of AI verification, where accurate product truth beats marketing.

## Why it matters

Agentic commerce transforms organic search from a source of cheap traffic into the mandatory gatekeeper of AI verification. When an agent checks products against structured data before recommending them, your feed accuracy and product truth decide whether you make the shortlist, so marketing arbitrage loses its value. The phrasing still oversells the near term: Handing an agent a credit card to buy freely is not close, because high-priced purchases are too risky to delegate and low-priced ones already run on subscriptions.

Say a mid-market DTC skincare brand keeps its product feed at 70% attribute completeness. When an agent builds a shortlist for a gentle vitamin C serum, the brand gets skipped because the agent cannot verify ingredients or skin-type fit. After the brand raises feed completeness to 98% with structured attributes, it starts appearing in agent-built comparisons and recovers lost placements.

## How to use this knowledge

Treat your product feed and structured data as the primary interface for agents, not your marketing pages, and make specs, price, availability, and attributes machine-verifiable. Adopt the emerging protocols, Google's Universal Commerce Protocol and checkout inside ChatGPT, so agents can complete tasks on the surfaces where buyers are. Redirect effort away from autonomous-buying scenarios that are not near-term and toward the verification role search now plays.

## Growth Memo guidance

> Agentic commerce transforms organic search from a source of cheap traffic into the mandatory gatekeeper of AI verification. Marketing arbitrage dies; product truth wins.
> — [How do you compete in Agentic Commerce?](https://www.growth-memo.com/p/how-do-you-compete-in-agentic-commerce)

> The phrasing "agentic commerce" sets the wrong expectation. Autonomous purchasing, where you give an agent a credit card and monthly allowance to buy on your behalf, is not becoming a reality in the near future.
> — [How do you compete in Agentic Commerce?](https://www.growth-memo.com/p/how-do-you-compete-in-agentic-commerce)

> Both Google and OpenAI have officially launched their infrastructure for Agentic Commerce.
> — [Growth Intelligence Brief #13](https://www.growth-memo.com/p/growth-intelligence-brief-13)

## Related concepts

- **AI verification** — the check an agent runs against product data before recommending you, which your product truth has to pass.
- **Universal Commerce Protocol** — Google's open standard that lets agents check inventory, manage carts, and check out.
- **Product feed** — the structured source agents read to confirm price, availability, and attributes.
- **Marketing arbitrage** — the older play of buying cheap traffic to convert, which agentic commerce erodes.
- **Zero-click search** — the related shift where the task completes on the platform instead of your site.

## Referenced in these Growth Memos

- [How do you compete in Agentic Commerce?](https://www.growth-memo.com/p/how-do-you-compete-in-agentic-commerce)
- [Growth Intelligence Brief #13](https://www.growth-memo.com/p/growth-intelligence-brief-13)
- [Growth Intelligence Brief #10](https://www.growth-memo.com/p/growth-intelligence-brief-10)

---

# Agentic commerce protocols (ACP, UCP)

**Suggested section**
SEO

**Subtitle**
Agentic commerce turns organic search from a source of cheap traffic into the mandatory gatekeeper of AI verification.

**Meta title**
What are agentic commerce protocols?

**Meta description**
Agentic commerce protocols are open standards (Google's UCP, OpenAI's ACP) that let AI agents check inventory, manage carts, and buy on AI surfaces.

## What it means
Agentic commerce protocols are open standards that let AI agents complete shopping tasks (checking real-time inventory, managing a cart, running secure checkout) directly on AI surfaces instead of on your website. Google's Universal Commerce Protocol (UCP) and OpenAI's Agentic Commerce Protocol (ACP) are the leading examples. They turn your site from a destination into a database that agents read through an API.

## Why it matters
Agentic commerce turns organic search from a source of cheap traffic into the mandatory gatekeeper of AI verification. The game shifts from optimizing landing page design for human eyes to optimizing data feeds for machine ingestion, which collapses a 14-click funnel into roughly 2 interactions: The model matches intent against real-time inventory, and the user clicks once to buy. If your shipping speed, inventory status, or return policy is not accessible via API, you are invisible to the agent. Imagine a DTC skincare brand whose product feed lists only titles and prices. When an agent filters for a fragrance-free moisturizer under $30 that ships in 2 days, the brand never surfaces, and a competitor with a complete feed wins the single click.

## How to use this knowledge
Expose the data agents evaluate: Shipping speed, inventory status, pricing, and return policy in a structured, machine-readable feed or API. Optimize the feed, not just the landing page, and keep it accurate in real time. Invest in product truth, since arbitrage on clever copy dies when the agent compares your data against third-party reviews. Audit whether your catalog is even reachable by an agent today.

## Growth Memo guidance
> Agentic commerce transforms organic search from a source of cheap traffic into the mandatory gatekeeper of AI verification. Marketing arbitrage dies; product truth wins.
> — [How do you compete in Agentic Commerce?](https://www.growth-memo.com/p/how-do-you-compete-in-agentic-commerce)

> If your shipping speed, inventory status, or return policy isn't accessible via API, you are invisible to the agent.
> — [How do you compete in Agentic Commerce?](https://www.growth-memo.com/p/how-do-you-compete-in-agentic-commerce)

> Google introduced the Universal Commerce Protocol (UCP), an open-source standard that allows AI agents to perform complex shopping tasks like real-time inventory checks, cart management, and secure checkout across different surfaces.
> — [Growth Intelligence Brief #13](https://www.growth-memo.com/p/growth-intelligence-brief-13)

## Related concepts
- **Product-led SEO** — product data, not editorial pages, drives visibility once agents mediate the purchase.
- **Zero-click search** — the endpoint of the collapsing funnel, where the agent completes the buy without a site visit.
- **AI Overviews (AIOs)** — the same shift from links to machine-read answers, playing out in search results.
- **Structured data feeds** — the machine-readable format agents ingest to evaluate and rank your products.

## Referenced in these Growth Memos
- [How do you compete in Agentic Commerce?](https://www.growth-memo.com/p/how-do-you-compete-in-agentic-commerce)
- [Growth Intelligence Brief #13](https://www.growth-memo.com/p/growth-intelligence-brief-13)

---

# Inventory indexation

**Suggested section**
SEO

**Subtitle**
Google indexes only about a third of the pages it finds, and pages that get in can be dropped just as fast.

**Meta title**
What is inventory indexation?

**Meta description**
Inventory indexation is the process of deciding which of your product or listing pages search engines should crawl and include in their index.

## What it means

Inventory indexation is the process of deciding which of your product or listing pages search engines should crawl and include in their index. It treats your pages as inventory (supply) and asks which ones earn a place, rather than assuming every page will. Kevin frames marketplace SEO as optimizing a site based on its inventory, and getting the right pages indexed protects crawl budget and content quality.

## Why it matters

Google indexes only about a third of the pages it finds, and pages that get in can be dropped just as fast. A bloated index of thin or duplicate pages wastes crawl budget on pages that will not rank and dilutes the quality signal for the ones that could. Kevin's point in SEOzempic is that you do not always need more pages to grow: Often you improve the inventory you already have.

Say a mid-market marketplace has 80,000 listing pages, but 40% are near-empty or duplicated. After the team prunes and consolidates them to 45,000 substantive listings, the share of indexed pages that earn impressions climbs from 22% to 55%.

## How to use this knowledge

Audit your page inventory and separate the pages that earn impressions from those that only consume crawl budget; Kevin recommends a monitoring system to read domain quality first. Prune, consolidate, or noindex thin and duplicate pages so the index reflects your best supply. Make each page you keep thorough, since a category page with 240 products competes better than one with 12. Speed up discovery with indexing APIs or IndexNow instead of waiting on crawl.

## Growth Memo guidance

> You do not always need to add more pages to grow. Often, you can improve your existing page inventory, but you need a monitoring system to figure this out in the first place.
> — [SEOzempic](https://www.growth-memo.com/p/seozempic)

> Google is very choosy and indexes only 1/3 of the pages it finds in a sample of 16 million over 5 years, but since 2023, it has been indexing more and more (>50% in 2025).
> — [March news](https://www.growth-memo.com/p/march-25-trends-and-news-roundup)

> A category page with 12 products competes badly against one with 240.
> — [What to do now that AIOs turned search into reading sessions](https://www.growth-memo.com/p/traditional-intents-can-no-longer)

## Related concepts

- **Crawl budget** — the finite crawl capacity you protect by keeping low-value pages out of the index.
- **Indexing APIs and IndexNow** — ways to push the pages you want indexed to search engines instead of waiting to be crawled.
- **Page pruning** — removing or consolidating thin pages so what stays indexed is higher quality.
- **Topical authority** — grows when your indexed pages cover a topic in depth instead of spreading thin.

## Referenced in these Growth Memos

- [IndexNow and the future of web crawling](https://www.growth-memo.com/p/indexnow-and-the-future-of-web-crawling)
- [March news](https://www.growth-memo.com/p/march-25-trends-and-news-roundup)
- [SEOzempic](https://www.growth-memo.com/p/seozempic)
- [The end of crawling and the beginning of API indexing](https://www.growth-memo.com/p/the-end-of-crawling-and-the-beginning-of-api-indexing)
- [What to do now that AIOs turned search into reading sessions](https://www.growth-memo.com/p/traditional-intents-can-no-longer)
- [Google’s index is smaller than we think - and might not grow at all](https://www.growth-memo.com/p/googles-index-is-smaller-than-we-think-and-might-not-grow-at-all)
- [Effective Marketplace SEO is more like Product Growth](https://www.growth-memo.com/p/effective-marketplace-seo-is-more)

---

# LLM citation

**Suggested section**
AI Research

**Subtitle**
Higher rankings raise your citation odds, but many AI answers cite nothing at all, so visibility no longer guarantees a click.

**Meta title**
What is an LLM citation?

**Meta description**
An LLM citation is the inline source attribution an AI engine shows beside an answer, crediting the pages its retriever pulled with a clickable URL.

## What it means

An LLM citation is the inline source attribution an AI engine shows beside or beneath a generated answer. It points to the chunks the retriever pulled and the generator chose to credit, each with a clickable URL. Not every source becomes a citation: A model can only cite pages that make it into the answer's retrieval set, and knowledge absorbed during pre-training usually goes unattributed. Live retrieval is what adds the URL that makes attribution possible.

## Why it matters

Higher rankings raise your citation odds, but many AI answers cite nothing at all, so visibility no longer guarantees a click. Because many LLMs use search engines as a retrieval source, ranking in Google's top 10 raises your probability of entering the candidate pool and getting cited. Yet citation rates stay low even when models reach relevant sources: One study Kevin cites found Gemini gives no clickable citation in 92% of answers, and about a quarter of ChatGPT responses fetch no live content at all.

Say a mid-market B2B SaaS company sits at position 8 for its priority queries and gets cited in roughly 1 of 10 tracked AI answers. It rewrites those pages to rank in the top 3 across a wider set of question variations, and its citation rate doubles to 2 of 10.

## How to use this knowledge

Rank in the top 10 for the question and long-tail variations around your core topics, since those pages have the highest citation probability. Track citations as a distinct metric from rankings and traffic, because you can be cited without earning a click. Watch model-level behavior: Routers that favor non-reasoning models show fewer citations and send less traffic, so referral swings are not always your fault. Make your pages easy to retrieve and quote so they survive into the answer the generator builds.

## Growth Memo guidance

> Higher organic rankings increase the probability of entering the LLM's candidate pool and receiving citations.
> — [State of AI Search Optimization 2026](https://www.growth-memo.com/p/state-of-ai-search-optimization-2026)

> The router favors non-reasoning models, which show fewer citations and send less traffic out.
> — [LLM traffic is shrinking](https://www.growth-memo.com/p/llm-traffic-is-shrinking)

> Grounding varies from model to model and not all LLMs prioritize pages ranking at the top of Google search.
> — [How much can we influence AI responses?](https://www.growth-memo.com/p/how-much-can-we-influence-ai-responses)

## Related concepts

- **AI mention** — the unlinked cousin of a citation, where your brand is named without a clickable URL.
- **Candidate pool** — the set of pages eligible for citation, which top-10 rankings help you enter.
- **Grounding** — the retrieval step that pulls live pages into an answer and makes attribution possible.
- **Query fan-out** — the expansion of one prompt into many retrievals, widening which pages can be cited.
- **Zero-click** — the outcome when an answer resolves without any citation click.

## Referenced in these Growth Memos

- [State of AI Search Optimization 2026](https://www.growth-memo.com/p/state-of-ai-search-optimization-2026)
- [LLM traffic is shrinking](https://www.growth-memo.com/p/llm-traffic-is-shrinking)
- [How much can we influence AI responses?](https://www.growth-memo.com/p/how-much-can-we-influence-ai-responses)

---

# LLM experience gain

**Suggested section**
AI Research

**Subtitle**
LLM experience gain flips the question: What can an LLM do for the person already on your page?

**Meta title**
What is LLM experience gain?

**Meta description**
LLM experience gain is building interactive AI features into your content and website so an LLM improves the experience for users already on your page.

## What it means

LLM experience gain, a term coined by Zack Notes, is the practice of building new user experiences with LLMs across your content and website. Instead of treating LLMs only as a retrieval layer that cites your pages elsewhere, you add interactive AI features (answer boxes, assistants, generated summaries, product-description scoring) that make the on-site experience better. The point is a better on-site experience powered by AI you control.

## Why it matters

Most AI-search advice optimizes for being read by an LLM somewhere else. LLM experience gain flips the question: What can an LLM do for the person already on your page? That matters because LLM referral traffic is volatile and concentrated. Kevin found that ChatGPT controls 85% of LLM referral traffic, and that the more organic visits a site gets, the lower its share of LLM traffic, so leaning only on external visibility is fragile. Interactive AI features keep the value on your own property.

Say a mid-market B2B SaaS company adds an AI assistant that turns its pricing docs into a tailored quote. Time on page rises from 40 seconds to 3 minutes, and demo requests from that page climb 18%, without depending on any external LLM to send the traffic.

## How to use this knowledge

Audit where an interactive AI feature would beat a static page: Long product descriptions, complex docs, comparison tables, configurators. Build for both LLMs and humans, using Kevin's marketplace test: Find the parts of your site an LLM would flag as thin or low-trust, and enrich them with real user data or visualizations. Score your own content the way an LLM would, the same way Kevin shipped an audit tool that scores product descriptions, to find weak pages before adding AI features. Keep the experience on your domain so you capture engagement even when external LLM referral traffic dips.

## Growth Memo guidance

> Plan content quality for both LLMs and actual humans.
> — [Effective Marketplace SEO is more like Product Growth](https://www.growth-memo.com/p/effective-marketplace-seo-is-more)

> ChatGPT controls 85% of LLM referral traffic.
> — [LLM traffic is shrinking](https://www.growth-memo.com/p/llm-traffic-is-shrinking)

> Search engines and LLMs are mapping relationships between entities and judging your brand's authority accordingly.
> — [Internal Linking Grows Up: Evolving from Link Juice to Entity Maps](https://www.growth-memo.com/p/internal-linking-grows-up-evolving)

## Related concepts

- **Answer engine optimization (AEO)** — optimizing to be the answer an LLM gives, the visibility side that experience gain complements on-site.
- **LLM referral traffic** — the external clicks from AI tools that experience gain reduces your dependence on.
- **Topical authority** — deep topic coverage that makes LLMs surface your brand and feeds better on-site AI experiences.
- **Product-led SEO** — using your product itself as the experience that drives growth, the closest cousin to building AI features into the page.

## Referenced in these Growth Memos

- [State of AI Search Optimization 2026](https://www.growth-memo.com/p/state-of-ai-search-optimization-2026)
- [How much can we influence AI responses?](https://www.growth-memo.com/p/how-much-can-we-influence-ai-responses)
- [Internal Linking Grows Up: Evolving from Link Juice to Entity Maps](https://www.growth-memo.com/p/internal-linking-grows-up-evolving)
- [Operationalizing your topic-first SEO strategy](https://www.growth-memo.com/p/operationalizing-your-topic-first)
- [Effective Marketplace SEO is more like Product Growth](https://www.growth-memo.com/p/effective-marketplace-seo-is-more)
- [LLM traffic is shrinking](https://www.growth-memo.com/p/llm-traffic-is-shrinking)

---

# LLM leaderboards

**Suggested section**
AI Research

**Subtitle**
A model that leads this quarter can slip in the next, so the choice is rarely permanent.

**Meta title**
What are LLM leaderboards?

**Meta description**
LLM leaderboards are public rankings that compare large language models by measured performance so builders can choose the right model.

## What it means
LLM leaderboards are public rankings that compare large language models by measured performance, using benchmarks, task suites, or head-to-head human votes. They let you see how models stack up on reasoning, coding, retrieval, cost, and speed without running every test yourself. If you build on top of gen AI, you use them to decide which model to ship inside your own product.

## Why it matters
The model you pick sets the ceiling for the quality, latency, and cost of anything you build on it. Leaderboards turn a crowded model market into a comparable ranking so you can match a model to your use case instead of defaulting to the most famous name. A model that leads this quarter can slip in the next, so the choice is rarely permanent.

Say a mid-market B2B SaaS company adds an AI support agent and defaults to the best-known model. A leaderboard check shows a model ranked within 2 points on the tasks it cares about at 60% lower cost per token. Switching cuts inference spend with no measurable drop in answer quality.

## How to use this knowledge
Start from your use case, not the top of the list, and filter leaderboards by the tasks you actually run, whether that is retrieval, code, long-context, or structured output. Weight cost and latency next to raw quality, since the top-ranked model is often the most expensive. Re-check on a set cadence and keep your integration model-agnostic so you can swap when rankings move. Validate the shortlist on your own prompts, because public benchmarks rarely match your exact workload.

## Growth Memo guidance
> Many LLMs use search engines as retrieval sources. Higher organic rankings increase the probability of entering the LLM's candidate pool and receiving citations.
> — [State of AI Search Optimization 2026](https://www.growth-memo.com/p/state-of-ai-search-optimization-2026)

> Pages ranking for many long-tail and question-based variations have higher citation probability.
> — [State of AI Search Optimization 2026](https://www.growth-memo.com/p/state-of-ai-search-optimization-2026)

## Related concepts
- **Model evaluation** — the internal testing that confirms what a public leaderboard suggests for your workload.
- **Benchmarks** — the standardized tasks that leaderboards aggregate into a single ranking.
- **Retrieval-augmented generation** — a use case where leaderboard rank alone misleads unless you test retrieval quality.
- **Inference cost** — the per-token price that should weigh against leaderboard position when you choose a model.

## Referenced in these Growth Memos
- [State of AI Search Optimization 2026](https://www.growth-memo.com/p/state-of-ai-search-optimization-2026)

---

# LLMs

**Suggested section**
Foundations

**Subtitle**
You can't optimize the model itself; the leverage is what an LLM retrieves and cites at query time, not what it already knows.

**Meta title**
What are LLMs?

**Meta description**
LLMs are AI systems trained to generate language, with knowledge frozen at training; they ground answers in sources retrieved at query time.

## What it means

LLMs (large language models) are AI systems trained on large amounts of text to predict and generate language. Their knowledge is frozen at training time. To answer current or specific questions, an LLM retrieves outside sources at query time and grounds its answer in what it pulls back.

## Why it matters

Because an LLM's parameters are fixed after training, you can't optimize the model itself. The leverage is what an LLM retrieves and cites at query time, not what it already knows. Many LLMs use search engines as retrieval sources, so ranking in the top 10 raises your probability of entering the candidate pool and getting cited.

Say a mid-market B2B SaaS company spends a quarter trying to "get into ChatGPT" by seeding brand mentions across the web. Nothing moves, because the model was trained months earlier. When the same team instead earns top-10 rankings for 30 long-tail question variations of its core topic, its citation rate climbs from near zero to showing up in 4 of 10 test prompts.

## How to use this knowledge

Stop trying to influence the model's weights and optimize the retrieval layer instead. Rank for many fan-out query variations, beyond the head terms, so more of your pages qualify for the candidate pool. Use internal links to signal entity relationships and topical authority. Track citations by model, since grounding varies from one model to the next and not every model prioritizes Google's top results.

## Growth Memo guidance

> Many LLMs use search engines as retrieval sources. Higher organic rankings increase the probability of entering the LLM's candidate pool and receiving citations.
> — [State of AI Search Optimization 2026](https://www.growth-memo.com/p/state-of-ai-search-optimization-2026)

> Search engines and LLMs are mapping relationships between entities and judging your brand's authority accordingly.
> — [Internal Linking Grows Up: Evolving from Link Juice to Entity Maps](https://www.growth-memo.com/p/internal-linking-grows-up-evolving)

> To earn lasting visibility, and not short-term visibility bought by hacky LLM visibility tricks, your brand needs to signal to search engines and LLMs that it's an authority in topics related to your offerings.
> — [Operationalizing your topic-first SEO strategy](https://www.growth-memo.com/p/operationalizing-your-topic-first)

## Related concepts

- **Query fan-out** — how a model splits one prompt into many sub-queries, each a fresh retrieval chance.
- **Candidate pool** — the set of pages a model can cite, which you enter by ranking in the top results.
- **Retrieval-augmented generation (RAG)** — the mechanism that lets an LLM ground answers in sources fetched at query time.
- **Topical authority** — the signal LLMs use to judge whether your brand is worth citing.
- **Entity maps** — the relationships an LLM forms from your internal links.

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
The bulk of search demand is spread across countless specific queries, not a handful of head terms.

**Meta title**
What are long-tail queries?

**Meta description**
Long-tail queries are specific, lower-volume searches that sit outside head terms but together form more demand than the hits.

## What it means

Long-tail queries are specific, lower-volume searches that sit outside the small set of high-volume head terms. Kevin frames them through Chris Anderson's 2004 "Long Tail": The internet is so vast and easy to search that it holds content and products for any taste or need, unlike offline shelves limited to what most people want. Each long-tail query is rare on its own. Combined, they add up to a market bigger than the hits.

## Why it matters

The bulk of search demand is spread across countless specific queries, not a handful of head terms. Kevin cites Anderson's example that more than half of Amazon's book sales come from outside its top 130,000 titles, and the same pattern holds in search. As people search with more words and full questions, more demand moves into the tail, and covering it completely is how you build topical relevance and win rankings on the head terms too.

Say a mid-market B2B SaaS company ranks for 5 head terms that drive most of its tracked volume. It publishes 200 pages answering specific product and use-case questions. Each page pulls only 20 to 80 visits a month, but together they add more traffic than the head terms and lift the core rankings alongside.

## How to use this knowledge

Build content that covers the full range of specific questions and entities in your topic, not just the head terms, since that coverage is what earns topical relevance. Mine Search Console, autocomplete, and People Also Ask for the exact multi-word phrasings people use. Structure pages so one asset captures many close long-tail variants instead of spinning up thin pages for each keyword.

## Growth Memo guidance

> The longtail is a concept SEOs are very familiar with, but it's also a significant differentiator between traditional and internet economics.
> — [Youtube's hashtag pages and the long-tail](https://www.growth-memo.com/p/youtubes-hashtag-pages-and-the-long-tail)

> The idea is simple: The internet is so vast and easy to search that it has content/products for anyone, no matter their taste or need.
> — [Here is the longtail](https://www.growth-memo.com/p/here-is-the-longtail)

## Related concepts

- **Topical authority** — built by covering the long tail of a topic's queries completely.
- **Head terms** — the few high-volume queries the long tail sits beneath.
- **Query share** — the portion of a topic's total demand, most of it long-tail, that your rankings capture.
- **Content clusters** — the structure for capturing many related long-tail queries with linked pages.
- **Search intent** — the specific need behind each long-tail phrasing that a page has to match.

## Referenced in these Growth Memos

- [Youtube's hashtag pages and the long-tail](https://www.growth-memo.com/p/youtubes-hashtag-pages-and-the-long-tail)
- [Here is the longtail](https://www.growth-memo.com/p/here-is-the-longtail)

---

# Marketplace SEO

**Suggested section**
SEO

**Subtitle**
For marketplaces, good SEO is the result of product growth, not just website optimization.

**Meta title**
What is marketplace SEO?

**Meta description**
Marketplace SEO is optimizing a platform based on its inventory (supply) so buyers (demand) land on your marketplace when they're ready to decide.

## What it means
Marketplace SEO is the practice of optimizing a site based on its inventory (supply) so that potential buyers (demand) land directly on your marketplace when they are about to make a decision, like booking a flight or scheduling an appointment. Marketplaces are platforms that connect supply and demand, and Kevin calls them SEO Aggregators. The work is heavy on the technical and product side because you manage hundreds of thousands, sometimes millions, of programmatically generated pages.

## Why it matters
For marketplaces, good SEO is the result of product growth, not just website optimization. Organic search brings the majority of new users, so SEO is the growth engine rather than a side channel, and low per-user revenue often makes paid acquisition too expensive to scale. That pushes the work toward page quality, crawl efficiency, and internal linking across the whole inventory. Imagine a mid-market marketplace connecting tutors with students. It has 200,000 tutor pages, but only 40,000 are indexed because of thin profiles and wasted crawl budget. Improving profile completeness and internal linking gets 150,000 indexed and lifts organic sessions 60% in 2 quarters, without adding a single editorial article.

## How to use this knowledge
Treat marketplace SEO as a product function: Prioritize supply quality and the system that generates pages, not one-off content. Fix the technical foundations that matter at scale, crawl efficiency, internal linking, and page quality. Use user-generated reviews to build trust, relevance, and long-tail keyword coverage. Plan for AI eroding the moat by diversifying page types and defending brand demand.

## Growth Memo guidance
> Good SEO is the result of Product Growth, not just website optimization.
> — [Effective Marketplace SEO is more like Product Growth](https://www.growth-memo.com/p/effective-marketplace-seo-is-more)

> The approach to Aggregator SEO is heavy on the technical and product side. You usually deal with hundreds of thousands, if not millions, of pages and need to optimize for page quality, crawl efficiency, and internal linking.
> — [Marketplace Deep Dive - Q1 (Case Study: Zillow)](https://www.growth-memo.com/p/marketplace-deep-dive-q1-case-study)

> Marketplace SEO is deeply intertwined with product evolution, UX, and brand strategy.
> — [Marketplace Deep Dive - Q2](https://www.growth-memo.com/p/marketplace-deep-dive-q2)

## Related concepts
- **Product-led SEO** — marketplace SEO is a product-led SEO function where supply generates the pages.
- **SEO Aggregator** — Kevin's name for marketplaces, sites that scale on inventory plus technical SEO.
- **Crawl efficiency** — decisive at marketplace scale, so engines index the pages that actually matter.
- **User-generated content (UGC)** — reviews and listings that build trust and spawn long-tail coverage.

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
For a growth team, microagents 10x repetitive work without waiting for one do-everything tool.

**Meta title**
What are microagents?

**Meta description**
Microagents are small, single-purpose AI agents you build to handle one narrow task and connect into larger workflows over time.

## What it means

Microagents are small, single-purpose AI agents you build to handle one narrow task, like pulling Google Search Console data and turning it into action items, or scraping and analyzing 2 pages. Instead of one system that does everything, you connect these simple building blocks and compose them into more capable workflows over time. Kevin's advice from his live session is to start with a defined, simple use case rather than a general-purpose agent.

## Why it matters

For a growth team, microagents are how you 10x repetitive work without waiting for one do-everything tool. Each agent owns a small, verifiable job, so you can chain them, swap one out, and trust the output more than you would a single black box. Kevin expects a major agent breakthrough and points to a growing number of agent platforms that make this buildable now.

Say a mid-market B2B SaaS growth team builds 3 microagents: One pulls weekly Search Console data and flags pages that dropped, one scrapes the 2 competitors ranking above them, and one drafts a fix brief. What used to be a 6-hour manual Monday audit runs in 20 minutes, and the SEO lead reviews the output instead of assembling it.

## How to use this knowledge

Define one simple, high-frequency task first, like extracting Search Console data and generating action items, and ship that before anything ambitious. Keep each agent narrow so its output is easy to verify. Chain the building blocks into a workflow, and add a checking step where one agent verifies another's work, the way multi-agent debate uses a second agent as a judge. Expand scope only once the simple agents are reliable.

## Growth Memo guidance

> Start with tasks like extracting data from Google Search Console and generating action items, or scraping and analyzing two pages. By connecting these simple building blocks, you can create more complex agents over time.
> — [Live Session recap - 1/31/25](https://www.growth-memo.com/p/live-session-recap-13125)

> The growing number of AI agent platforms is exciting, and I believe we'll see a major agent breakthrough this year.
> — [Live Session recap - 1/31/25](https://www.growth-memo.com/p/live-session-recap-13125)

> New technology, like "ReliabilityRAG" or "Multi-Agent Debate," where one AI agent retrieves the info and another agent acts as a "judge" to verify it.
> — [10 SEO, marketing, and tech predictions for 2026](https://www.growth-memo.com/p/10-seo-marketing-and-tech-predictions)

## Related concepts

- **AI agents** — the broader category; microagents are the small, task-scoped version you compose.
- **Multi-agent debate** — pairing a retriever agent with a judge agent to verify output before you trust it.
- **Agentic web** — the emerging infrastructure where agents, not only people, retrieve and transact on the open web.
- **Workflow automation** — the result when you chain microagents into a repeatable process.

## Referenced in these Growth Memos

- [Live Session recap - 1/31/25](https://www.growth-memo.com/p/live-session-recap-13125)
- [10 SEO, marketing, and tech predictions for 2026](https://www.growth-memo.com/p/10-seo-marketing-and-tech-predictions)

---

# NEG SEO

**Suggested section**
SEO

**Subtitle**
Most negative SEO fails because Google weighs negative signals with thresholds, so only sustained, high-volume attacks move rankings.

**Meta title**
What is NEG SEO (negative SEO)?

**Meta description**
Negative SEO (NEG SEO) is a third party attacking your rankings with spammy backlinks, scraped content, or fake signals you don't control.

## What it means

Negative SEO (NEG SEO) is a third party trying to damage your search rankings with tactics you don't control: Floods of spammy backlinks aimed at your domain, scraped or duplicated copies of your content, fake reviews, or hacking. Google evaluates many negative signals with thresholds rather than one by one, so isolated problems rarely move rankings. Sustained attacks are the concern, because they can push a site past the point where Google reinforces negative consequences.

## Why it matters

Most negative SEO fails because Google weighs negative signals with thresholds, so only sustained, high-volume attacks actually move rankings. Kevin's read of ranking factors is that a few 404s won't hurt, but past a certain percentage Google starts reinforcing negative consequences. The same technical signals an attack exploits, 404s, duplicates, canonicals, are often not impactful on their own. Your exposure depends on ratio and volume relative to a healthy baseline.

Say a mid-market B2B SaaS company suddenly gets 40,000 spammy backlinks from gambling sites in a month. Its rankings hold, because the toxic links stay a small share of a large, healthy profile. A DTC brand with only 200 referring domains hit by the same attack is more exposed, because the toxic ratio crosses a threshold.

## How to use this knowledge

Monitor your backlink profile for sudden spikes in low-quality referring domains, and disavow toxic links when the ratio gets dangerous. Watch for scraped or duplicated versions of your pages, and file removal requests or assert canonicals. Keep your own technical hygiene tight so you are never sitting near a negative threshold when an attack lands. Set alerts on rankings and links so you catch an attack early rather than after traffic drops.

## Growth Memo guidance

> Google also seems to measure negative factors with thresholds: A few 404s won't hurt, but after a certain percentage Google seems to reinforce negative consequences.
> — [The 10 SEO ranking factors we know to be true](https://www.growth-memo.com/p/the-10-seo-ranking-factors-we-know-to-be-true)

> 404s, duplicates, load time, non-canonicals, JavaScript errors, backlinks - these all can be important but not impactful on traffic growth.
> — [Marketing Mix](https://www.growth-memo.com/p/marketing-mix)

## Related concepts

- **Toxic backlinks** — the spammy inbound links a negative SEO attack points at your domain.
- **Disavow file** — the tool for telling Google to ignore links you can't get removed.
- **Content scraping** — republishing your content elsewhere to create duplicate-content pressure.
- **Negative ranking factors** — the site-level faults Google measures with thresholds, which attacks try to exploit.
- **Technical hygiene** — the ongoing maintenance that keeps you clear of those thresholds.

## Referenced in these Growth Memos

- [The 10 SEO ranking factors we know to be true](https://www.growth-memo.com/p/the-10-seo-ranking-factors-we-know-to-be-true)
- [Marketing Mix](https://www.growth-memo.com/p/marketing-mix)

---

# NLPs

**Suggested section**
Foundations

**Subtitle**
NLP is the technology that decides whether a machine understands what your page is about, and it can still get it wrong.

**Meta title**
What is NLP (natural language processing)?

**Meta description**
NLP (natural language processing) is how search engines and LLMs parse, interpret, and generate human language, the base layer under generative AI.

## What it means

NLP (natural language processing) is the set of techniques that let machines parse, interpret, and generate human language. Search engines use it to understand what a query and a page mean; LLMs use it to read your content and produce answers. Generative AI is an application of NLP, which is why the same technology that writes fluent answers can also state them wrongly.

## Why it matters

NLP is the technology that decides whether a machine understands what your page is about, and it can still get it wrong. When it works, your content gets matched to the right queries and summarized accurately. When it fails, you get hallucination: Kevin points out that LLMs can make things up in a very convincing way, with a Stanford study finding hallucination rates between 69% and 88% for legal queries. One NLP misread of your brand reaches users at scale.

Say a mid-market B2B SaaS company publishes a security page an LLM misinterprets, and the model tells buyers the product lacks SOC 2 compliance it actually holds. One NLP misread becomes a lost deal before a human ever checks the source.

## How to use this knowledge

Write for machine parsing: Clear structure, plain claims, explicit entities, so NLP systems read your pages with less ambiguity. Do not trust LLM output blindly. Fact-check AI-generated content about your brand and category before it ships, and monitor how AI tools describe you. Feed models unambiguous facts (specs, compliance, pricing) in plain text near the top of the page so there is less room for an NLP system to guess.

## Growth Memo guidance

> LLMs can make things up in a very convincing way.
> — [#6 - The state of generative AI for SEO (PT.2)](https://www.growth-memo.com/p/the-state-of-generative-ai-pt2)

> The key point is to not trust LLM output blindly, just like we shouldn't trust everything we read on the internet.
> — [#6 - The state of generative AI for SEO (PT.2)](https://www.growth-memo.com/p/the-state-of-generative-ai-pt2)

## Related concepts

- **Large language models (LLMs)** — NLP systems trained on huge text corpora that generate the answers users now read instead of clicking.
- **Hallucination** — the failure mode where an NLP model produces confident, false output, including about your brand.
- **Entities** — the people, products, and concepts NLP maps to understand meaning and judge relevance.
- **Generative AI** — the application of NLP that writes summaries and answers, and the main adoption risk Kevin flags.

## Referenced in these Growth Memos

- [#6 - The state of generative AI for SEO (PT.2)](https://www.growth-memo.com/p/the-state-of-generative-ai-pt2)

---

# PR SEO

**Suggested section**
SEO

**Subtitle**
Third-party signals carry a large share of brand visibility, especially in AI answers that weigh what other sources say about you.

**Meta title**
What is PR SEO?

**Meta description**
PR SEO uses public relations and digital PR to earn the third-party signals that lift your organic and AI search visibility.

## What it means
PR SEO is the practice of using public relations, digital PR, and press coverage to earn the third-party signals that lift your organic and AI search visibility. Instead of publishing more of your own content, you get other credible sites to mention, cite, and link to you. Those external signals feed both Google's ranking systems and the way LLMs decide which brands to surface.

## Why it matters
Third-party signals now carry a large share of brand visibility, especially in AI answers where models weigh what other sources say about you. That makes PR a growth channel, not only a communications function. As classic search sends fewer clicks, being the brand that publications and forums reference is how you stay visible when your own pages are not the ones being read.

Say a mid-market B2B SaaS company runs an original data study and pitches it to trade publications. Ten pickups later, its brand starts showing up in ChatGPT answers for category prompts it never ranked for, and branded search plus referral traffic climbs 15% in a quarter. The coverage did the work its blog could not.

## How to use this knowledge
Budget digital PR as a real line item in your SEO capacity, not an afterthought. Build assets worth citing, like original data studies and surveys, then pitch them to the publications and creators your buyers already trust. Track branded prompts in ChatGPT and other engines to see whether your PR is turning into mentions. Prioritize the sources those engines cite most in your category as your outreach targets.

## Growth Memo guidance
> Third-party signals drive 85% of brand visibility in LLMs.
> — [Budget SEO for capacity, not output](https://www.growth-memo.com/p/budget-seo-for-capacity-not-output)

> Invest in monitoring and optimizing your visibility across relevant ChatGPT prompts with targeted content, PR campaigns, content syndication, and content repurposing.
> — [How is answer engine optimization different from SEO?](https://www.growth-memo.com/p/is-geoaeo-the-same-as-seo)

## Related concepts
- **Digital PR** — the link-and-mention earning tactic at the core of PR SEO.
- **Brand authority** — the reputation signal that third-party coverage builds and that AI engines reward.
- **Answer engine optimization** — the AI-search discipline where PR-driven mentions decide whether you appear in answers.
- **Content syndication** — republishing your work on other outlets to widen the third-party footprint PR SEO depends on.

## Referenced in these Growth Memos
- [How is answer engine optimization different from SEO?](https://www.growth-memo.com/p/is-geoaeo-the-same-as-seo)
- [Budget SEO for capacity, not output](https://www.growth-memo.com/p/budget-seo-for-capacity-not-output)

---

# Query relevance

**Suggested section**
AI Research

**Subtitle**
LLM prompts are longer and more specific than keywords, so relevance is judged against a hyperspecific query, often at the passage level.

**Meta title**
What is query relevance?

**Meta description**
Query relevance is how well a page or passage matches what a search wants, measured in AI search as embedding similarity between query and text.

## What it means

Query relevance is how well a page, or a single passage, matches what a search actually wants. In classic search, Google infers intent from the query and rewards content structured to match it. In AI search, relevance is scored as embedding similarity: The query and your text are turned into numeric vectors, and the closest vectors get retrieved and cited.

## Why it matters

LLM prompts are longer and more specific than keywords, so relevance is judged against a hyperspecific query, often at the passage level. Query fan-out multiplies this: One prompt becomes many sub-queries, each needing a relevant passage. Pages that match many long-tail and question variations have higher citation probability in AI Overviews (AIOs) and chat answers.

Imagine a DTC skincare brand with one broad page on "best moisturizer." Against a conversational prompt like "fragrance-free moisturizer for eczema-prone skin in winter," that page scores low on embedding similarity and never gets retrieved. Break it into specific passages that each answer one variation, and the brand goes from 0 citations to appearing for 12 hyperspecific prompts.

## How to use this knowledge

Write to specific intent instead of head keywords. Mine query refinements, autosuggest, and real customer language for the exact variations people ask. Structure content in passages that each answer one question, so each passage can be retrieved on its own. Match the format the query implies, the way Google rewards a step-by-step layout for "starting a business." Then rank in the top 10 for fan-out variations to raise retrieval and citation probability.

## Growth Memo guidance

> In classic Search, Google returns one ranked list for a query. In AI Mode, Gemini explodes your prompt into a swarm of sub-queries, each aimed at a different part of what you really care about.
> — [Query fan-out](https://www.growth-memo.com/p/query-fan-out)

> Since LLM prompts are conversational and varied, pages ranking for many long-tail and question-based variations have higher citation probability.
> — [State of AI Search Optimization 2026](https://www.growth-memo.com/p/state-of-ai-search-optimization-2026)

> More data came out showing a high overlap between pages cited in AIOs and pages ranking in the top spots for the same query.
> — [AI on Innovation](https://www.growth-memo.com/p/ai-on-innovation)

## Related concepts

- **Search intent** — the classic name for what query relevance measures: What the searcher is really after.
- **Embeddings** — the vectors that let AI systems score relevance by meaning instead of keyword match.
- **Query fan-out** — the reason relevance is now judged against many hyperspecific sub-queries per prompt.
- **Query refinements** — the filter-style pills that reveal the specific variations users care about.
- **Topical authority** — the site-level signal that complements passage-level relevance in earning citations.

## Referenced in these Growth Memos

- [Query refinements](https://www.growth-memo.com/p/query-refinements)
- [State of AI Search Optimization 2026](https://www.growth-memo.com/p/state-of-ai-search-optimization-2026)
- [Query fan-out](https://www.growth-memo.com/p/query-fan-out)
- [AI on Innovation](https://www.growth-memo.com/p/ai-on-innovation)
- [Trust Still Lives in Blue Links](https://www.growth-memo.com/p/trust-still-lives-in-blue-links)
- [The impact of SERP Features on traffic](https://www.growth-memo.com/p/augmentation)
- [10 factors that make great content great](https://www.growth-memo.com/p/10-factors-that-make-great-content-great)

---

# Query share

**Suggested section**
SEO

**Subtitle**
Query share turns an abstract claim about topical authority into a number you can track and benchmark against competitors.

**Meta title**
What is query share?

**Meta description**
Query share is the portion of total search demand for a topic that your site captures through its rankings, used as a proxy for topical authority.

## What it means

Query share is the portion of total search demand for a topic that your site or page captures through its rankings. You take the queries that define a topic, weight them by search volume, and measure how much of that demand your rankings win. Kevin uses it as a proxy for topical authority: The more of a topic's queries you rank for, the more completely you cover it in Google's eyes.

## Why it matters

Topical authority is hard to prove to a boss or a budget holder. Kevin built his measurement approach around that exact problem, quoting the reader whose boss keeps asking them to prove why topical authority is important and measure it. Query share turns an abstract claim about topical authority into a number you can track over time and benchmark against competitors on the same query set.

Say a mid-market B2B SaaS company defines its core topic with 400 queries worth 500,000 monthly searches combined. It holds top-10 rankings for queries representing 12% of that demand. After 6 months of filling entity and question gaps, its query share reaches 30%, and it uses that rise to justify continued content investment.

## How to use this knowledge

Define a topic as a volume-weighted set of queries, then measure the share where you hold meaningful rankings and track that percentage over time. Benchmark your query share against competitors on the same set to see who actually owns the topic. Treat the queries you do not rank for as your content and internal-linking backlog, the same work that builds topical relevance.

## Growth Memo guidance

> Internal docs leaks and public signals from Google show that topical relevance, ie, how completely a site covers related entities and questions, is a real and important factor in ranking.
> — [How to measure topical authority](https://www.growth-memo.com/p/how-to-measure-topical-authority)

> Google rewards sites that cover a topic in-depth, and it does so by comparing how well the site covers relevant entities.
> — [How to measure topical authority](https://www.growth-memo.com/p/how-to-measure-topical-authority)

## Related concepts

- **Topical authority** — the broader signal query share is meant to quantify.
- **Topical relevance** — Google's term for how completely you cover a topic, which query share estimates.
- **Share of voice** — the visibility metric query share adapts to a single topic's query set.
- **Content gap analysis** — the method that turns unclaimed query share into a content backlog.

## Referenced in these Growth Memos

- [How to measure topical authority](https://www.growth-memo.com/p/how-to-measure-topical-authority)

---

# Query universe

**Suggested section**
SEO

**Subtitle**
Anyone can build a large list of keywords, but creating strong filters and sorting is the hard part.

**Meta title**
What is a query universe?

**Meta description**
A query universe is the full pool of search queries your audience uses around a topic, built as a living, sorted list you prioritize over time.

## What it means
A query universe is the full pool of search queries your target audience uses to find you around a topic. Kevin builds this as a Keyword Universe: A large, living list that surfaces the most important queries and phrases at the top and lives in a spreadsheet or database like BigQuery. Instead of running a keyword sprint every quarter, you build the universe once and work through it across your site over time. Mapping it is the first step to measuring and optimizing topical authority.

## Why it matters
Anyone can build a large list of keywords, but creating strong filters and sorting is the hard part. Keyword and intent research is often static, done monthly and quickly outdated, so a query universe turns it into a continuous prioritization system. The value is the sorting, not the list: Search volume alone is weak because high-volume keywords often convert worse, or return no traffic once AI Overviews (AIOs) answer them. Say a mid-market B2B SaaS company maps a query universe of 12,000 queries for its category. By sorting on conversion potential instead of volume, it finds 300 mid-volume queries no competitor covers well, prioritizes them, and lifts qualified organic signups 25% while ignoring head terms buried under AIOs.

## How to use this knowledge
Mine seed keywords through rank trackers, then expand and store everything in one database instead of scattered spreadsheets. Sort and filter on more than volume: Intent, conversion potential, and difficulty. Map your coverage against competitors to find the entities and questions you are missing, which is how you build topical authority. Keep the universe live and revisit it, rather than repeating a one-off research sprint.

## Growth Memo guidance
> A Keyword Universe is a big pool of language your target audience uses when they search that will help them find you.
> — [Keywords are dead. But the Keyword Universe Isn't.](https://www.growth-memo.com/p/universe)

> Anyone can create a large list of keywords, but creating strong filters and sorting mechanisms is hard.
> — [Keywords are dead. But the Keyword Universe Isn't.](https://www.growth-memo.com/p/universe)

> Topical relevance, how completely a site covers related entities and questions, is a real and important factor in ranking.
> — [How to measure topical authority](https://www.growth-memo.com/p/how-to-measure-topical-authority)

## Related concepts
- **Topical authority** — you map the query universe to measure and build it.
- **Keyword mining** — the workflow that generates the raw list feeding the universe.
- **Search intent** — the sorting axis that matters more than raw search volume.
- **AIOs** — why many high-volume head terms now return little or no traffic.

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
Overlap tells you where authority is shared and where it is not.

**Meta title**
What is ranking overlap?

**Meta description**
Ranking overlap is the share of queries or topics where more than one site ranks at once, a read on topical competitiveness and authority.

## What it means

Ranking overlap is the portion of queries or topics where more than one site ranks at the same time. You measure it to see how much your visibility intersects with a competitor's and how contested a topic is. The more queries where you and a rival both appear, the more directly you compete for the same demand, which is a practical read on topical competitiveness and the authority you need to win.

## Why it matters

Overlap tells you where authority is shared and where it is not. Kevin's cross-engine research shows overlap is not uniform: Guides and tutorials share the most ground, while product pages and homepages share the least. Read against competitors, low overlap on a topic can signal an opening, and high overlap means you need more topical depth to break through.

Say a mid-market B2B SaaS company maps its top 300 commercial queries against 2 competitors. It finds 70% overlap with one rival on "alternatives" and "versus" queries but only 15% on how-to topics. The team shifts content investment into the low-overlap how-to space, where authority is easier to build, and picks up rankings on 40 new queries within a quarter.

## How to use this knowledge

Pull the queries you rank for and intersect them with each key competitor to get an overlap percentage per topic. Treat high overlap as a contested area that needs more topical depth, meaning more complete coverage of the entities and questions, to win. Treat low overlap as either an opening or an irrelevant space, and validate it against real demand before you invest. Segment overlap by page type, since explanatory content overlaps and travels further than brand or product pages.

## Growth Memo guidance

> Topical relevance, ie, how completely a site covers related entities and questions, is a real and important factor in ranking.
> — [How to measure topical authority](https://www.growth-memo.com/p/how-to-measure-topical-authority)

> Guides and tutorials have the highest cross-engine overlap at 2.3%, followed by blogs at 1.8%, category pages at 1.6%, product pages at 1.2%, and homepages at 1.1%.
> — [The Consensus Gap](https://www.growth-memo.com/p/the-consensus-gap)

> One of the lowest hanging fruits in SEO is to create content for "alternative" and "versus" queries.
> — [3 best practices of competitor SEO you should know](https://www.growth-memo.com/p/3-best-practices-of-competitor-seo-you-should-know)

## Related concepts

- **Topical authority** — the depth of coverage that decides who wins the high-overlap queries.
- **Keyword cannibalization** — internal overlap, where 2 of your own pages compete for one query.
- **Competitor SEO** — using "alternative" and "versus" queries to deliberately overlap with a rival's demand.
- **Cross-engine overlap** — the AI-search version, measuring how often the same source is cited across ChatGPT, Perplexity, and AI Overviews.
- **Share of voice** — the aggregate visibility metric that overlap analysis rolls up into.

## Referenced in these Growth Memos

- [The Consensus Gap](https://www.growth-memo.com/p/the-consensus-gap)
- [How to measure topical authority](https://www.growth-memo.com/p/how-to-measure-topical-authority)
- [3 best practices of competitor SEO you should know](https://www.growth-memo.com/p/3-best-practices-of-competitor-seo-you-should-know)
- [Solving keyword cannibalization with AI {free workflow}](https://www.growth-memo.com/p/an-ai-powered-workflow-to-solve-content)

---

# Relevance engineering

**Suggested section**
Foundations

**Subtitle**
Google and LLMs increasingly judge you on brand demand and authority, which makes engineering relevance the real job.

**Meta title**
What is relevance engineering?

**Meta description**
Relevance engineering is building the brand and authority signals that get you surfaced, cited, and recommended by search engines and LLMs.

## What it means

Relevance engineering is the practice of building the signals that make your brand and content read as relevant and authoritative to search engines and LLMs, so you get surfaced, cited, and recommended. It reframes the job around brand and authority instead of keywords and links. As Kevin puts it, the discipline is moving toward a "new SEO" that sits closer to brand marketing, where success is measured by citations, brand mentions, and influence.

## Why it matters

Google and LLMs increasingly judge you on brand demand and authority, which makes engineering relevance the real job. Google evaluates sites heavily on brand search volume and user signals, so differentiated brands with real demand get more room from algorithm updates while undifferentiated ones struggle. On AI surfaces, the goal shifts from earning clicks to getting your data ingested with enough authority that the model recommends you.

Say a mid-market B2B SaaS company has strong rankings but weak brand demand. Its pages get crawled but rarely recommended in AI answers. It moves budget from publishing more articles to earning third-party authority and lifting branded search 25% over 2 quarters, and its citation rate across tracked prompts rises alongside it.

## How to use this knowledge

Shift how you measure success: Add citations, brand mentions, and conversion influence next to traffic rather than replacing your old metrics, so leadership can follow the change. Invest in what engines and models reward, brand demand, third-party authority, and differentiated content competitors can't cheaply copy. For agentic and AI surfaces, engineer your data for ingestion with clean feeds and structured product data. Pitch the reframe internally as a move from performance marketing toward brand, which is where richer budgets tend to sit.

## Growth Memo guidance

> In this environment, SEO shifts from optimizing for clicks to optimizing for ingestion.
> — [How do you compete in Agentic Commerce?](https://www.growth-memo.com/p/how-do-you-compete-in-agentic-commerce)

> With a "new SEO" (SEO 2.0?) comes an opportunity to detach from performance marketing and get closer to brand marketing.
> — [Growth Intelligence Brief #3](https://www.growth-memo.com/p/growth-intelligence-brief-3)

> Success will be measured by citations, brand mentions and conversion influence, not raw traffic.
> — [The impact of AI Mode on SEO -analysis of 10 studies](https://www.growth-memo.com/p/the-impact-of-ai-mode-on-seo-analysis)

## Related concepts

- **Answer engine optimization (AEO)** — the AI-search sibling discipline focused on winning prompts and citations.
- **Brand authority** — the trust signal relevance engineering is built to grow.
- **Information gain** — serving data and findings competitors don't have, which raises your relevance to engines and models.
- **Ingestion** — getting your data into an AI system's working set with enough authority to be recommended.
- **Differentiation** — the brand strength that earns you more room from algorithm updates.

## Referenced in these Growth Memos

- [How do you compete in Agentic Commerce?](https://www.growth-memo.com/p/how-do-you-compete-in-agentic-commerce)
- [Growth Intelligence Brief #3](https://www.growth-memo.com/p/growth-intelligence-brief-3)
- [The impact of AI Mode on SEO -analysis of 10 studies](https://www.growth-memo.com/p/the-impact-of-ai-mode-on-seo-analysis)
- [Differentiation](https://www.growth-memo.com/p/differentiation)
- [Information Gainz](https://www.growth-memo.com/p/information-gainz)
- [How is answer engine optimization different from SEO?](https://www.growth-memo.com/p/is-geoaeo-the-same-as-seo)
- [The impact of GPT-3 on Google Search, a complex-adaptive system](https://www.growth-memo.com/p/the-impact-of-gpt-3-on-google-search)

---

# Retrieval-augmented generation (RAG)

**Suggested section**
Foundations

**Subtitle**
Retrieval-augmented generation exists to reduce generative AI's biggest risk: Hallucination.

**Meta title**
What is retrieval-augmented generation (RAG)?

**Meta description**
Retrieval-augmented generation (RAG) grounds an LLM's answer in documents it retrieves at query time, cutting hallucination by tying output to sources.

## What it means

Retrieval-augmented generation (RAG), sometimes called RAG LLMs, is an architecture that grounds a language model's answer in documents it retrieves at query time instead of relying only on what the model memorized in training. The model searches a source set, pulls the relevant passages, and writes its answer from them. AI Overviews (AIOs) and AI chatbots work this way, which is why the pages they retrieve decide what the answer says.

## Why it matters

RAG exists to reduce the biggest risk in generative AI: Hallucination. Kevin flags that LLMs can make things up in a very convincing way, with a Stanford study finding hallucination rates between 69% and 88% for legal queries. Grounding an answer in retrieved sources ties it back to real documents, so retrieval is also your way in: If an AI system retrieves and cites your page, your content shapes the answer. If it does not, you are absent from it.

Say a mid-market B2B SaaS company keeps a current, well-structured docs page on integrations. A RAG-based assistant retrieves it and answers a buyer's setup question accurately, while a competitor with outdated PDFs gets paraphrased wrong or skipped. The retrievable page wins the mention.

## How to use this knowledge

Make your pages retrievable and machine-readable: Clear structure, plain claims, stable URLs, so a RAG system can pull the right passage. Put the answer near the top, because RAG grabs the most relevant passage and rewards leading with the fact over burying it. Keep source content current and accurate, since RAG reduces hallucination only when the retrieved documents are correct, so stale or vague pages produce wrong answers about you. Fact-check AI answers about your brand, because retrieval lowers hallucination but does not remove it.

## Growth Memo guidance

> LLMs can make things up in a very convincing way.
> — [#6 - The state of generative AI for SEO (PT.2)](https://www.growth-memo.com/p/the-state-of-generative-ai-pt2)

> The key point is to not trust LLM output blindly, just like we shouldn't trust everything we read on the internet.
> — [#6 - The state of generative AI for SEO (PT.2)](https://www.growth-memo.com/p/the-state-of-generative-ai-pt2)

## Related concepts

- **Hallucination** — the false-but-confident output RAG is designed to reduce by grounding answers in sources.
- **AIOs** — a RAG-style surface that retrieves pages and writes an answer from them.
- **AI citation** — the payoff when a RAG system retrieves and credits your page as a source.
- **Large language models (LLMs)** — the generation half of RAG, paired with a retrieval step over documents.

## Referenced in these Growth Memos

- [#6 - The state of generative AI for SEO (PT.2)](https://www.growth-memo.com/p/the-state-of-generative-ai-pt2)

---

# Retrieval Surface Optimization

**Suggested section**
SEO

**Subtitle**
If you optimize only for blue-link rankings, you go missing on the surfaces where attention is actually moving.

**Meta title**
What is Retrieval Surface Optimization?

**Meta description**
Retrieval Surface Optimization structures your content so it can be retrieved and rendered across SERP features, AI Overviews, and chat answers.

## What it means
A retrieval surface is any place a search or AI system pulls content to build an answer: Blue links, SERP features like carousels, AI Overviews (AIOs), AI Mode, and chat answers in ChatGPT. Retrieval Surface Optimization is structuring your content and markup so it can be retrieved and rendered on each of those surfaces, not only the 10 blue links. It matters because these systems now use your content to build answers, not just point to it.

## Why it matters
Google spent years moving results from text to visuals, and AI answers now sit on top. Each surface has its own retrieval rules: A carousel needs ItemList schema, an AIO needs content it can extract and verify, a fan-out query needs pages that answer related sub-questions. If you optimize only for blue-link rankings, you go missing on the surfaces where attention is actually moving.

Say a mid-market B2B SaaS company ranks 3rd for its head term but never appears in the AIO or the People Also Ask box above it. After restructuring the page into clear Q&A blocks and adding schema, it gets pulled into the AIO and its click share recovers from 4% to 11%.

## How to use this knowledge
Map every surface your priority queries trigger, then match structure to each: ItemList schema for carousels, extractable answer blocks for AIOs, and coverage of the sub-questions a fan-out query expands into. Monitor SERP features and AI answers, not just keyword ranks, so you see where you win and lose ground. Track branded prompts to confirm AI engines return the information you want.

## Growth Memo guidance
> The system is no longer just pointing to information, it is using it. The goal shifts from "fetch the best documents" to "fetch the best information to synthesize into a reliable, verifiable answer."
> — [What to do now that AIOs turned search into reading sessions](https://www.growth-memo.com/p/traditional-intents-can-no-longer)

> ItemList schema on the category page lets Google render product carousels in the SERP. A carousel takes more vertical space than a single listing and dominates the back-scroll.
> — [What to do now that AIOs turned search into reading sessions](https://www.growth-memo.com/p/traditional-intents-can-no-longer)

> This list will help you optimize your content ecosystem to fully address the multifaceted needs behind your target user's search goal.
> — [Query fan-out](https://www.growth-memo.com/p/query-fan-out)

## Related concepts
- **Query fan-out** — the AI Mode process that turns one query into many, defining which sub-questions your pages must answer.
- **SERP features** — the carousels, snippets, and boxes that are retrieval surfaces beyond blue links.
- **AI Overviews** — the synthesized answer surface where extractable, verifiable content wins placement.
- **Schema markup** — the structured data that makes your content eligible for carousels and rich results.
- **Answer engine optimization** — the broader discipline of earning visibility inside AI answers that this operationalizes surface by surface.

## Referenced in these Growth Memos
- [The state of ecommerce SERP Features](https://www.growth-memo.com/p/the-state-of-ecommerce-serp-features)
- [Google E-commerce SERP Features 2025 vs 2024](https://www.growth-memo.com/p/google-e-commerce-serp-features-2025)
- [Query refinements](https://www.growth-memo.com/p/query-refinements)
- [What to do now that AIOs turned search into reading sessions](https://www.growth-memo.com/p/traditional-intents-can-no-longer)
- [Query fan-out](https://www.growth-memo.com/p/query-fan-out)
- [Users behave differently in AI Overviews vs. AI Mode](https://www.growth-memo.com/p/users-behave-differently-in-ai-overviews)
- [The impact of AI Mode on SEO -analysis of 10 studies](https://www.growth-memo.com/p/the-impact-of-ai-mode-on-seo-analysis)
- [Eroding moats](https://www.growth-memo.com/p/eroding-moats)

---

# SEO AI agents

**Suggested section**
SEO

**Subtitle**
Your job shifts from doing SEO tasks to designing the systems that do them, and from tuning one channel to orchestrating across many.

**Meta title**
What are SEO AI agents?

**Meta description**
SEO AI agents are systems you build to run SEO tasks on their own, from pulling data to flagging issues and updating content without manual steps.

## What it means

SEO AI agents are AI systems you build to run SEO tasks on their own: Pulling data, spotting patterns, and taking actions such as flagging cannibalization or refreshing content, without you doing each step by hand. Kevin's view is that building these agents and workflows is now a core SEO skill.

## Why it matters

The work you used to do by hand, like downloading Search Console data and building pivot tables, can move to an agent that runs on its own. That shifts your job from doing tasks to designing the systems that do them, and from tuning one channel to orchestrating across AI Overviews (AIOs), chatbots, and feeds.

Say a mid-market B2B SaaS company has one SEO analyst who spends 2 days a month hunting keyword cannibalization in spreadsheets. She builds an agent that watches Search Console, flags true cannibalization, and drafts the fix. The 2-day job becomes a 20-minute review, and she reinvests the rest in strategy.

## How to use this knowledge

Pick a repetitive, rules-based task you run every month (cannibalization checks, internal-link audits, content refreshes) and build an agent or workflow to handle it. Learn the building blocks of connecting data sources, writing prompts, and chaining steps. Move up to orchestration, piping data from classic SERPs, AIOs, and AI Mode into one view. Make your own site readable to outside agents with clean semantic HTML, labeled buttons, and action schema, so agents can act instead of failing.

## Growth Memo guidance

> SEOs should learn how to build AI agents and workflows that automate tasks. AI changes the way search works but also the way SEOs work.
> — [System builders](https://www.growth-memo.com/p/system-builders)

> Your job is no longer to fine-tune a single channel; it's to keep an entire orchestra in time as search fragments across AI Overviews, chatbots, and social feeds.
> — [The New Normal](https://www.growth-memo.com/p/the-new-normal)

> SEO shifts from optimizing for clicks to optimizing for ingestion. The goal isn't to get a human to visit your landing page; it's to get your product data in front of the agent with enough authority that it recommends you.
> — [How do you compete in Agentic Commerce?](https://www.growth-memo.com/p/how-do-you-compete-in-agentic-commerce)

## Related concepts

- **AI workflows** — the chained, automated task sequences an agent runs to replace manual SEO steps.
- **Agentic commerce** — AI agents transacting for users, which changes what your site has to expose.
- **Large Action Models (LAMs)** — models built to navigate the web and click, the engines behind acting agents.
- **Optimizing for ingestion** — getting your data into an agent's answer set instead of chasing human clicks.

## Referenced in these Growth Memos

- [System builders](https://www.growth-memo.com/p/system-builders)
- [The New Normal](https://www.growth-memo.com/p/the-new-normal)
- [How do you compete in Agentic Commerce?](https://www.growth-memo.com/p/how-do-you-compete-in-agentic-commerce)
- [10 SEO, marketing, and tech predictions for 2026](https://www.growth-memo.com/p/10-seo-marketing-and-tech-predictions)
- [February '25 trends and news roundup](https://www.growth-memo.com/p/february-25-trends-and-news-roundup)

---

# SEO moat

**Suggested section**
SEO

**Subtitle**
A moat is what keeps a single algorithm update or a well-funded competitor from erasing your rankings.

**Meta title**
What is an SEO moat?

**Meta description**
An SEO moat is a durable competitive advantage that protects your rankings, so rivals cannot outrank you without heavy investment in content and brand.

## What it means

An SEO moat is a durable competitive advantage that protects your search rankings, so competitors cannot outrank you without heavy investment in content depth, breadth, links, and brand. Kevin describes organic search as a non-linear system where content, links, and user experience together produce a stronger effect than any factor in isolation, which makes an established position hard to attack. The moat is the accumulated coverage and brand strength a rival would need years to replicate.

## Why it matters

A moat is what keeps a single algorithm update or a well-funded competitor from erasing your rankings. Organic traffic can lift a business fast, and Google can pull it back just as fast, so the depth of your topical coverage, a brand that drives direct traffic, and design quality all raise the cost of competing against you. Kevin's recurring advice is to diversify beyond head terms and build a brand so you are not exposed to a single point of failure.

Say a mid-market B2B SaaS company ranks for 50 head terms with thin pages and no brand. A funded competitor copies its content and outranks it within 2 quarters. Give the same company 800 interlinked pages covering the topic end to end, a brand that drives 40% of traffic directly, and years of links, and catching up would cost a multi-year, seven-figure effort, so the competitor does not try. That gap is the moat.

## How to use this knowledge

Build topical depth and breadth so your coverage of a topic is expensive to replicate, the same work that earns topical authority. Invest in brand so a meaningful share of traffic arrives direct and survives ranking swings, which Kevin lists as a core defensive move. Differentiate with strategy and design: Play to your unique strengths and treat web aesthetics as a real advantage. Diversify into the long tail so no single ranking loss is fatal.

## Growth Memo guidance

> Having great content, links, and user experience seems to have a stronger effect than each factor added in isolation.
> — [The 10 SEO ranking factors we know to be true](https://www.growth-memo.com/p/the-10-seo-ranking-factors-we-know-to-be-true)

> Build strong brands to drive direct traffic.
> — [Search On 2021 - The rise of visual indexing](https://www.growth-memo.com/p/search-on-2021-the-rise-of-visual-indexing)

> A strategy that differentiates your company and plays to its unique strengths.
> — [Live Session September '24 - recap](https://www.growth-memo.com/p/live-session-september-24-recap)

## Related concepts

- **Topical authority** — the depth-and-breadth coverage that forms the content side of a moat.
- **Brand authority** — direct traffic and recognition that insulate rankings from algorithm swings.
- **Compounding ranking factors** — the way content, links, and UX reinforce each other into a defensible position.
- **Web aesthetics** — design quality as a differentiator competitors cannot copy quickly.
- **Keyword diversification** — spreading demand across the long tail so no single loss is fatal.

## Referenced in these Growth Memos

- [The 10 SEO ranking factors we know to be true](https://www.growth-memo.com/p/the-10-seo-ranking-factors-we-know-to-be-true)
- [Search On 2021 - The rise of visual indexing](https://www.growth-memo.com/p/search-on-2021-the-rise-of-visual-indexing)
- [The forgotten but impactful art of web aesthetics](https://www.growth-memo.com/p/the-forgotten-but-impactful-art-of)
- [Follow-up: David vs. Goliath](https://www.growth-memo.com/p/follow-up-david-vs-goliath)
- [Live Session September '24 - recap](https://www.growth-memo.com/p/live-session-september-24-recap)
- [How to measure topical authority](https://www.growth-memo.com/p/how-to-measure-topical-authority)

---

# SEO RFP

**Suggested section**
Foundations

**Subtitle**
The quality of the proposals you get back matches the quality of the RFP you send.

**Meta title**
What is an SEO RFP?

**Meta description**
An SEO RFP (Request for Proposal) is a document you send to agencies to solicit bids for SEO work, defining scope, goals, budget, and criteria.

## What it means
An SEO RFP (Request for Proposal) is a document you send to agencies or consultants to solicit bids for SEO work. It defines the scope, goals, budget range, deliverables, timeline, and the criteria you will use to compare responses. A clear SEO RFP lets you evaluate vendors on the same terms instead of on sales polish.

## Why it matters
The quality of the proposals you get back matches the quality of the RFP you send. Vague goals produce vague bids you cannot compare side by side. Because SEO competes for less than 10% of most marketing budgets and its returns are hard to predict, a strong RFP forces you to name the business outcome you are buying, revenue, market share, or growth, rather than a list of tasks. That framing is also what executives weigh when they approve the spend. Say a mid-market B2B SaaS company issues an SEO RFP that only asks for "more traffic." It gets 6 proposals ranging from $3,000 to $30,000 a month with no shared definition of success. After rewriting the RFP to specify a target of 25% more qualified signups in 12 months and a scenario-based budget, the same agencies return comparable, scoped bids.

## How to use this knowledge
Define the outcome rather than the task list, and tie it to revenue or qualified pipeline. Give a budget range and size it by capacity and scenarios, rather than by last year's traffic. Ask each vendor how they predict and measure ROI, and how often they report. Set your evaluation criteria before responses arrive so you score every bid on the same rubric.

## Growth Memo guidance
> Executives live in a world where attention is scarce but information abundant.
> — [How to explain the value of SEO to executives](https://www.growth-memo.com/p/how-to-explain-the-value-of-seo-to)

> Your budget planning must be scenario-based, not traffic-forecasted.
> — [Budget SEO for capacity, not output](https://www.growth-memo.com/p/budget-seo-for-capacity-not-output)

> They need to think like Product Managers who talk to customers, prioritize features and break their impact down all the way to revenue.
> — [The ultimate guide to effective inhouse SEO [2025 edition]](https://www.growth-memo.com/p/how-to-master-the-art-of-inhouse-seo)

## Related concepts
- **SEO ROI** — the return projection a strong RFP asks every vendor to justify.
- **Business case for SEO** — the internal argument the RFP operationalizes to win budget.
- **Scope of work (SOW)** — the contract that follows once you pick an RFP response.
- **Capacity-based budgeting** — how Kevin recommends sizing the SEO investment you put in the RFP.

## Referenced in these Growth Memos
- [How to explain the value of SEO to executives](https://www.growth-memo.com/p/how-to-explain-the-value-of-seo-to)
- [Budget SEO for capacity, not output](https://www.growth-memo.com/p/budget-seo-for-capacity-not-output)
- [The ultimate guide to effective inhouse SEO [2025 edition]](https://www.growth-memo.com/p/how-to-master-the-art-of-inhouse-seo)
- [The ROI of SEO - how to predict traffic and revenue](https://www.growth-memo.com/p/the-roi-of-seo-how-to-predict-traffic-and-revenue)

---

# SERP coverage

**Suggested section**
SEO

**Subtitle**
A single ranking is a small slice of a page Google fills with features.

**Meta title**
What is SERP coverage?

**Meta description**
SERP coverage is the breadth and depth of your visibility across search results for a topic: The queries you rank for and the features you hold.

## What it means

SERP coverage is the breadth and depth of your visibility across search results for a topic: How many of the relevant queries you appear for, and how much of each results page you hold. Because a SERP is more than 10 blue links, coverage includes SERP Features like images, product carousels, and People Also Ask, not just classic rankings. The more of a topic's queries and result surfaces you occupy, the stronger your coverage.

## Why it matters

A single ranking is a small slice of a page Google fills with features. Kevin's e-commerce analysis shows images became the most visible feature in product queries, and product listings now appear in roughly 55% of position-1 queries, so classic rank alone can misread your real visibility. SERP Features can also make a whole class of keywords redundant by answering the query on the page, which shrinks the coverage worth chasing.

Say a DTC skincare brand ranks in position 3 for 200 topic queries but shows in the image carousel on only 12% of them. After it adds image sitemaps and structured product data, it starts appearing in image and product features on 45% of those queries. Classic rank barely moves, yet clicks rise because the brand now holds more of each SERP.

## How to use this knowledge

Measure coverage on 2 axes: Query breadth (how many topic queries you appear for) and SERP depth (how many surfaces per page you hold, classic result plus features). Track which SERP Features show for your topic and which you are eligible for, and watch them over time, because Google switches features on and off. Prioritize the features that dominate your queries, like images and product listings in shopping, with the right feeds and structured data. Discount queries where a feature answers the question on the page and sends no click.

## Growth Memo guidance

> SERP Features are Google's way of augmenting search results with potentially helpful direct answers. In some cases, SERP Features can make a whole class of keywords redundant for SEO.
> — [The impact of SERP Features on traffic](https://www.growth-memo.com/p/augmentation)

> In 2025, images are now the most visible SERP feature in product-related queries.
> — [Google E-commerce SERP Features 2025 vs 2024](https://www.growth-memo.com/p/google-e-commerce-serp-features-2025)

> Topical relevance, ie, how completely a site covers related entities and questions, is a real and important factor in ranking.
> — [How to measure topical authority](https://www.growth-memo.com/p/how-to-measure-topical-authority)

## Related concepts

- **SERP Features** — the carousels, snippets, and People Also Ask boxes that make up the depth half of coverage.
- **Topical authority** — broad coverage of a topic's queries is how you earn and signal it.
- **Zero-click searches** — queries where a feature answers on the page, coverage that sends no traffic.
- **Share of voice** — the aggregate metric your SERP coverage across a topic rolls up into.

## Referenced in these Growth Memos

- [The impact of SERP Features on traffic](https://www.growth-memo.com/p/augmentation)
- [Google E-commerce SERP Features 2025 vs 2024](https://www.growth-memo.com/p/google-e-commerce-serp-features-2025)
- [How to measure topical authority](https://www.growth-memo.com/p/how-to-measure-topical-authority)

---

# Static pretrained LLMs

**Suggested section**
AI Research

**Subtitle**
A static pretrained model can confidently state wrong or outdated facts about your brand, category, or pricing.

**Meta title**
What is a static pretrained LLM?

**Meta description**
A static pretrained LLM answers only from knowledge frozen at training time, with no live retrieval, so it can be outdated and prone to hallucination.

## What it means

A static pretrained LLM is a model whose knowledge is frozen at training time. It answers from patterns learned during pretraining, with no live retrieval, so it can't see anything published after its cutoff and it can invent facts that sound convincing. Kevin's guidance is to treat its output like anything else you read online: Verify before you trust it.

## Why it matters

A static pretrained model can confidently state wrong or outdated facts about your brand, category, or pricing. Hallucination is the biggest risk in using generative AI, and the rates are high enough to matter: A Stanford study found 69% to 88% for legal queries, with broad estimates of 15-20% for older models like GPT-3 and around 3% for GPT-4 class models. A model with frozen knowledge has no way to correct itself between training runs.

Say a mid-market B2B SaaS company launches a new pricing tier. A static pretrained model trained before the launch keeps recommending the old plan and misstates the price to buyers who ask, and it stays wrong until the model is retrained or given live retrieval.

## How to use this knowledge

Assume a static model's answers about you can be stale or invented, and check what the major models say about your brand. Favor AI surfaces that use live retrieval, where you can influence the pages that get pulled in, since a purely static model can't see your latest content. Verify model output before acting on it, especially for high-stakes or fast-moving facts. Keep your most authoritative, widely-referenced content accurate so the next training run learns the right version of you.

## Growth Memo guidance

> LLMs can make things up in a very convincing way.
> — [#6 - The state of generative AI for SEO (PT.2)](https://www.growth-memo.com/p/the-state-of-generative-ai-pt2)

> The key point is to not trust LLM output blindly, just like we shouldn't trust everything we read on the internet.
> — [#6 - The state of generative AI for SEO (PT.2)](https://www.growth-memo.com/p/the-state-of-generative-ai-pt2)

## Related concepts

- **Hallucination** — the convincing but false output a static model produces when it fills gaps from memory.
- **Knowledge cutoff** — the training date past which a static model knows nothing.
- **Retrieval-augmented generation (RAG)** — feeding live sources to a model so it isn't limited to frozen knowledge.
- **Grounding** — attaching answers to retrieved sources, which lowers reliance on pretrained memory.

## Referenced in these Growth Memos

- [#6 - The state of generative AI for SEO (PT.2)](https://www.growth-memo.com/p/the-state-of-generative-ai-pt2)

---

# Supply and demand (in SEO context)

**Suggested section**
SEO

**Subtitle**
Marketplace SEO lives or dies on connecting supply, your listings, to demand, the queries buyers actually search.

**Meta title**
What is supply and demand in SEO?

**Meta description**
In marketplace SEO, supply is your inventory of listings and demand is buyer search queries; the strategy connects the two at the moment of decision.

## What it means

In SEO, supply is the inventory your site can turn into pages (product listings, profiles, service pages, reviews) and demand is the search queries and buyer interest pointing at that inventory. Kevin defines marketplace SEO as optimizing a site based on its inventory (supply) so potential buyers (demand) land on your marketplace right when they are about to decide. Your job is to match the two: Build pages from what you supply that answer what people demand.

## Why it matters

Marketplace SEO lives or dies on connecting supply, your listings, to demand, the queries buyers actually search. When your inventory maps cleanly onto real queries, each listing becomes a landing page for buyers at the decision point. When supply and demand drift apart, you either generate pages nobody searches for or miss queries you have no page to answer. Kevin's rule is that the product itself supplies the data that produces pages, so growth comes from expanding and structuring supply against proven demand.

Say a mid-market B2B SaaS review marketplace holds 5,000 product profiles (supply) but ranks for only 1,200 "best [category] software" queries (demand). Mapping the missing categories to new profile pages and enriching them lifts qualified organic entries by 30% in 2 quarters, because supply finally covers the demand.

## How to use this knowledge

Inventory your supply: List every entity your site can turn into a page (products, profiles, locations, categories). Size the demand: Pull search volume and query variations for those entities, then find where demand exists but you have no page, and where you have pages with no demand. Build supply-driven pages programmatically from your data, and enrich thin or duplicate listings with real user content so they earn relevance. Treat this as product growth: Work with product and engineering to expand inventory where demand is proven.

## Growth Memo guidance

> Marketplace SEO is the practice of optimizing a site based on its inventory (supply) so that potential buyers (demand) land directly on your marketplace when they're about to make a decision.
> — [Effective Marketplace SEO is more like Product Growth](https://www.growth-memo.com/p/effective-marketplace-seo-is-more)

> For supply-driven product-led SEO, remember: The product itself "supplies" data. That's the content that produces pages for optimization.
> — [3 examples of product-led SEO](https://www.growth-memo.com/p/5-examples-of-product-led-seo)

> Sites fall into one of two buckets: They're either inventory or content-driven.
> — [Great product features are key to SEO traffic for large sites](https://www.growth-memo.com/p/great-product-features-are-key-to-seo-traffic-for-large-sites)

## Related concepts

- **Marketplace SEO** — the practice of turning inventory (supply) into pages that meet buyer queries (demand).
- **Product-led SEO** — using the product's own data as the supply that generates ranking pages.
- **Programmatic SEO** — templated page generation that scales supply across large query sets.
- **Search demand** — the query volume and buyer intent that tells you which supply is worth building pages for.
- **User-generated content** — reviews and commentary that differentiate supply-driven pages and add relevance.

## Referenced in these Growth Memos

- [Effective Marketplace SEO is more like Product Growth](https://www.growth-memo.com/p/effective-marketplace-seo-is-more)
- [3 examples of product-led SEO](https://www.growth-memo.com/p/5-examples-of-product-led-seo)
- [Great product features are key to SEO traffic for large sites](https://www.growth-memo.com/p/great-product-features-are-key-to-seo-traffic-for-large-sites)

---

# Time to value

**Suggested section**
Foundations

**Subtitle**
The clicks that still come through are higher intent and harder to earn, so wasting them on a slow landing experience is expensive.

**Meta title**
What is time to value?

**Meta description**
Time to value is the time between a user's first contact with your page and the moment they get the value they came for.

## What it means
Time to value is the time between a user's first contact with your product or page and the moment they get the value they came for. It comes from product and UX, where the job is getting a user to the "Aha Moment," the point where they realize the value and stick around, as fast as possible. In search, the same clock now runs on the click: When someone leaves an answer engine and lands on your site, time to value is how quickly that page delivers what the query promised.

## Why it matters
Clicks from search are getting scarcer as AI Overviews (AIOs) and chat answers resolve more queries without a visit. The clicks that still come through are higher intent and harder to earn, so wasting them on a slow, cluttered, or off-topic landing experience is expensive. A user who does not reach value quickly bounces back to the answer engine, and you lose the one chance that scarce click gave you.

Say a mid-market B2B SaaS company earns a click from an AI answer to a comparison page buried under a newsletter popup and 400 words of preamble. Moving the answer the searcher wanted above the fold cuts time to value and lifts the page's trial conversion from 2% to 5%.

## How to use this knowledge
Treat the landing page as the continuation of the answer the user just read, and put the payoff first instead of making them scroll past intros and popups. For each priority query, define the exact value the searcher expects and measure how many seconds and clicks it takes to reach it. Cut the friction between arrival and value with a fast load, a headline that matches the query, and the core answer above the fold. Because clicks are scarcer, fix the pages that receive real answer-engine and organic traffic first.

## Growth Memo guidance
> Your job as a Growth Marketer is getting users to the "Aha Moment" as quickly as possible.
> — [Why product market-fit is so important for Growth Marketing](https://www.growth-memo.com/p/why-product-market-fit-is-so-important-for-growth-marketing)

> The "Aha Moment" is the moment a user realizes the value of a product and retains.
> — [Why product market-fit is so important for Growth Marketing](https://www.growth-memo.com/p/why-product-market-fit-is-so-important-for-growth-marketing)

> Outcomes trump insights. In 2025, the value of AI is getting things done.
> — [The Alpha is not LLM monitoring](https://www.growth-memo.com/p/the-alpha-is-not-llm-monitoring)

## Related concepts
- **Aha Moment** — the activation milestone time to value races toward.
- **Zero-click search** — the trend that makes each remaining click, and its time to value, more valuable.
- **Bounce rate** — the symptom you see when time to value is too long and users return to the results.
- **Activation** — the funnel stage where fast time to value turns a visit into a retained user.

## Referenced in these Growth Memos
- [Why product market-fit is so important for Growth Marketing](https://www.growth-memo.com/p/why-product-market-fit-is-so-important-for-growth-marketing)
- [The Alpha is not LLM monitoring](https://www.growth-memo.com/p/the-alpha-is-not-llm-monitoring)

---

# Topic graph

**Suggested section**
SEO

**Subtitle**
Search engines reward sites that cover a topic completely, judging your coverage of relevant entities and questions against their own model.

**Meta title**
What is a topic graph?

**Meta description**
A topic graph is the connected structure of topics, subtopics, and entities your site covers, mapped as a network instead of separate keywords.

## What it means

A topic graph is the connected structure of the topics, subtopics, and entities your site covers, and how they relate. Instead of treating each keyword as a separate target, you map a subject area as a network and build content that covers it. Kevin frames this as topic-first SEO: The topic has replaced the keyword as the atomic unit of organic authority.

## Why it matters

Search engines reward sites that cover a topic completely, comparing how well you cover relevant entities and questions against their own model of the topic. LLMs do the same when they judge authority and choose citations. A topic graph shows you where your coverage runs deep and where the gaps are, so you can report on topics as a whole instead of tracking scattered keywords.

Say a mid-market B2B SaaS company tracks 400 keywords as separate rows and can't tell why rankings stall. Mapped as a topic graph, the same data shows deep coverage of 2 parent topics but 6 missing subtopics competitors own. Filling those gaps lifts the whole cluster, and topical visibility climbs 30% in a quarter.

## How to use this knowledge

Build a topic map: Lock your parent topics, then fan out subtopics by persona, funnel stage, and the problems you pull from sales calls and community threads. Group content into clusters and track topic performance as a whole rather than keyword by keyword. Measure coverage against competitors to find the gaps worth filling. Assign SMEs or trained writers to own each core topic so depth compounds over time.

## Growth Memo guidance

> Topics are the foundation and framing of your site's organic authority and visibility.
> — [Topic-first SEO: The smarter way to scale authority](https://www.growth-memo.com/p/topic-first-seo-the-smarter-way-to)

> A topic matrix is a strategic method that compiles your brand's key topics, subtopics, and content formats needed to cover a subject area for search visibility.
> — [Operationalizing your topic-first SEO strategy](https://www.growth-memo.com/p/operationalizing-your-topic-first)

> Topical relevance, how completely a site covers related entities and questions, is a real and important factor in ranking.
> — [How to measure topical authority](https://www.growth-memo.com/p/how-to-measure-topical-authority)

## Related concepts

- **Topic clusters** — the content structure that builds and reinforces a topic graph.
- **Topical authority** — the ranking signal a complete topic graph earns.
- **Entity relationships** — the connections between people, places, and things a topic graph encodes.
- **Topic map** — Kevin's working document of parent topics and subtopics, the practical form of a topic graph.

## Referenced in these Growth Memos

- [Topic-first SEO: The smarter way to scale authority](https://www.growth-memo.com/p/topic-first-seo-the-smarter-way-to)
- [Operationalizing your topic-first SEO strategy](https://www.growth-memo.com/p/operationalizing-your-topic-first)
- [Personas are critical for AI search](https://www.growth-memo.com/p/personas-are-critical-for-ai-search)
- [#4 IBM](https://www.growth-memo.com/p/ibm)
- [How to measure topical authority](https://www.growth-memo.com/p/how-to-measure-topical-authority)
- [A better approach to keyword research for content marketing](https://www.growth-memo.com/p/a-better-approach-to-keyword-research-for-content-marketing)

---

# Topical relevance

**Suggested section**
SEO

**Subtitle**
The doc leaks moved topical authority from a buzzword you cannot defend to a ranking factor you can point to.

**Meta title**
What is topical relevance?

**Meta description**
Topical relevance is how completely your site covers the related entities and questions in a topic, the term Google's signals use for topical authority.

## What it means

Topical relevance is how completely your site covers the related entities and questions that make up a topic. It is the term Google's own signals use for what SEOs long called topical authority. Kevin points to internal doc leaks and public signals from Google showing topical relevance is a real ranking factor: The more fully you cover a topic's entities and questions, the stronger your position. He treats topical relevance and topical authority as the same idea, with topical relevance being the name grounded in Google's documentation.

## Why it matters

The doc leaks moved topical authority from a buzzword you cannot defend to a ranking factor you can point to. Kevin spent years skeptical, calling topical authority an SEO ghost concept people used to justify link-building or content-depth plays, until the leaks showed topical relevance is real and measurable. That matters when you ask for content budget, and it matters more now: In the era of AI Overviews (AIOs) and LLM-powered snippets, how completely you cover a topic decides whether you earn the click or get buried.

Say a mid-market B2B SaaS company has 30 pages on its core topic but leaves half the related subtopics uncovered. A competitor covering 90% of the topic's entities outranks it across the board. When the first company closes those gaps, its rankings on 40 queries it never directly targeted climb from page 3 to page 1, because Google reads the fuller coverage as stronger relevance.

## How to use this knowledge

Stop pitching topical authority as a vague idea. Frame it as topical relevance, the coverage of entities and questions Google's signals reward, and measure your coverage against the topic. Map the entities, subtopics, and questions that define your topic, then close the gaps with content and internal links. Pair topical relevance with brand authority, which Kevin calls a close cousin, since both shape whether you earn clicks in AIOs and snippets.

## Growth Memo guidance

> Internal docs leaks and public signals from Google show that topical relevance, ie, how completely a site covers related entities and questions, is a real and important factor in ranking.
> — [How to measure topical authority](https://www.growth-memo.com/p/how-to-measure-topical-authority)

> Topical Authority is one of many SEO ghost concepts that are often used to justify recommendations.
> — [Topical Authority: myth or reality?](https://www.growth-memo.com/p/topical-authority-myth-or-reality)

> The idea behind Topical Authority is by covering all aspects of a topic, sites get a ranking boost.
> — [Topical Authority: myth or reality?](https://www.growth-memo.com/p/topical-authority-myth-or-reality)

## Related concepts

- **Topical authority** — the older SEO name for the same idea, now grounded in Google's term topical relevance.
- **Brand authority** — what Kevin calls a close cousin, shaping clicks in AIOs and snippets.
- **Entity coverage** — how completely you address the entities that define a topic, which topical relevance measures.
- **Query share** — a way to quantify how much of a topic's demand your coverage captures.
- **Content clusters** — the structure for covering a topic's entities and questions.

## Referenced in these Growth Memos

- [How to measure topical authority](https://www.growth-memo.com/p/how-to-measure-topical-authority)
- [Topical Authority: myth or reality?](https://www.growth-memo.com/p/topical-authority-myth-or-reality)

---

# Traffic cannibalization

**Suggested section**
SEO

**Subtitle**
The biggest misconception about cannibalization is that it happens at the keyword level; it actually happens at the intent level.

**Meta title**
What is traffic cannibalization?

**Meta description**
Traffic cannibalization is when your own pages compete for the same user intent and split rankings and clicks, instead of one page winning them.

## What it means
Traffic cannibalization is when your own pages compete for the same user intent, so they split rankings and clicks instead of one page winning them. Kevin argues it happens at the user intent level, not the keyword level. That is why he calls it content cannibalization rather than keyword cannibalization. A second, external form is growing: AI search surfaces like Google's AI Overviews (AIOs) cannibalize the clicks that used to reach your site at all.

## Why it matters
The biggest misconception about cannibalization is that it happens at the keyword level; it actually happens at the intent level. Cannibalization hurts organic traffic and revenue, and it is hard to detect because it shifts over time and exists across degrees of overlap. When Google updates its read of intent in a core update, 2 pages that used to coexist can suddenly collide. The external angle matters more each quarter: Google is cannibalizing its own results with AI, which devours the referral traffic you depend on. Say a DTC skincare brand publishes 2 posts, "best moisturizer for dry skin" and "how to treat dry skin," that serve the same intent. Their cosine similarity is 0.82, so Google keeps swapping which one ranks and neither breaks the top 5. Merging them into one page moves the URL from position 8 to position 3 and doubles its clicks.

## How to use this knowledge
Detect overlap at the intent level rather than the keyword level, using content or cosine similarity (above 0.7 is high, below 0.5 is low). Decide the fix by degree of overlap: Consolidate and redirect near-duplicates, or differentiate pages that serve genuinely different intent. Set content governance so new pages do not recreate the problem. Watch the AI side too, tracking impressions against clicks as AIOs absorb queries you used to win.

## Growth Memo guidance
> The biggest misconception about cannibalization is that it happens on the keyword level. It's actually happening on the user intent level.
> — [Solving keyword cannibalization with AI {free workflow}](https://www.growth-memo.com/p/an-ai-powered-workflow-to-solve-content)

> Adding an AI chatbot to Search would mean Google would cannibalize itself, which only a few companies in history successfully accomplished.
> — [How to cannibalize your own product well](https://www.growth-memo.com/p/why-product-cannibalization-can-be-a-good-idea-and-why-not)

> Product cannibalization, or market cannibalization, is often seen as something bad, but it can be good or even necessary.
> — [Standby as Google cannibalizes itself (while also devouring all of us)](https://www.growth-memo.com/p/standby-as-google-cannibalizes-itself)

## Related concepts
- **Content cannibalization** — Kevin's preferred name, since the overlap is defined by user intent, not keywords.
- **Cosine similarity** — the metric you use to grade how badly 2 pages overlap.
- **AIOs** — the external form of cannibalization, absorbing clicks before they ever reach you.
- **Core updates** — they redraw intent, turning coexisting pages into competitors overnight.

## Referenced in these Growth Memos
- [Standby as Google cannibalizes itself (while also devouring all of us)](https://www.growth-memo.com/p/standby-as-google-cannibalizes-itself)
- [Solving keyword cannibalization with AI {free workflow}](https://www.growth-memo.com/p/an-ai-powered-workflow-to-solve-content)
- [How to cannibalize your own product well](https://www.growth-memo.com/p/why-product-cannibalization-can-be-a-good-idea-and-why-not)

---

# Traffic per query

**Suggested section**
SEO

**Subtitle**
Ranking for a query is not the same as getting traffic from it.

**Meta title**
What is traffic per query?

**Meta description**
Traffic per query estimates how much traffic a single query sends you, based on its search volume and the clicks you capture at your rank.

## What it means

Traffic per query is an estimate of how much traffic a single query sends you, based on its search volume and the clicks you actually capture at your rank and SERP layout. You use it to turn a flat list of queries into a weighted view of value: Query share and the payoff of covering a topic depend on the traffic each query brings, not the count of queries you rank for.

## Why it matters

Ranking for a query is not the same as getting traffic from it. Zero-click results, SERP Features, and AI Overviews (AIOs) cut the clicks a query sends, so 2 pages at the same position can earn very different traffic. Kevin's Featured Snippet case study showed traffic climbing 17% on mobile and 21% on desktop while rankings stayed flat. Weighting coverage by traffic per query stops you from investing in queries that rank but do not pay.

Say a mid-market B2B SaaS company ranks in the top 3 for 500 topic queries. A traffic-per-query estimate shows the top 50 queries drive 80% of clicks, while 200 long-tail queries send almost none because AIOs answer them. The team moves budget from more long-tail coverage to defending the 50 queries that pay, and organic clicks hold flat even as query count drops.

## How to use this knowledge

Estimate traffic per query as search volume times an expected click-through rate that reflects your rank and the SERP Features present. Weight query share and topical coverage by traffic per query, so a big count of low-traffic queries does not inflate perceived value. Recheck it as SERPs change, since zero-click features and AIOs can drop traffic per query even when your rank holds. In ROI cases, narrow traffic predictions to the queries and page templates that actually move.

## Growth Memo guidance

> The more results Google shows, the more they're clicked.
> — [What 20,000 keywords say about Google’s first page](https://www.growth-memo.com/p/what-20-000-keywords-say-about-googles-first-page)

> Our FS traffic started to pick up (mobile: +17%, desktop: +21%) even though our rankings stayed flat.
> — [A case study of 2,000 Featured Snippets about deduplicating](https://www.growth-memo.com/p/a-case-study-of-2000-featured-snippets-about-deduplicating)

> Short-head keyword demand is in permanent decline and likely contributing to slowed traffic growth or decline.
> — [The Great Decoupling](https://www.growth-memo.com/p/the-great-decoupling)

## Related concepts

- **Query share** — the metric traffic per query feeds; your slice of a topic's total available traffic.
- **Zero-click searches** — queries where SERP Features answer on the page and traffic per query falls to near zero.
- **Click-through rate** — the multiplier that turns a query's search volume into a traffic estimate.
- **Topical authority** — broad coverage only pays when the covered queries carry real traffic per query.
- **SERP coverage** — breadth across queries that you weight by traffic per query to judge its worth.

## Referenced in these Growth Memos

- [What 20,000 keywords say about Google’s first page](https://www.growth-memo.com/p/what-20-000-keywords-say-about-googles-first-page)
- [A case study of 2,000 Featured Snippets about deduplicating](https://www.growth-memo.com/p/a-case-study-of-2000-featured-snippets-about-deduplicating)
- [The ROI of SEO - how to predict traffic and revenue](https://www.growth-memo.com/p/the-roi-of-seo-how-to-predict-traffic-and-revenue)
- [The Great Decoupling](https://www.growth-memo.com/p/the-great-decoupling)
- [Is your organic traffic eroding?](https://www.growth-memo.com/p/is-your-organic-traffic-eroding)
- [The impact of AI Mode on SEO -analysis of 10 studies](https://www.growth-memo.com/p/the-impact-of-ai-mode-on-seo-analysis)

---

# User intent alignment

**Suggested section**
SEO

**Subtitle**
As search shifts to AI Mode, exact keywords matter less and matching the intent behind a prompt is what earns you visibility.

**Meta title**
What is user intent alignment?

**Meta description**
User intent alignment is matching your content to the goal behind a query, the top factor for ranking and now for getting surfaced in AI answers.

## What it means

User intent alignment is matching your content to the goal behind a query rather than to its exact keywords. Kevin calls meeting user intent one of the most critical ranking factors in Google; a page won't rank without fulfilling the right intent. In AI search the point is sharper: Only 6% of AI Overviews (AIOs) contain the exact search query, so the model rewards content that satisfies the underlying goal, not the wording.

## Why it matters

As search shifts to AI Mode, exact keywords matter less and matching the intent behind a prompt is what earns you visibility. Google's query fan-out breaks a prompt into parts and assembles an answer from the best pieces, so your content has to satisfy a spread of related intents instead of a single keyword. Content aligned to that goal enters more of those retrievals and gets surfaced, which AIOs and AI Mode reward even more than classic rankings did.

Say a mid-market B2B SaaS company ranks for its head term, but its page only answers the literal query. In AI Mode, the fan-out spawns 8 sub-questions about pricing, security, and alternatives, and the page satisfies 2 of them. The team expands the page to cover the full intent set, and its citation rate across those prompts rises from 2 of 8 to 6 of 8.

## How to use this knowledge

Read the intent Google already rewards by studying the top results, SERP features, and the AIO, then match or beat it. Use content tuning: Publish, then monitor which queries Google tries to rank you for and expand to cover them. Map intent to the buyer journey and to your personas so each stage's questions are answered. Optimize for query fan-out by covering the cluster of related sub-questions a prompt spawns, not one head keyword.

## Growth Memo guidance

> Only 6% of AIOs contain the search query, so meeting user intent in the content is much more important.
> — [AI on Innovation - part 2](https://www.growth-memo.com/p/ai-on-innovation-part-2)

> Without fulfilling the right user intent, a page won't rank.
> — [User intent mapping on steroids](https://www.growth-memo.com/p/user-intent-mapping-steroids)

> Google's AI Mode, and the query fan-out process, mirrors how humans think, breaking a question into parts and piecing together the best information to solve a need.
> — [Query fan-out](https://www.growth-memo.com/p/query-fan-out)

## Related concepts

- **Query fan-out** — the AI Mode process that splits a prompt into sub-questions, each with its own intent to satisfy.
- **Search intent** — the goal behind a query, the thing alignment is measured against.
- **Content tuning** — publishing, then expanding a page to match the queries Google actually ranks it for.
- **SEO personas** — the lens that tells you which intents and questions matter to your buyers.
- **Topic clusters** — the structure that lets you cover a spread of related intents around a theme.

## Referenced in these Growth Memos

- [What is User Intent? How to optimize for it like a pro](https://www.growth-memo.com/p/what-is-user-intent-how-to-optimize-for-it-like-a-pro)
- [User intent mapping on steroids](https://www.growth-memo.com/p/user-intent-mapping-steroids)
- [AI on Innovation - part 2](https://www.growth-memo.com/p/ai-on-innovation-part-2)
- [Query fan-out](https://www.growth-memo.com/p/query-fan-out)
- [Butterfly Effect](https://www.growth-memo.com/p/butterfly-effect)
- [Trust Still Lives in Blue Links](https://www.growth-memo.com/p/trust-still-lives-in-blue-links)
- [Personas are critical for AI search](https://www.growth-memo.com/p/personas-are-critical-for-ai-search)

---

# User-generated content (UGC)

**Suggested section**
SEO

**Subtitle**
User-generated content is what turns a thin, templated marketplace page into something a search engine treats as unique.

**Meta title**
What is user-generated content (UGC)?

**Meta description**
User-generated content (UGC) is content your users create, like reviews, comments, and seller descriptions, that differentiates thin marketplace pages.

## What it means

User-generated content (UGC) is content your users create rather than your team: Reviews, comments, seller and product descriptions, profiles, and templates. In marketplace and product-led SEO, UGC is the primary fuel that fills pages at scale. Kevin describes user contributions (templates, profiles, reviews) becoming the main SEO input for aggregator sites, where the community, not a content team, produces the pages.

## Why it matters

Marketplace pages are easy to generate and easy to duplicate, which makes them thin by default. User-generated content is what turns a thin, templated marketplace page into something a search engine treats as unique. Kevin notes that marketplaces rely on user-generated reviews and commentary, plus the right technical signals, to build trust and relevance. Without UGC, two competing listings look identical; with it, each page carries original detail that only your users could supply.

Say a mid-market B2C booking marketplace runs 20,000 near-identical venue pages. Adding verified reviews, photos, and Q&A to each one lifts them out of thin-content territory, and organic entries to those pages rise 25% as the once-duplicate templates start ranking on their own detail.

## How to use this knowledge

Find your thin and duplicate pages first, the templated listings with little original text, and target them for UGC. Build product loops that generate UGC: Prompt for reviews, ratings, photos, and Q&A at the right moment in the user journey. Surface and structure the UGC on the page (visible reviews, aggregate ratings, markup) so both users and search engines can read it. Moderate for quality and trust, since low-value or spammy UGC hurts more than it helps.

## Growth Memo guidance

> With UGC-based PLSEO, user contributions (templates, profiles, reviews) become the primary SEO fuel.
> — [3 examples of product-led SEO](https://www.growth-memo.com/p/5-examples-of-product-led-seo)

> Marketplaces often rely on user-generated reviews and commentary (plus the right technical signals) to build trust and relevance.
> — [Effective Marketplace SEO is more like Product Growth](https://www.growth-memo.com/p/effective-marketplace-seo-is-more)

## Related concepts

- **Marketplace SEO** — the practice UGC powers by differentiating inventory pages at scale.
- **Product-led SEO** — the model where user contributions, not a content team, produce the ranking pages.
- **Thin content** — the low-value, duplicate pages UGC is meant to fix.
- **Supply and demand** — UGC enriches the supply side so listings can meet buyer demand credibly.
- **Topical authority** — the trust and relevance signal that reviews and commentary help build.

## Referenced in these Growth Memos

- [3 examples of product-led SEO](https://www.growth-memo.com/p/5-examples-of-product-led-seo)
- [Effective Marketplace SEO is more like Product Growth](https://www.growth-memo.com/p/effective-marketplace-seo-is-more)

---

# Zero volume search queries

**Suggested section**
SEO

**Subtitle**
Search volume is averaged and often wrong, so a reported zero can hide steady, high-intent searches.

**Meta title**
What are zero volume search queries?

**Meta description**
Zero volume search queries are searches that keyword tools report as having little or no measurable volume, yet real users still search them.

## What it means
Zero volume search queries are searches that keyword tools report as having little or no measurable volume, yet real users still type or speak them. They are mostly long-tail, conversational, and question-based. Tools miss them because volume estimates are averaged, rounded, and slow to catch new or fragmenting demand.

## Why it matters
Short-head demand is in permanent decline as it fragments into thousands of specific, low-volume queries, and more of those queries now happen inside AI interfaces like AI Mode and AI Overviews (AIOs). If you pick targets only by reported volume, you skip the long-tail queries that collectively hold the demand and that these systems actually pull from. Search volume is averaged and often wrong, so a reported zero can hide steady, high-intent searches.

Say a mid-market B2B SaaS company builds content only for keywords above 100 monthly searches and skips the "0 volume" questions its buyers ask. A competitor answers 200 of those questions, gets cited across AI answers and long-tail SERPs, and captures qualified traffic the volume-gated team never saw.

## How to use this knowledge
Mine query refinements, autosuggest, People Also Ask, and real customer conversations to find zero volume queries your tools do not list. Judge topics by intent and buyer relevance, not by the volume number alone, and cluster related low-volume queries into pages that answer the full question set. Because AI Mode fans one prompt into many sub-queries, build content that covers those variations so you stay retrievable even when each query shows no volume. Confirm the demand in Search Console, where zero volume queries surface as real impressions.

## Growth Memo guidance
> Demand did not disappear, it atomized into thousands of specific queries.
> — [The Great Decoupling](https://www.growth-memo.com/p/the-great-decoupling)

> Query refinements lead to search queries without search volume.
> — [Query refinements](https://www.growth-memo.com/p/query-refinements)

> Search volume is a double-edged sword. On one side, it guides our strategy. On the other side, it's so fundamentally flawed that we should handle it with extreme care.
> — [The inaccuracy and flaws of search volume](https://www.growth-memo.com/p/the-inaccuracy-and-flaws-of-search-volume)

## Related concepts
- **Long-tail keywords** — the specific, low-volume queries most zero volume searches fall into.
- **Query fan-out** — the AI Mode process that expands one prompt into many zero-volume sub-queries.
- **Topical authority** — the coverage you build by answering the full set of low-volume questions in a topic.
- **Search volume** — the flawed metric whose blind spots create the zero-volume category in the first place.
- **Zero-click search** — a separate trend often confused with zero volume, where a search resolves without a click.

## Referenced in these Growth Memos
- [The inaccuracy and flaws of search volume](https://www.growth-memo.com/p/the-inaccuracy-and-flaws-of-search-volume)
- [The Great Decoupling](https://www.growth-memo.com/p/the-great-decoupling)
- [Query refinements](https://www.growth-memo.com/p/query-refinements)
- [Query fan-out](https://www.growth-memo.com/p/query-fan-out)
- [How to measure topical authority](https://www.growth-memo.com/p/how-to-measure-topical-authority)

---

# Agent readiness

**Suggested section**
SEO

**Subtitle**
AI agents now browse and buy for users; if your site blocks or confuses them, the agent fails and the user never sees you.

**Meta title**
What is agent readiness?

**Meta description**
Agent readiness is how well your site lets an AI agent find, read, extract, and act on your content without getting stuck.

## What it means

Agent readiness is how well your website lets an AI agent do its job: Find the page, read it, pull out the exact answer, and act on it, whether that is citing you, booking, or buying. It measures your site from the machine's point of view rather than a human visitor's.

## Why it matters

AI agents now browse, compare, and transact for people. If your site hides content behind heavy JavaScript, unlabeled buttons, or anti-bot walls, the agent fails and the user never sees you. This is the infrastructure shift Kevin covers as the Agentic Web, where agents, not human visitors, do much of the work.

Say a mid-market B2B SaaS company keeps its pricing inside an interactive widget that only renders for logged-in users. When an AI agent tries to answer "what does this tool cost," it can't extract a number, so it cites a competitor's clean pricing page instead. Expose the price as plain text and the agent cites you, turning a miss into a recommendation.

## How to use this knowledge

Test your key pages the way an agent sees them: Fetch the raw HTML and check whether the answer (a price, a spec, an integration) is present as plain text rather than locked in script or images. Fix the blockers with clean semantic HTML, labeled buttons, no CAPTCHAs on informational pages, and action schema where an agent needs to transact. Start with the pages agents hit most: Pricing, integrations, security, and docs.

## Growth Memo guidance

> We're looking at the infrastructure of the Agentic Web and the inversion of traditional internet economics.
> — [Growth Intelligence Brief #14](https://www.growth-memo.com/p/growth-intelligence-brief-14)

## Related concepts

- **Agentic Web** — the emerging web where AI agents, not people, do much of the browsing and buying.
- **Agentic commerce** — agents transacting on a user's behalf, which raises the stakes for agent readiness.
- **Optimizing for ingestion** — getting your data into an agent's answer instead of chasing human clicks.
- **Action schema** — the structured markup that lets an agent run a task on your site.

## Referenced in these Growth Memos

- [Growth Intelligence Brief #14](https://www.growth-memo.com/p/growth-intelligence-brief-14)
