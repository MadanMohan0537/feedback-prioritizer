# Real-Time Customer Feedback Analyzer

A streaming pipeline that ingests customer feedback (reviews, tickets, social
mentions), models what customers are talking about, scores how they feel
about it, and ranks the result into a prioritized action list — live, on a
Streamlit dashboard.

## Problem statement

Feedback about a product arrives everywhere — app store reviews, support
tickets, tweets, in-app surveys — and by the time a PM manually reads enough
of it to notice a pattern, the pattern is usually already a fire. This tool
closes that loop: it clusters incoming feedback into topics automatically,
scores sentiment per message, and computes a priority score that blends
*how often* something is mentioned, *how negative* it is, and *how much it
matters to the business* (a UI nitpick and a billing double-charge are not
the same problem even at equal volume).

## User personas

- **Product manager** — wants to know, this week, what the top three things
  breaking or delighting customers are, without reading 600 reviews.
- **Support lead** — wants emerging issues (a spike in "crashes") surfaced
  before ticket volume explodes.
- **Eng lead** — wants feature requests separated from bug reports and
  ranked by demand, not by whoever shouted loudest in Slack.

## Success metrics (how you'd know this is working in production)

- Time-to-detect an emerging issue (e.g. a crash spike) drops from days
  (waiting for a support-ticket trend report) to hours.
- % of roadmap decisions citing an aggregated feedback signal, rather than
  anecdote, increases.
- Reduction in duplicate/low-priority tickets triaged manually, because
  clustering already grouped them.

## Architecture

## Running it

```bash
pip install -r requirements.txt         # heavy deps optional, see below
python data/generate_sample_data.py     # regenerate sample_reviews.csv
streamlit run app.py
```

In the sidebar: **Play** starts the simulated stream (replays the CSV at a
configurable rate), **Step** advances one batch manually, and the
prioritization weights are live sliders — drag "Business impact" up during
a launch week to see the ranked list re-sort instantly, no retrain needed
(only topic modeling requires retraining; scoring is cheap and recomputed
on every interaction).

### Minimal install (no heavy ML deps)

The app runs with just `streamlit pandas plotly numpy scikit-learn
vaderSentiment` — sentiment falls back to VADER and topics fall back to
TF-IDF/KMeans. This is enough to demo the full pipeline end-to-end.

### Full install (BERTopic + transformer sentiment)

Add `torch transformers sentence-transformers bertopic umap-learn hdbscan`.
First run will download the embedding model (~90MB) and the sentiment model
(~500MB) from Hugging Face, so it needs network access once.

## Design decisions & tradeoffs

- **Batch retraining over incremental updates.** BERTopic supports
  `partial_fit`-style online updates, but topic IDs can drift/renumber
  between updates in ways that are confusing on a live dashboard. Instead,
  the model re-fits on a rolling window (last 250 messages) every 20 new
  messages — cheap at this volume, and topic labels stay stable within a
  window. At production scale this step moves to an async batch job (e.g.
  a scheduled Spark/Airflow task) rather than running inline.
- **Prioritization is a weighted, normalized blend, not a keyword count.**
  Raw frequency rewards noisy topics; raw negativity rewards any complaint
  equally. Normalizing each factor to 0..1 within the current window before
  blending means a rare-but-severe billing bug can outrank a common-but-mild
  UI complaint, and the weights are exposed as sliders so a PM can express
  "impact matters more than volume this week" directly.
- **Feature requests are split out, not penalized.** A message like "please
  add dark mode" is neutral-to-positive in sentiment but still a strong
  signal — it's ranked by frequency within its own list rather than folded
  into the negativity-driven issue ranking.
- **Everything degrades gracefully.** A demo/portfolio project that hard-
  requires a GPU and Hugging Face network access to even boot is a bad demo.
  Every model-backed stage has a lexicon/statistical fallback so the full
  pipeline is always runnable.

## Known limitations (and how they'd be addressed in production)

- **Sarcasm & mixed sentiment in one message** ("oh great, ANOTHER crash")
  are not reliably handled by either sentiment backend — both are trained
  for direct sentiment, not irony. A production system would add a
  sarcasm-detection pass or route ambiguous cases to human review.
  Similarly, "love the design but it keeps crashing" resolves to a single
  polarity score rather than two separate aspect-level scores; true
  aspect-based sentiment analysis (ABSA) would split this correctly.
- **Topic granularity depends on cluster-size parameters** (`min_topic_size`
  for BERTopic, `n_topics` for the fallback). Too coarse merges distinct
  issues; too fine fragments one issue into near-duplicates. This needs
  periodic human review of topic labels, especially after a big product
  change shifts what people write about.
- **Business impact keywords are hand-curated**, not learned. A more robust
  version would train impact weights from historical data (e.g. correlate
  topics with churn or refund-request rates) rather than a static keyword
  list.
- **Synthetic demo data.** `sample_reviews.csv` is generated, not scraped,
  to avoid licensing/ToS issues with real app-store or Twitter data in a
  portfolio repo. Swapping in a real dataset (Amazon reviews, App Store
  scrapes) only requires matching the `timestamp, text, source` schema.

## Extending to production

- **Ingestion:** replace `src/ingest.py`'s CSV replay with a Kafka
  consumer (or Twitter API v2 / Reddit API / Zendesk webhook), keeping the
  same `{timestamp, text, source}` item shape downstream code expects.
- **Storage:** persist enriched messages (topic, sentiment, impact) to a
  real-time-queryable store (e.g. Postgres + TimescaleDB, or a data
  warehouse with a streaming ingest path) instead of in-memory
  `st.session_state`.
- **Topic model serving:** move retraining to an async worker; serve the
  latest fitted model via a small internal API so the dashboard reads
  precomputed results instead of retraining inline.
- **Alerting:** when a topic's priority score crosses a threshold, auto-file
  a Jira ticket or post to a Slack channel (`#voice-of-customer`) with the
  topic, top examples, and trend chart attached — turning this from a
  dashboard you have to check into a system that pages you.
- **Scale:** BERTopic + sentence-transformers comfortably handles tens of
  thousands of messages per batch on CPU; beyond that, move embeddings to a
  GPU inference service and shard clustering by time window or product area.
