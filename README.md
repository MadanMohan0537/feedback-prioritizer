# Feedback Prioritizer

**Real-time customer feedback analysis with topic modeling, sentiment scoring, and PM-grade prioritization.**

A streaming pipeline that ingests customer feedback (reviews, tickets, social mentions), automatically identifies what customers are talking about and how they feel about it, and surfaces a ranked, actionable issue/feature-request list on a live Streamlit dashboard.

## Table of contents

- [Problem statement](#problem-statement)
- [Who this is for](#who-this-is-for)
- [Demo](#demo)
- [How it works](#how-it-works)
- [Quickstart](#quickstart)
- [Repo structure](#repo-structure)
- [The prioritization formula](#the-prioritization-formula)
- [Design decisions](#design-decisions--tradeoffs)
- [Known limitations](#known-limitations-and-how-theyd-be-addressed-in-production)
- [Extending to production](#extending-to-production)
- [Tech stack](#tech-stack)

## Problem statement

Feedback about a product arrives everywhere — app store reviews, support tickets, tweets, in-app surveys — and by the time a PM manually reads enough of it to notice a pattern, the pattern is usually already a fire. This tool closes that loop: it clusters incoming feedback into topics automatically, scores sentiment per message, and computes a priority score that blends *how often* something is mentioned, *how negative* it is, and *how much it matters to the business* — a UI nitpick and a billing double-charge are not the same problem even at equal volume.

## Who this is for

- **Product managers** who want to know, this week, what the top three things breaking or delighting customers are — without reading 600 reviews.
- **Support leads** who want an emerging issue (a spike in "crashes") surfaced before ticket volume explodes.
- **Eng leads** who want feature requests separated from bug reports and ranked by demand, not by whoever shouted loudest in Slack.

**Success looks like:** time-to-detect an emerging issue dropping from days (waiting for a manual ticket-trend report) to hours; more roadmap decisions citing an aggregated feedback signal instead of anecdote; fewer duplicate tickets triaged by hand because clustering already grouped them.

## Demo

Run `streamlit run app.py` and use the sidebar to replay the included sample dataset (650 synthetic reviews for a fictional productivity app, "TaskFlow," across 14 days). **Play** streams messages in continuously; **Step** advances one batch at a time so you can watch scores update. The dashboard has four views:

- **Prioritized Issues & Requests** — ranked bar charts and expandable cards, each backed by real example messages
- **Live Feed** — the raw stream as it's scored, color-coded by sentiment
- **Trends** — sentiment distribution, topic frequency over time, sentiment over time
- **Topic Explorer** — a word cloud per discovered topic

## How it works

- **`frequency_norm`** — the topic's message count, min-max normalized against every other topic currently in view, so volume is relative rather than absolute.
- **`negativity_norm`** — derived from average sentiment polarity (`-1` to `+1`), rescaled so *more negative* topics score *higher*. This is a complaint prioritizer: negative + frequent + high-impact bubbles to the top.
- **`impact_norm`** — the average business-impact weight of the topic's messages, from a keyword match (`crash`, `billing`, `security` → 1.0 severe; `slow`, `bug`, `sync` → 0.6 moderate; `ui`, `design` → 0.3 minor; unmatched → 0.15 neutral default).

Default weights (`w_freq=0.40, w_sent=0.35, w_impact=0.25`) live in `src/config.py` and are also exposed as live sliders in the dashboard sidebar, so a PM can express "impact matters more than volume this week" without touching code — the ranked list re-sorts instantly, no model retrain required.

**Feature requests are tracked separately, not penalized.** A topic is classified as a feature-request cluster when >=40% of its messages match request phrases ("I wish," "please add," "would be nice," "any plans to add," etc.). These are ranked by frequency rather than negativity — "please add dark mode" is neutral-to-positive sentiment but still a strong demand signal.

## Design decisions & tradeoffs

- **Batch retraining over incremental updates.** BERTopic supports online-update patterns, but topic IDs can drift or renumber between updates in ways that are confusing on a live dashboard. Instead, the model re-fits on the rolling window every `RETRAIN_EVERY_N` new messages (default 20) — cheap at this data volume, and topic labels stay stable within a window. At production scale this step would move to an async batch job (e.g. a scheduled Spark/Airflow task) instead of running inline on the request thread.
- **Everything degrades gracefully.** A demo that hard-requires a GPU and Hugging Face network access just to boot is a bad demo. Every model-backed stage — sentiment and topic modeling — has a lexicon/statistical fallback, so the pipeline is always runnable, and the sidebar always shows which backend actually served the current session.

## Known limitations (and how they'd be addressed in production)

- **Sarcasm and mixed sentiment in one message** ("oh great, ANOTHER crash") are not reliably handled by either sentiment backend — both are trained for direct sentiment, not irony. A production system would add a sarcasm-detection pass or route ambiguous cases to human review. Similarly, "love the design but it keeps crashing" resolves to one dominant polarity score rather than two aspect-level scores; true aspect-based sentiment analysis (ABSA) would split this correctly.
- **Topic granularity depends on cluster-size parameters** (`min_topic_size` for BERTopic, `n_topics` for the fallback). Too coarse merges distinct issues together; too fine fragments one issue into near-duplicate clusters. This needs periodic human review of topic labels, especially after a product change shifts what people write about.
- **Business-impact keywords are hand-curated**, not learned. A more robust version would train impact weights from historical outcome data (e.g. correlating topics with churn or refund-request rates) rather than a static keyword list.
- **Synthetic demo data.** `sample_reviews.csv` is generated, not scraped, to avoid licensing/ToS issues with real app-store or social data in a public repo. Swapping in a real dataset only requires matching the `timestamp, text, source` schema.

## Extending to production

- **Ingestion** — replace `src/ingest.py`'s CSV replay with a Kafka consumer, or a Twitter API v2 / Reddit API / Zendesk webhook, keeping the same `{timestamp, text, source}` item shape everything downstream already expects.
- **Storage** — persist enriched messages (topic, sentiment, impact) to a real-time-queryable store (Postgres + TimescaleDB, or a warehouse with a streaming ingest path) instead of in-memory `st.session_state`.
- **Topic model serving** — move retraining to an async worker; serve the latest fitted model via a small internal API so the dashboard reads precomputed results instead of retraining inline.
- **Alerting** — when a topic's priority score crosses a threshold, auto-file a Jira ticket or post to a Slack channel (`#voice-of-customer`) with the topic, top examples, and trend chart attached, turning this from a dashboard you have to check into a system that pages you.
- **Scale** — BERTopic + sentence-transformers comfortably handles tens of thousands of messages per batch on CPU; beyond that, move embeddings to a GPU inference service and shard clustering by time window or product area.

## Tech stack

| Layer | Library |
|---|---|
| Dashboard | Streamlit, Plotly |
| Sentiment | Hugging Face Transformers (RoBERTa) -> VADER fallback |
| Topic modeling | BERTopic (Sentence-Transformers + UMAP + HDBSCAN) -> TF-IDF + KMeans fallback |
| Data handling | pandas, NumPy, scikit-learn |
| Visualization | Plotly, WordCloud, Matplotlib |

## License

No license file is included yet — add one (MIT is a reasonable default for a portfolio project) if you intend for others to reuse this code.
