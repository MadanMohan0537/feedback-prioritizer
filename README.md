# Feedback Prioritizer

Turn a pile of scattered customer feedback into a ranked, defensible "fix this first" list — automatically.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![CI](https://github.com/MadanMohan0537/feedback-prioritizer/actions/workflows/ci.yml/badge.svg)](https://github.com/MadanMohan0537/feedback-prioritizer/actions/workflows/ci.yml)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Most feedback tools either dump raw reviews on you or bury a real signal under a generic sentiment score. This one goes further: it clusters incoming feedback into topics on its own, scores how customers feel about each one, and combines that with *how often it's mentioned* and *how much it costs the business* into a single priority number. Point it at your own CSV, or replay the bundled 650-row demo dataset to see it work end to end in under a minute.

---

## Table of Contents

- [Why this exists](#why-this-exists)
- [What it does](#what-it-does)
- [Quickstart](#quickstart)
- [The prioritization formula](#the-prioritization-formula)
- [Project layout](#project-layout)
- [Design decisions](#design-decisions)
- [Honest limitations](#honest-limitations)
- [Where this goes next](#where-this-goes-next)
- [Tech stack](#tech-stack)
- [License](#license)

---

## Why this exists

A support inbox, an app-store review feed, and a Twitter mentions column all say something true about your product, but nobody reads all three every day. By the time a pattern is obvious to a human, it's usually already costing you retention. This tool is meant to sit where a PM or support lead would otherwise be manually skimming — it reads everything, groups it, and hands back a short list worth acting on today.

---

## What it does

1. **Ingests** feedback as a stream — either the included synthetic dataset (650 reviews for a fictional app, "TaskFlow," across 14 days) or a CSV you upload directly in the sidebar.
2. **Scores sentiment** on each message as it arrives, using a transformer model tuned on informal text, with an automatic fallback to a lexicon-based analyzer if no GPU/network is available.
3. **Discovers topics** on a rolling window of recent messages — no predefined category list, no manual tagging.
4. **Prioritizes** each topic with a transparent formula that blends volume, negativity, and a business-impact signal (crashes and billing bugs outrank UI nitpicks, even at equal mention count).
5. **Separates feature requests from complaints** automatically, so "please add dark mode" doesn't get buried under bug reports just because its sentiment reads neutral.
6. **Displays it all live** on a Streamlit dashboard: a ranked issue list, a live feed, trend charts, and topic word clouds.

---

## Quickstart

**Prerequisites:** Python 3.9+

```bash
git clone https://github.com/MadanMohan0537/feedback-prioritizer.git
cd feedback-prioritizer

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

pip install -r requirements.txt      # lightweight, fully functional fallback stack
streamlit run app.py
```

For transformer sentiment and BERTopic, install the optional model stack instead:

```bash
pip install -r requirements-full.txt
```

Open the local URL Streamlit prints, hit **Play** in the sidebar, and watch the priority list build itself.

**To use your own data** instead of the demo set, open the **"Use your own data"** panel in the sidebar and upload a CSV. At minimum it needs a `text` column — `timestamp` and `source` are used if present and auto-filled if not.

**No GPU required.** The full transformer + BERTopic stack is used automatically when installed and reachable; otherwise the app runs the same pipeline on lightweight fallbacks (VADER sentiment, TF-IDF/KMeans topics) with zero code changes and no loss of functionality. The active backend for each stage is always shown in the sidebar.

**Smoke test** (exercises the full pipeline without launching Streamlit):

```bash
python tests/smoke_test.py
```

**Unit tests** (exercise scoring bounds, explanations, validation, and keyword matching):

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## The prioritization formula

```
priority_score = 100 × ( w_freq   × frequency_norm
                        + w_sent   × negativity_norm
                        + w_impact × impact_norm )
```

| Component | Default weight | Description |
|---|---|---|
| `frequency_norm` | 0.40 | How often the topic comes up, relative to everything else currently in view |
| `negativity_norm` | 0.35 | How negative the topic reads on average — more negative scores higher |
| `impact_norm` | 0.25 | Business-impact keyword signal (`crash`/`billing`/`security` → 1.0 · `slow`/`bug` → 0.6 · `ui`/`design` → 0.3) |

All three weights are live sliders in the dashboard, not hardcoded — drag "business impact" up during a launch week and the ranked list re-sorts instantly, no retrain needed. The engine normalizes slider values to sum to one, so the score always remains within 0–100. Each issue also shows the point contribution from frequency, negativity, and impact, making the rank explainable. Defaults live in `src/config.py`.

**Feature requests are ranked separately.** A topic is classified as a feature-request cluster when ≥40% of its messages match request phrases ("I wish", "please add", "would be nice", …). These rank by frequency, not negativity.

---

## Project layout

```
feedback-prioritizer/
├── app.py                        # Streamlit dashboard — entry point
├── requirements.txt              # Python dependencies
├── requirements-full.txt         # Optional transformer + BERTopic stack
├── src/
│   ├── config.py                 # Every tuneable in one place
│   ├── ingest.py                 # Stream simulation + batch pull
│   ├── sentiment.py              # Transformer sentiment + VADER fallback
│   ├── topic_model.py            # BERTopic + TF-IDF/KMeans fallback
│   └── prioritize.py             # Scoring, ranking, feature-request detection
├── data/
│   ├── sample_reviews.csv        # 650-row synthetic dataset (TaskFlow app, 14 days)
│   └── generate_sample_data.py   # Script that generated the demo data
├── tests/
│   └── smoke_test.py             # End-to-end check without launching Streamlit
└── CHANGELOG.md
```

---

## Design decisions

**Batch retraining over incremental updates.** BERTopic supports online-update patterns, but topic IDs can drift or renumber between updates in ways that are confusing on a live dashboard. Instead, the model re-fits on a rolling window every `RETRAIN_EVERY_N` new messages (default 20) — cheap at this data volume, and topic labels stay stable within a window. At production scale this moves to an async batch job.

**Everything degrades gracefully.** A demo that hard-requires a GPU and Hugging Face network access just to boot is a bad demo. Every model-backed stage has a lexicon/statistical fallback so the pipeline is always runnable. The sidebar always shows which backend is active.

---

## Honest limitations

- **Sarcasm and mixed sentiment** ("love the design but it keeps crashing") aren't split correctly by either backend — both resolve to one dominant score. True aspect-based sentiment analysis would fix this.
- **Topic granularity** is sensitive to cluster-size parameters; too coarse merges distinct issues, too fine fragments one issue into near-duplicate clusters. Worth a periodic human pass on topic labels, especially after a product change shifts what customers write about.
- **Business-impact weights are hand-curated**, not learned from outcome data (churn, refunds). A production version should train these from historical correlation.
- **Synthetic demo data.** `sample_reviews.csv` is generated — not scraped — to avoid licensing/ToS issues in a public repo. Swapping in real data only requires matching the `text` / `timestamp` / `source` schema.

---

## Where this goes next

- **Alerting** — when a topic's priority score crosses a threshold, auto-file a Jira ticket or post to `#voice-of-customer` Slack, turning a dashboard you have to check into a system that pages you.
- **Live sources** — swap the CSV replay in `src/ingest.py` for a Kafka consumer, a Zendesk webhook, or the Twitter/Reddit API. Nothing downstream changes.
- **Async retraining** — move topic-model retraining to a background worker so the dashboard reads precomputed results instead of retraining inline.
- **Learned impact weights** — replace the keyword list with weights trained from historical churn/refund correlation.

---

## Tech stack

| Layer | Library |
|---|---|
| Dashboard | [Streamlit](https://streamlit.io/), [Plotly](https://plotly.com/) |
| Sentiment | Hugging Face Transformers (`cardiffnlp/twitter-roberta-base-sentiment-latest`) → VADER fallback |
| Topic modeling | [BERTopic](https://maartengr.github.io/BERTopic/) (Sentence-Transformers + UMAP + HDBSCAN) → TF-IDF + KMeans fallback |
| Data handling | pandas, NumPy, scikit-learn |
| Visualization | Plotly, WordCloud, Matplotlib |

---

## License

MIT — see [LICENSE](LICENSE) for details.
