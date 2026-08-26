# Feedback Prioritizer

Turn a pile of scattered customer feedback into a ranked, defensible "fix this first" list — automatically.

Most feedback tools either dump raw reviews on you or bury a real signal under a generic sentiment score. This one goes a step further: it clusters incoming feedback into topics on its own, scores how customers feel about each one, and combines that with *how often it's mentioned* and *how much it costs the business* into a single priority number. Point it at your own CSV, or replay the bundled 650-row demo dataset to see it work end to end in under a minute.

## Why this exists

A support inbox, an app-store review feed, and a Twitter mentions column all say something true about your product, but nobody reads all three every day. By the time a pattern is obvious to a human, it's usually already costing you retention. This tool is meant to sit where a PM or support lead would otherwise be manually skimming — it reads everything, groups it, and hands back a short list worth acting on today.

## What it does

1. **Ingests** feedback as a stream — either the included synthetic dataset (650 reviews for a fictional app, "TaskFlow," across 14 days) or a CSV you upload directly in the sidebar.
2. **Scores sentiment** on each message as it arrives, using a transformer model tuned on informal text, with an automatic fallback to a lexicon-based analyzer if no GPU/network is available.
3. **Discovers topics** on a rolling window of recent messages — no predefined category list, no manual tagging.
4. **Prioritizes** each topic with a transparent formula that blends volume, negativity, and a business-impact signal (crashes and billing bugs outrank UI nitpicks, even at equal mention count).
5. **Separates feature requests from complaints** automatically, so "please add dark mode" doesn't get buried under bug reports just because its sentiment reads neutral.
6. **Displays it all live** on a Streamlit dashboard: a ranked issue list, a live feed, trend charts, and topic word clouds.

## Try it

```bash
git clone https://github.com/MadanMohan0537/feedback-prioritizer.git
cd feedback-prioritizer
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints, hit **Play** in the sidebar, and watch the priority list build itself. To use your own data instead of the demo set, open the **"Use your own data"** panel in the sidebar and upload a CSV with at minimum a `text` column (a `timestamp` and `source` column are used if present, and are auto-filled otherwise).

No GPU required. The full transformer + BERTopic stack is used automatically when installed and reachable; otherwise the app runs the same pipeline on lightweight fallbacks (VADER sentiment, TF-IDF/KMeans topics) with zero code changes and no loss of functionality — just simpler models. The active backend for each stage is always shown in the sidebar.

## The prioritization formula

```
priority_score = 100 × ( w_freq × frequency_norm
                        + w_sent × negativity_norm
                        + w_impact × impact_norm )
```

- `frequency_norm` — how often the topic comes up, relative to everything else currently in view.
- `negativity_norm` — how negative the topic reads on average; more negative scores higher.
- `impact_norm` — a business-impact weight from keyword signals (`crash`, `billing`, `security` score highest; `ui`, `design` score lowest).

All three weights are live sliders in the dashboard, not hardcoded — drag "business impact" up during a launch week and the ranked list re-sorts instantly, no retrain needed. Defaults live in `src/config.py`.

## What's new (v1.1)

- **Bring your own data.** Upload a CSV directly in the sidebar instead of only replaying the bundled demo set.
- **Minimum-priority filter.** A slider on the Issues tab lets you hide low-priority noise and focus the chart on what's actually worth discussing.

## Project layout

```
feedback-prioritizer/
├── app.py                        # Streamlit dashboard
├── src/
│   ├── ingest.py                 # stream simulation + batch pull
│   ├── sentiment.py              # transformer sentiment + VADER fallback
│   ├── topic_model.py            # BERTopic + TF-IDF/KMeans fallback
│   ├── prioritize.py             # scoring, ranking, feature-request detection
│   └── config.py                 # every tunable in one place
├── data/
│   ├── generate_sample_data.py   # synthetic demo dataset generator
│   └── sample_reviews.csv
├── tests/
│   └── smoke_test.py             # end-to-end check without launching Streamlit
└── requirements.txt
```

## Honest limitations

- Sarcasm and mixed-sentiment messages ("love the design but it keeps crashing") aren't split correctly by either sentiment backend — both resolve to one dominant score. True aspect-based sentiment analysis would fix this.
- Topic granularity is sensitive to cluster-size parameters; too coarse merges distinct issues, too fine fragments one issue into duplicates. Worth a periodic human pass on topic labels.
- Business-impact weights are a hand-curated keyword list, not learned from outcome data (churn, refunds). A production version should train these.
- The bundled dataset is synthetic, generated to avoid scraping-and-licensing headaches in a public repo — swap in real data by matching the `text`/`timestamp`/`source` schema.

## Where this goes next

- Slack/Jira integration so a topic crossing a priority threshold auto-files a ticket instead of waiting to be noticed.
- Swap the CSV replay for a live source — Kafka, a Zendesk webhook, the Twitter or Reddit API — without touching anything downstream of `src/ingest.py`.
- Move topic-model retraining to an async worker so the dashboard reads precomputed results instead of retraining inline.
- Learn business-impact weights from historical churn/refund correlation instead of a static keyword list.

## Stack

Streamlit + Plotly for the dashboard · Hugging Face Transformers (RoBERTa) with a VADER fallback for sentiment · BERTopic (Sentence-Transformers + UMAP + HDBSCAN) with a TF-IDF/KMeans fallback for topics · pandas/scikit-learn underneath.

## License

Not yet licensed — add one (MIT is a reasonable default) if you want others to reuse this.
