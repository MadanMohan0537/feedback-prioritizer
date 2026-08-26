"""
Real-Time Customer Feedback Analyzer -- Streamlit dashboard.

Run with:
    streamlit run app.py

The app simulates a live feedback stream (replaying data/sample_reviews.csv),
scores sentiment as each message arrives, re-runs topic modeling on a rolling
window every N messages, and surfaces a prioritized issue/feature-request
list plus supporting charts.
"""

import sys
import os
import time
from collections import Counter
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.ingest import load_feedback_rows, get_batch
from src.sentiment import get_analyzer
from src.topic_model import build_topic_model
from src.prioritize import compute_priorities, split_issues_and_requests, business_impact_score

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False


st.set_page_config(page_title="Feedback Analyzer", page_icon="📊", layout="wide")


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
def init_state():
    if "initialized" in st.session_state:
        return
    st.session_state.initialized = True
    st.session_state.all_rows = load_feedback_rows()
    st.session_state.cursor = 0
    st.session_state.history = []           # enriched items processed so far
    st.session_state.topic_info = {}
    st.session_state.priorities = []
    st.session_state.since_retrain = 0
    st.session_state.playing = False
    st.session_state.model_backend = None
    st.session_state.sentiment_backend = None


init_state()


@st.cache_resource(show_spinner=False)
def get_sentiment_analyzer():
    return get_analyzer()


def process_batch(batch):
    analyzer = get_sentiment_analyzer()
    for item in batch:
        result = analyzer.analyze(item["text"])
        item["polarity"] = result.polarity
        item["sentiment_label"] = result.label
        item["sentiment_score"] = result.score
        item["impact"] = business_impact_score(item["text"])
        item["topic_id"] = -99  # unassigned until next retrain
        st.session_state.history.append(item)
    st.session_state.sentiment_backend = analyzer._backend


def retrain_topics():
    window = st.session_state.history[-config.ROLLING_WINDOW_SIZE:]
    if len(window) < config.MIN_MESSAGES_TO_MODEL:
        return
    docs = [it["text"] for it in window]
    model = build_topic_model()
    topic_ids, topic_info = model.fit_transform(docs)
    st.session_state.model_backend = model.name

    for it, tid in zip(window, topic_ids):
        it["topic_id"] = int(tid)
    st.session_state.topic_info = topic_info

    priorities = compute_priorities(window, topic_info)
    st.session_state.priorities = priorities
    st.session_state.since_retrain = 0


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.title("📊 Feedback Analyzer")
st.sidebar.caption("Real-time topic modeling, sentiment & prioritization")

batch_size = st.sidebar.slider("Messages per tick", 1, 20, 5)
refresh_ms = st.sidebar.slider("Auto-play interval (ms)", 500, 5000, 1500, step=250)

col_a, col_b = st.sidebar.columns(2)
if col_a.button("▶ Play" if not st.session_state.playing else "⏸ Pause"):
    st.session_state.playing = not st.session_state.playing
if col_b.button("⏭ Step"):
    batch, next_idx = get_batch(st.session_state.all_rows, st.session_state.cursor, batch_size)
    if batch:
        process_batch(batch)
        st.session_state.cursor = next_idx
        st.session_state.since_retrain += len(batch)
        if st.session_state.since_retrain >= config.RETRAIN_EVERY_N:
            retrain_topics()

if st.sidebar.button("↺ Reset stream"):
    st.session_state.cursor = 0
    st.session_state.history = []
    st.session_state.topic_info = {}
    st.session_state.priorities = []
    st.session_state.since_retrain = 0
    st.session_state.playing = False

total_rows = len(st.session_state.all_rows)
st.sidebar.progress(min(st.session_state.cursor / max(total_rows, 1), 1.0))
st.sidebar.caption(f"{st.session_state.cursor} / {total_rows} messages replayed")

with st.sidebar.expander("Prioritization weights"):
    w_freq = st.slider("Frequency weight", 0.0, 1.0, config.PRIORITY_WEIGHTS["frequency"])
    w_sent = st.slider("Sentiment weight", 0.0, 1.0, config.PRIORITY_WEIGHTS["sentiment"])
    w_imp = st.slider("Business impact weight", 0.0, 1.0, config.PRIORITY_WEIGHTS["impact"])
    live_weights = {"frequency": w_freq, "sentiment": w_sent, "impact": w_imp}

if st.session_state.model_backend:
    st.sidebar.info(f"Topic model: **{st.session_state.model_backend}**")
if st.session_state.sentiment_backend:
    st.sidebar.info(f"Sentiment model: **{st.session_state.sentiment_backend}**")

# Auto-play tick
if st.session_state.playing:
    if HAS_AUTOREFRESH:
        st_autorefresh(interval=refresh_ms, key="autorefresh")
    batch, next_idx = get_batch(st.session_state.all_rows, st.session_state.cursor, batch_size)
    if batch:
        process_batch(batch)
        st.session_state.cursor = next_idx
        st.session_state.since_retrain += len(batch)
        if st.session_state.since_retrain >= config.RETRAIN_EVERY_N:
            retrain_topics()
    else:
        st.session_state.playing = False


# ---------------------------------------------------------------------------
# Recompute priorities live if weights changed (cheap, no retraining needed)
# ---------------------------------------------------------------------------
window = st.session_state.history[-config.ROLLING_WINDOW_SIZE:]
if window and st.session_state.topic_info:
    st.session_state.priorities = compute_priorities(window, st.session_state.topic_info, live_weights)


# ---------------------------------------------------------------------------
# Header KPIs
# ---------------------------------------------------------------------------
st.title("Voice of Customer — Live Command Center")

history = st.session_state.history
n_total = len(history)
n_neg = sum(1 for h in history if h.get("sentiment_label") == "negative")
avg_polarity = sum(h.get("polarity", 0) for h in history) / n_total if n_total else 0.0
n_topics = len({h["topic_id"] for h in window if h.get("topic_id", -99) not in (-1, -99)})
critical = [p for p in st.session_state.priorities if p["priority_score"] >= 60 and not p["is_feature_request"]]

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Messages processed", n_total)
k2.metric("% Negative", f"{(n_neg/n_total*100):.0f}%" if n_total else "—")
k3.metric("Avg sentiment polarity", f"{avg_polarity:+.2f}")
k4.metric("Active topics (window)", n_topics)
k5.metric("Critical issues (score ≥ 60)", len(critical))

if not history:
    st.info("Click **▶ Play** or **⏭ Step** in the sidebar to start streaming feedback.")
    st.stop()


tab_priority, tab_feed, tab_trends, tab_topics = st.tabs(
    ["🚨 Prioritized Issues & Requests", "📝 Live Feed", "📈 Trends", "☁️ Topic Explorer"]
)

# ---------------------------------------------------------------------------
# Tab: Prioritized issues & requests
# ---------------------------------------------------------------------------
with tab_priority:
    if not st.session_state.priorities:
        st.warning(f"Need at least {config.MIN_MESSAGES_TO_MODEL} messages before topics are modeled. Keep streaming.")
    else:
        issues, requests = split_issues_and_requests(st.session_state.priorities)

        st.subheader("Top Issues (by priority score)")
        if issues:
            df_issues = pd.DataFrame([{
                "Topic": i["label"],
                "Priority": i["priority_score"],
                "Count": i["count"],
                "% Negative": i["pct_negative"],
                "Avg Impact": i["avg_impact"],
                "Avg Polarity": i["avg_polarity"],
            } for i in issues])
            fig = px.bar(
                df_issues.head(10)[::-1], x="Priority", y="Topic", orientation="h",
                color="Priority", color_continuous_scale="Reds",
                text="Count", labels={"Priority": "Priority Score"},
            )
            fig.update_layout(height=400, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

            for i in issues[:6]:
                with st.expander(f"🔴 {i['label']}  —  score {i['priority_score']}  ·  {i['count']} mentions  ·  {i['pct_negative']}% negative"):
                    st.write(f"**Keywords:** {', '.join(i['keywords']) if i['keywords'] else '—'}")
                    st.write(f"**Avg business impact:** {i['avg_impact']}  |  **Avg polarity:** {i['avg_polarity']}")
                    st.write("**Example feedback:**")
                    for ex in i["examples"]:
                        st.markdown(f"> {ex['text']}  \n*({ex['source']}, polarity {ex['polarity']:+.2f})*")
        else:
            st.info("No issue clusters yet.")

        st.subheader("Top Feature Requests (by frequency)")
        if requests:
            df_req = pd.DataFrame([{
                "Topic": r["label"], "Mentions": r["count"], "% of cluster requesting": r["pct_feature_request"],
            } for r in requests])
            fig2 = px.bar(df_req.head(10)[::-1], x="Mentions", y="Topic", orientation="h", color_discrete_sequence=["#2E86DE"])
            fig2.update_layout(height=350)
            st.plotly_chart(fig2, use_container_width=True)
            for r in requests[:6]:
                with st.expander(f"💡 {r['label']} — {r['count']} mentions"):
                    for ex in r["examples"]:
                        st.markdown(f"> {ex['text']}  \n*({ex['source']})*")
        else:
            st.info("No feature-request clusters detected yet.")

# ---------------------------------------------------------------------------
# Tab: Live feed
# ---------------------------------------------------------------------------
with tab_feed:
    st.subheader("Most recent feedback")
    recent = history[-30:][::-1]
    feed_df = pd.DataFrame([{
        "Time": h["timestamp"], "Source": h["source"], "Text": h["text"],
        "Sentiment": h.get("sentiment_label", "—"),
        "Polarity": round(h.get("polarity", 0), 2),
        "Topic": st.session_state.topic_info.get(h.get("topic_id"), {}).get("label", "unmodeled"),
    } for h in recent])

    def color_sentiment(val):
        color = {"positive": "#d4f7d4", "negative": "#f7d4d4", "neutral": "#eeeeee"}.get(val, "")
        return f"background-color: {color}"

    style_fn = getattr(feed_df.style, "map", None) or feed_df.style.applymap
    st.dataframe(
        style_fn(color_sentiment, subset=["Sentiment"]),
        use_container_width=True, height=600,
    )

# ---------------------------------------------------------------------------
# Tab: Trends
# ---------------------------------------------------------------------------
with tab_trends:
    st.subheader("Sentiment distribution")
    sc = Counter(h.get("sentiment_label", "neutral") for h in history)
    fig_pie = px.pie(
        names=list(sc.keys()), values=list(sc.values()),
        color=list(sc.keys()),
        color_discrete_map={"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c"},
        hole=0.4,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Topic frequency over time")
    if st.session_state.topic_info:
        df_hist = pd.DataFrame(window)
        df_hist = df_hist[df_hist["topic_id"].apply(lambda t: t not in (-1, -99))]
        if not df_hist.empty:
            df_hist["label"] = df_hist["topic_id"].map(lambda t: st.session_state.topic_info.get(t, {}).get("label", str(t)))
            df_hist["ts"] = pd.to_datetime(df_hist["timestamp"])
            df_hist["bucket"] = df_hist["ts"].dt.floor("6h")
            trend = df_hist.groupby(["bucket", "label"]).size().reset_index(name="count")
            top_labels = df_hist["label"].value_counts().head(6).index
            trend = trend[trend["label"].isin(top_labels)]
            fig_line = px.line(trend, x="bucket", y="count", color="label", markers=True)
            fig_line.update_layout(height=420, legend_title="Topic")
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Topic trend appears once modeling has run.")
    else:
        st.info("Topic trend appears once modeling has run.")

    st.subheader("Sentiment over time")
    df_all = pd.DataFrame(history)
    df_all["ts"] = pd.to_datetime(df_all["timestamp"])
    df_all["bucket"] = df_all["ts"].dt.floor("6h")
    sentiment_trend = df_all.groupby("bucket")["polarity"].mean().reset_index()
    fig_sent = go.Figure()
    fig_sent.add_trace(go.Scatter(x=sentiment_trend["bucket"], y=sentiment_trend["polarity"],
                                   mode="lines+markers", line=dict(color="#8e44ad")))
    fig_sent.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_sent.update_layout(height=300, yaxis_title="Avg polarity (-1..1)")
    st.plotly_chart(fig_sent, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab: Topic explorer (word clouds)
# ---------------------------------------------------------------------------
with tab_topics:
    st.subheader("Topic keyword clouds")
    if not st.session_state.topic_info:
        st.info("Topics appear once enough messages have streamed in.")
    else:
        try:
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt
            have_wc = True
        except Exception:
            have_wc = False
            st.warning("wordcloud/matplotlib not installed — showing keyword lists instead.")

        sorted_topics = sorted(
            [(tid, info) for tid, info in st.session_state.topic_info.items() if tid not in (-1, -99)],
            key=lambda kv: kv[1]["size"], reverse=True,
        )
        cols = st.columns(3)
        for idx, (tid, info) in enumerate(sorted_topics[:9]):
            with cols[idx % 3]:
                st.markdown(f"**{info['label']}** ({info['size']} msgs)")
                if have_wc and info["keywords"]:
                    freqs = {kw: (len(info["keywords"]) - i) for i, kw in enumerate(info["keywords"])}
                    wc = WordCloud(width=300, height=180, background_color="white").generate_from_frequencies(freqs)
                    fig, ax = plt.subplots(figsize=(3, 1.8))
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.write(", ".join(info["keywords"]) or "—")
