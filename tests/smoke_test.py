"""
Quick end-to-end smoke test that exercises the pipeline without Streamlit:
loads sample data, scores sentiment, fits a topic model, computes
priorities, and prints a summary.

Run: python3 tests/smoke_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter

from src.ingest import load_feedback_rows
from src.pipeline import FeedbackPipeline
from src.prioritize import split_issues_and_requests
from src.sentiment import SentimentAnalyzer


def main():
    rows = load_feedback_rows()
    print(f"Loaded {len(rows)} feedback rows")

    sample = rows[:200]
    # Prefer VADER in CI / offline environments so the smoke test never
    # blocks on a Hugging Face download.
    analyzer = SentimentAnalyzer(prefer_transformer=False)
    pipe = FeedbackPipeline(analyzer=analyzer, min_messages=15, retrain_every=50)

    items = [{
        "id": i, "text": row["text"], "source": row["source"],
        "timestamp": row["timestamp"], "rating": row.get("rating"),
    } for i, row in enumerate(sample)]
    pipe.ingest(items, force_retrain=True)

    print(f"Sentiment backend: {pipe.sentiment_backend}")
    print("Sentiment distribution:", Counter(e["sentiment_label"] for e in pipe.history))
    print(f"Topic backend: {pipe.model_backend}")
    print(f"Discovered {len([t for t in pipe.topic_info if t not in (-1,)])} topics")
    for tid, info in sorted(pipe.topic_info.items(), key=lambda kv: -kv[1]["size"])[:5]:
        print(f"  topic {tid}: {info['label']}  (n={info['size']})")

    issues, requests = split_issues_and_requests(pipe.priorities)

    print("\nTop 5 prioritized issues:")
    for i in issues[:5]:
        print(f"  [{i['priority_score']:5.1f}] {i['label']}  (n={i['count']}, "
              f"{i['pct_negative']}% negative, impact={i['avg_impact']})")

    print("\nTop 5 feature requests:")
    for r in requests[:5]:
        print(f"  [{r['count']} mentions] {r['label']}")

    assert len(pipe.history) == 200
    assert pipe.priorities, "Expected at least one prioritized topic"
    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
