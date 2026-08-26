"""
Quick end-to-end smoke test that exercises the pipeline without Streamlit:
loads sample data, scores sentiment, fits a topic model, computes
priorities, and prints a summary.

Run: python3 tests/smoke_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingest import load_feedback_rows
from src.sentiment import get_analyzer
from src.topic_model import build_topic_model
from src.prioritize import compute_priorities, split_issues_and_requests


def main():
    rows = load_feedback_rows()
    print(f"Loaded {len(rows)} feedback rows")

    sample = rows[:200]
    analyzer = get_analyzer()

    enriched = []
    for i, row in enumerate(sample):
        res = analyzer.analyze(row["text"])
        enriched.append({
            "id": i, "text": row["text"], "source": row["source"],
            "timestamp": row["timestamp"], "polarity": res.polarity,
            "sentiment_label": res.label,
        })
    print(f"Sentiment backend: {analyzer._backend}")
    from collections import Counter
    print("Sentiment distribution:", Counter(e["sentiment_label"] for e in enriched))

    model = build_topic_model()
    docs = [e["text"] for e in enriched]
    topic_ids, topic_info = model.fit_transform(docs)
    for e, tid in zip(enriched, topic_ids):
        e["topic_id"] = int(tid)
    print(f"Topic backend: {model.name}")
    print(f"Discovered {len([t for t in topic_info if t not in (-1,)])} topics")
    for tid, info in sorted(topic_info.items(), key=lambda kv: -kv[1]["size"])[:5]:
        print(f"  topic {tid}: {info['label']}  (n={info['size']})")

    priorities = compute_priorities(enriched, topic_info)
    issues, requests = split_issues_and_requests(priorities)

    print("\nTop 5 prioritized issues:")
    for i in issues[:5]:
        print(f"  [{i['priority_score']:5.1f}] {i['label']}  (n={i['count']}, "
              f"{i['pct_negative']}% negative, impact={i['avg_impact']})")

    print("\nTop 5 feature requests:")
    for r in requests[:5]:
        print(f"  [{r['count']} mentions] {r['label']}")

    assert len(enriched) == 200
    assert priorities, "Expected at least one prioritized topic"
    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
