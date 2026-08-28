"""Evaluate the fallback pipeline on the labeled synthetic fixture."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import ranking_metrics, sentiment_metrics, topic_metrics
from src.ingest import load_feedback_rows
from src.opinion_units import expand_feedback_item
from src.prioritize import compute_priorities
from src.sentiment import SentimentAnalyzer
from src.topic_model import SimpleTopicModel


def main():
    source_rows = load_feedback_rows()
    analyzer = SentimentAnalyzer(prefer_transformer=False)
    rows = []
    for index, source in enumerate(source_rows):
        source = {**source, "id": index}
        for unit in expand_feedback_item(source):
            result = analyzer.analyze(unit["text"])
            rows.append({**unit, "polarity": result.polarity})

    model = SimpleTopicModel()
    predicted, info = model.fit_transform([row["text"] for row in rows])
    for row, topic_id in zip(rows, predicted):
        row["topic_id"] = int(topic_id)
    priorities = compute_priorities(rows, info)

    report = {
        "topic_model": topic_metrics([row.get("true_topic", "unknown") for row in rows], predicted),
        "sentiment": sentiment_metrics([row.get("rating") for row in rows], [row["polarity"] for row in rows]),
        "ranking": ranking_metrics(priorities),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
