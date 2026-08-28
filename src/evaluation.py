"""Repeatable quality metrics for clustering, sentiment, and prioritization."""


def topic_metrics(true_labels, predicted_labels):
    if len(true_labels) != len(predicted_labels) or not true_labels:
        raise ValueError("Topic labels must be non-empty and have equal length.")
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    return {
        "adjusted_rand_index": round(adjusted_rand_score(true_labels, predicted_labels), 4),
        "normalized_mutual_info": round(normalized_mutual_info_score(true_labels, predicted_labels), 4),
        "sample_count": len(true_labels),
    }


def sentiment_metrics(ratings, polarities):
    pairs = []
    for rating, polarity in zip(ratings, polarities):
        try:
            rating = float(rating)
        except (TypeError, ValueError):
            continue
        expected = "negative" if rating <= 2 else "positive" if rating >= 4 else "neutral"
        predicted = "negative" if polarity < -0.05 else "positive" if polarity > 0.05 else "neutral"
        pairs.append((expected, predicted))
    if not pairs:
        return {"rating_proxy_accuracy": None, "rated_sample_count": 0}
    correct = sum(expected == predicted for expected, predicted in pairs)
    return {
        "rating_proxy_accuracy": round(correct / len(pairs), 4),
        "rated_sample_count": len(pairs),
    }


def ranking_metrics(priorities):
    scores = [row["priority_score"] for row in priorities]
    return {
        "topic_count": len(priorities),
        "scores_bounded": all(0 <= score <= 100 for score in scores),
        "scores_descending": scores == sorted(scores, reverse=True),
        "explanations_present": all(bool(row.get("score_contributions")) for row in priorities),
    }
