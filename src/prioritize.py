"""
Prioritization engine: turns (topic, sentiment, business-impact keywords)
into a single ranked list a PM can act on.

Priority Score (0..100) combines three normalized factors:

  frequency_norm   = topic count in the rolling window, min-max normalized
                      across topics currently in view
  negativity_norm  = (1 - avg_polarity) / 2   -- rescales polarity(-1..1)
                      so more-negative topics score higher (this is a
                      *complaint* prioritizer: negative + frequent + high
                      impact bubbles to the top)
  impact_norm      = average business-impact weight of messages in the
                      topic, from keyword matches (0.3 / 0.6 / 1.0) or the
                      default neutral weight

  score = 100 * (w_freq * frequency_norm + w_sent * negativity_norm + w_impact * impact_norm)

Weights are configurable in config.PRIORITY_WEIGHTS so a PM can re-tune
"what matters" (e.g. weight impact higher during a launch week) without
touching code.

Feature requests are tracked separately from issues: a topic is tagged as
a feature-request cluster if a high enough fraction of its messages match
FEATURE_REQUEST_PATTERNS. These are ranked by frequency + (positive lean is
fine -- a feature request isn't a complaint) rather than negativity.
"""

from collections import defaultdict

from . import config


def business_impact_score(text: str) -> float:
    text_l = text.lower()
    for weight in (1.0, 0.6, 0.3):
        for kw in config.BUSINESS_IMPACT_KEYWORDS[weight]:
            if kw in text_l:
                return weight
    return config.DEFAULT_IMPACT_SCORE


def is_feature_request(text: str) -> bool:
    text_l = text.lower()
    return any(p in text_l for p in config.FEATURE_REQUEST_PATTERNS)


def _minmax_norm(values):
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi == lo:
        return {k: 0.5 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def compute_priorities(enriched_items, topic_info, weights=None):
    """
    enriched_items: list of dicts, each with at least:
        topic_id, text, polarity (float -1..1), timestamp
    topic_info: {topic_id: {"label", "keywords", "size"}}

    Returns a list of topic summary dicts sorted by priority_score desc,
    each with: topic_id, label, keywords, count, avg_polarity,
    avg_impact, pct_negative, pct_feature_request, priority_score,
    examples (up to 3 representative messages), is_feature_request (bool,
    majority vote).
    """
    weights = weights or config.PRIORITY_WEIGHTS
    by_topic = defaultdict(list)
    for item in enriched_items:
        by_topic[item["topic_id"]].append(item)

    freq_raw, impact_raw, polarity_raw = {}, {}, {}
    fr_fraction = {}

    for tid, items in by_topic.items():
        if tid in (-1, -99):
            continue  # -1: unclustered noise from BERTopic; -99: not yet classified (pending retrain)
        freq_raw[tid] = len(items)
        impacts = [business_impact_score(it["text"]) for it in items]
        impact_raw[tid] = sum(impacts) / len(impacts)
        polarity_raw[tid] = sum(it["polarity"] for it in items) / len(items)
        fr_fraction[tid] = sum(1 for it in items if is_feature_request(it["text"])) / len(items)

    freq_norm = _minmax_norm(freq_raw)
    impact_norm = _minmax_norm(impact_raw)

    results = []
    for tid, items in by_topic.items():
        if tid in (-1, -99):
            continue
        negativity_norm = (1 - polarity_raw[tid]) / 2  # polarity -1..1 -> negativity 1..0
        score = 100 * (
            weights["frequency"] * freq_norm.get(tid, 0)
            + weights["sentiment"] * negativity_norm
            + weights["impact"] * impact_norm.get(tid, 0)
        )
        pct_negative = sum(1 for it in items if it["polarity"] < -0.05) / len(items) * 100
        info = topic_info.get(tid, {"label": f"Topic {tid}", "keywords": [], "size": len(items)})

        examples = sorted(items, key=lambda it: it["polarity"])[:3]  # most negative first

        results.append({
            "topic_id": tid,
            "label": info["label"],
            "keywords": info["keywords"],
            "count": len(items),
            "avg_polarity": round(polarity_raw[tid], 3),
            "avg_impact": round(impact_raw[tid], 3),
            "pct_negative": round(pct_negative, 1),
            "pct_feature_request": round(fr_fraction[tid] * 100, 1),
            "is_feature_request": fr_fraction[tid] >= 0.4,
            "priority_score": round(score, 1),
            "examples": [{"text": it["text"], "polarity": it["polarity"],
                          "source": it.get("source"), "timestamp": it.get("timestamp")}
                         for it in examples],
        })

    results.sort(key=lambda r: r["priority_score"], reverse=True)
    return results


def feature_request_score(topic_summary: dict) -> float:
    """
    Rank feature-request clusters by demand (frequency) with a small boost
    for how cleanly the cluster is "request-shaped" (pct_feature_request).
    Positive sentiment is fine here — people asking for features are often
    otherwise happy customers.
    """
    demand = topic_summary.get("count", 0)
    purity = topic_summary.get("pct_feature_request", 0) / 100.0
    return demand * (0.7 + 0.3 * purity)


def split_issues_and_requests(priority_list):
    """Convenience split for the dashboard's two ranked lists."""
    issues = [r for r in priority_list if not r["is_feature_request"]]
    requests = [r for r in priority_list if r["is_feature_request"]]
    requests.sort(key=feature_request_score, reverse=True)
    return issues, requests
