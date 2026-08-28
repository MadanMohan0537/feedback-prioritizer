"""
Prioritization engine: turns (topic, sentiment, business-impact keywords)
into a single ranked list a PM can act on.

Priority Score (0..100) combines five normalized factors:

  frequency_norm   = topic count in the rolling window, min-max normalized
                      across topics currently in view
  negativity_norm  = (1 - avg_polarity) / 2   -- rescales polarity(-1..1)
                      so more-negative topics score higher (this is a
                      *complaint* prioritizer: negative + frequent + high
                      impact bubbles to the top)
  impact_norm      = average business-impact weight of messages in the
                      topic, from keyword matches (0.3 / 0.6 / 1.0) or the
                      default neutral weight
  trend_norm       = whether the topic is accelerating in the recent half
                      of the current analysis window
  customer_value   = ARR, customer tier, and explicit churn risk for the
                      accounts represented in the topic

The configured weights are normalized to sum to one before scoring.

Weights are configurable in config.PRIORITY_WEIGHTS so a PM can re-tune
"what matters" (e.g. weight impact higher during a launch week) without
touching code.

Feature requests are tracked separately from issues: a topic is tagged as
a feature-request cluster if a high enough fraction of its messages match
FEATURE_REQUEST_PATTERNS. These are ranked by frequency + (positive lean is
fine -- a feature request isn't a complaint) rather than negativity.
"""

from collections import defaultdict
from datetime import datetime, timezone
import re

from . import config


def business_impact_score(text: str) -> float:
    text_l = text.lower()
    for weight in (1.0, 0.6, 0.3):
        for kw in config.BUSINESS_IMPACT_KEYWORDS[weight]:
            if re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", text_l):
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


def _normalize_weights(weights):
    """Return non-negative weights that sum to one."""
    required = ("frequency", "sentiment", "impact", "trend", "customer_value")
    normalized = {}
    for key in required:
        try:
            value = float(weights.get(key, 0))
        except (TypeError, ValueError):
            raise ValueError(f"Priority weight '{key}' must be numeric.") from None
        if value < 0:
            raise ValueError(f"Priority weight '{key}' cannot be negative.")
        normalized[key] = value
    total = sum(normalized.values())
    if total <= 0:
        raise ValueError("At least one priority weight must be greater than zero.")
    return {key: value / total for key, value in normalized.items()}


def _parse_timestamp(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except (TypeError, ValueError):
        return None


def _topic_trend(items, midpoint):
    """Compare recent topic mentions with the preceding half-window."""
    timestamps = [_parse_timestamp(item.get("timestamp")) for item in items]
    timestamps = [value for value in timestamps if value is not None]
    if not timestamps or midpoint is None:
        return 0.5
    recent = sum(value >= midpoint for value in timestamps)
    previous = len(timestamps) - recent
    # Smoothed ratio: 0.5 is flat, values near 1 are emerging.
    return (recent + 1) / (recent + previous + 2)


def _arr_value(item):
    try:
        return max(0.0, float(item.get("arr") or 0))
    except (TypeError, ValueError):
        return 0.0


def _customer_value(item):
    arr = _arr_value(item)
    arr_signal = min(1.0, arr / 100_000)
    tier = str(item.get("customer_tier") or "unknown").strip().lower()
    tier_signal = config.CUSTOMER_TIER_VALUES.get(tier, config.CUSTOMER_TIER_VALUES["unknown"])
    churn_signal = 1.0 if str(item.get("churn_risk", "")).lower() in {"1", "true", "high", "yes"} else 0.0
    return max(arr_signal, tier_signal, churn_signal)


def _affected_arr(items):
    """Count each known account once, using its largest observed ARR value."""
    by_account = {}
    for item in items:
        account_id = str(item.get("account_id") or "").strip()
        if account_id:
            by_account[account_id] = max(by_account.get(account_id, 0.0), _arr_value(item))
    return sum(by_account.values())


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
    weights = _normalize_weights(weights or config.PRIORITY_WEIGHTS)
    by_topic = defaultdict(list)
    for item in enriched_items:
        by_topic[item["topic_id"]].append(item)

    freq_raw, impact_raw, polarity_raw, trend_raw, customer_value_raw = {}, {}, {}, {}, {}
    fr_fraction = {}

    parsed_times = sorted(
        value for value in (_parse_timestamp(item.get("timestamp")) for item in enriched_items)
        if value is not None
    )
    midpoint = parsed_times[len(parsed_times) // 2] if parsed_times else None

    for tid, items in by_topic.items():
        if tid in (-1, -99):
            continue  # -1: unclustered noise from BERTopic; -99: not yet classified (pending retrain)
        freq_raw[tid] = len(items)
        impacts = [business_impact_score(it["text"]) for it in items]
        impact_raw[tid] = sum(impacts) / len(impacts)
        polarity_raw[tid] = sum(it["polarity"] for it in items) / len(items)
        trend_raw[tid] = _topic_trend(items, midpoint)
        customer_value_raw[tid] = sum(_customer_value(it) for it in items) / len(items)
        fr_fraction[tid] = sum(1 for it in items if is_feature_request(it["text"])) / len(items)

    freq_norm = _minmax_norm(freq_raw)
    impact_norm = _minmax_norm(impact_raw)
    trend_norm = _minmax_norm(trend_raw)
    customer_value_norm = _minmax_norm(customer_value_raw)

    results = []
    for tid, items in by_topic.items():
        if tid in (-1, -99):
            continue
        negativity_norm = (1 - polarity_raw[tid]) / 2  # polarity -1..1 -> negativity 1..0
        components = {
            "frequency": freq_norm.get(tid, 0),
            "sentiment": negativity_norm,
            "impact": impact_norm.get(tid, 0),
            "trend": trend_norm.get(tid, 0),
            "customer_value": customer_value_norm.get(tid, 0),
        }
        contributions = {
            key: 100 * weights[key] * components[key]
            for key in components
        }
        score = sum(contributions.values())
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
            "trend_score": round(trend_raw[tid], 3),
            "avg_customer_value": round(customer_value_raw[tid], 3),
            "affected_accounts": len({it.get("account_id") for it in items if it.get("account_id")}),
            "affected_arr": round(_affected_arr(items), 2),
            "pct_negative": round(pct_negative, 1),
            "pct_feature_request": round(fr_fraction[tid] * 100, 1),
            "is_feature_request": fr_fraction[tid] >= 0.4,
            "priority_score": round(score, 1),
            "score_components": {key: round(value, 3) for key, value in components.items()},
            "score_contributions": {key: round(value, 1) for key, value in contributions.items()},
            "normalized_weights": {key: round(value, 3) for key, value in weights.items()},
            "examples": [{"text": it["text"], "polarity": it["polarity"],
                          "source": it.get("source"), "timestamp": it.get("timestamp")}
                         for it in examples],
        })

    results.sort(key=lambda r: r["priority_score"], reverse=True)
    return results


def split_issues_and_requests(priority_list):
    """Convenience split for the dashboard's two ranked lists."""
    issues = [r for r in priority_list if not r["is_feature_request"]]
    requests = [r for r in priority_list if r["is_feature_request"]]
    requests.sort(key=lambda r: r["count"], reverse=True)
    return issues, requests
