"""
Orchestrates the streaming enrichment loop used by the dashboard and
offline runners: ingest → sentiment → (periodic) topic model → prioritize.

Keeping this logic out of app.py makes the Streamlit layer thin and lets
tests / notebooks exercise the same path without a UI.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from . import config
from .prioritize import business_impact_score, compute_priorities
from .sentiment import SentimentAnalyzer, get_analyzer
from .topic_model import BaseTopicModel, build_topic_model


class FeedbackPipeline:
    """
    Stateful rolling-window pipeline.

    Call `ingest(items)` with newly arrived feedback dicts (timestamp, text,
    source). Sentiment + impact are applied immediately; topic modeling
    re-runs every `retrain_every` messages once the window has enough data.
    """

    def __init__(
        self,
        analyzer: Optional[SentimentAnalyzer] = None,
        rolling_window: int = None,
        retrain_every: int = None,
        min_messages: int = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.analyzer = analyzer or get_analyzer()
        self.rolling_window = rolling_window or config.ROLLING_WINDOW_SIZE
        self.retrain_every = retrain_every or config.RETRAIN_EVERY_N
        self.min_messages = min_messages or config.MIN_MESSAGES_TO_MODEL
        self.weights = weights or dict(config.PRIORITY_WEIGHTS)

        self.history: List[Dict[str, Any]] = []
        self.topic_info: Dict[int, Dict[str, Any]] = {}
        self.priorities: List[Dict[str, Any]] = []
        self.since_retrain = 0
        self.model_backend: Optional[str] = None
        self._topic_model: Optional[BaseTopicModel] = None

    @property
    def sentiment_backend(self) -> Optional[str]:
        return getattr(self.analyzer, "_backend", None)

    @property
    def window(self) -> List[Dict[str, Any]]:
        return self.history[-self.rolling_window :]

    def reset(self) -> None:
        self.history.clear()
        self.topic_info.clear()
        self.priorities.clear()
        self.since_retrain = 0
        self.model_backend = None
        self._topic_model = None

    def ingest(self, items: List[Dict[str, Any]], force_retrain: bool = False) -> int:
        """Enrich and append items. Returns how many were ingested."""
        for item in items:
            result = self.analyzer.analyze(item["text"])
            item["polarity"] = result.polarity
            item["sentiment_label"] = result.label
            item["sentiment_score"] = result.score
            item["impact"] = business_impact_score(item["text"])
            item["topic_id"] = item.get("topic_id", -99)
            self.history.append(item)

        self.since_retrain += len(items)
        if force_retrain or self.since_retrain >= self.retrain_every:
            self.retrain_topics()
        elif self.topic_info:
            # Cheap re-score if topics already exist (e.g. weight tweaks).
            self.priorities = compute_priorities(self.window, self.topic_info, self.weights)
        return len(items)

    def retrain_topics(self) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]]]:
        window = self.window
        if len(window) < self.min_messages:
            return self.topic_info, self.priorities

        docs = [it["text"] for it in window]
        model = build_topic_model()
        self._topic_model = model
        self.model_backend = model.name
        topic_ids, topic_info = model.fit_transform(docs)

        for it, tid in zip(window, topic_ids):
            it["topic_id"] = int(tid)

        self.topic_info = topic_info
        self.priorities = compute_priorities(window, topic_info, self.weights)
        self.since_retrain = 0
        return self.topic_info, self.priorities

    def set_weights(self, weights: Dict[str, float]) -> None:
        self.weights = dict(weights)
        if self.topic_info and self.window:
            self.priorities = compute_priorities(self.window, self.topic_info, self.weights)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "n_messages": len(self.history),
            "n_window": len(self.window),
            "model_backend": self.model_backend,
            "sentiment_backend": self.sentiment_backend,
            "n_topics": len([t for t in self.topic_info if t not in (-1, -99)]),
            "priorities": list(self.priorities),
            "topic_info": dict(self.topic_info),
        }
