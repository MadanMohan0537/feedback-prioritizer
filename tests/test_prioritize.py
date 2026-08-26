"""
Unit tests for prioritization + feature-request detection.

Run: python tests/test_prioritize.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prioritize import (
    business_impact_score,
    is_feature_request,
    compute_priorities,
    split_issues_and_requests,
    feature_request_score,
)
from src.pipeline import FeedbackPipeline
from src.sentiment import SentimentAnalyzer


class TestBusinessImpact(unittest.TestCase):
    def test_critical_keywords(self):
        self.assertEqual(business_impact_score("The app keeps crashing on launch"), 1.0)
        self.assertEqual(business_impact_score("I was billed twice this month"), 1.0)
        self.assertEqual(business_impact_score("Payment failed and no refund"), 1.0)

    def test_moderate_keywords(self):
        self.assertEqual(business_impact_score("Everything is so slow and laggy"), 0.6)
        self.assertEqual(business_impact_score("There's a bug in notifications"), 0.6)

    def test_default_when_no_match(self):
        score = business_impact_score("Nice product overall, thanks")
        self.assertLess(score, 0.3)


class TestFeatureRequests(unittest.TestCase):
    def test_detects_wish_phrases(self):
        self.assertTrue(is_feature_request("I wish there was dark mode"))
        self.assertTrue(is_feature_request("Please add Slack integration"))
        self.assertTrue(is_feature_request("Would be nice to export to PDF"))

    def test_rejects_plain_complaints(self):
        self.assertFalse(is_feature_request("The app keeps crashing"))
        self.assertFalse(is_feature_request("Support never replies"))


class TestPriorities(unittest.TestCase):
    def _items(self):
        return [
            {"topic_id": 0, "text": "App crashes constantly, unusable", "polarity": -0.9, "source": "app_store", "timestamp": "2026-01-01"},
            {"topic_id": 0, "text": "Crash on launch again", "polarity": -0.8, "source": "twitter", "timestamp": "2026-01-01"},
            {"topic_id": 0, "text": "Keeps crashing after update", "polarity": -0.85, "source": "google_play", "timestamp": "2026-01-01"},
            {"topic_id": 1, "text": "Font color is a bit off", "polarity": -0.1, "source": "in_app_survey", "timestamp": "2026-01-01"},
            {"topic_id": 1, "text": "Minor UI nitpick on spacing", "polarity": 0.0, "source": "in_app_survey", "timestamp": "2026-01-01"},
            {"topic_id": 2, "text": "I wish you would add dark mode", "polarity": 0.2, "source": "twitter", "timestamp": "2026-01-01"},
            {"topic_id": 2, "text": "Please add a dark mode option", "polarity": 0.3, "source": "app_store", "timestamp": "2026-01-01"},
            {"topic_id": 2, "text": "Would be nice to have dark mode", "polarity": 0.25, "source": "google_play", "timestamp": "2026-01-01"},
        ]

    def test_high_impact_negative_outranks_mild(self):
        topic_info = {
            0: {"label": "crashes", "keywords": ["crash"], "size": 3},
            1: {"label": "ui nit", "keywords": ["font"], "size": 2},
            2: {"label": "dark mode", "keywords": ["dark", "mode"], "size": 3},
        }
        results = compute_priorities(self._items(), topic_info)
        by_id = {r["topic_id"]: r for r in results}
        self.assertGreater(by_id[0]["priority_score"], by_id[1]["priority_score"])
        self.assertTrue(by_id[2]["is_feature_request"])

    def test_split_and_feature_score(self):
        topic_info = {
            0: {"label": "crashes", "keywords": ["crash"], "size": 3},
            1: {"label": "ui nit", "keywords": ["font"], "size": 2},
            2: {"label": "dark mode", "keywords": ["dark", "mode"], "size": 3},
        }
        results = compute_priorities(self._items(), topic_info)
        issues, requests = split_issues_and_requests(results)
        self.assertTrue(any(i["topic_id"] == 0 for i in issues))
        self.assertTrue(any(r["topic_id"] == 2 for r in requests))
        self.assertGreater(feature_request_score(requests[0]), 0)


class TestPipeline(unittest.TestCase):
    def test_ingest_and_retrain(self):
        # Force VADER so the test never needs network / torch.
        analyzer = SentimentAnalyzer(prefer_transformer=False)
        pipe = FeedbackPipeline(
            analyzer=analyzer,
            rolling_window=100,
            retrain_every=10,
            min_messages=10,
        )
        texts = [
            "The app keeps crashing every time I open it.",
            "I was billed twice for my subscription.",
            "Please add dark mode, I wish it existed.",
            "Sync is painfully slow between devices.",
            "Support never replies to my tickets.",
            "Would be nice to export tasks to PDF.",
            "Login is broken after password reset.",
            "Love the redesign, looks much cleaner.",
            "Payment failed but you still charged me.",
            "The search feature times out constantly.",
            "Hope you add offline mode soon.",
            "Crashes whenever I attach a file.",
        ]
        items = [
            {"text": t, "source": "app_store", "timestamp": f"2026-01-01 10:{i:02d}:00", "id": i}
            for i, t in enumerate(texts)
        ]
        pipe.ingest(items[:10], force_retrain=True)
        self.assertGreaterEqual(len(pipe.history), 10)
        self.assertTrue(pipe.topic_info)
        self.assertTrue(pipe.priorities)
        self.assertEqual(pipe.sentiment_backend, "vader")

        snap = pipe.snapshot()
        self.assertEqual(snap["n_messages"], 10)
        self.assertIsNotNone(snap["model_backend"])


if __name__ == "__main__":
    unittest.main()
