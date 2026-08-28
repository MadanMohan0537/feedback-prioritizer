import unittest

from src.prioritize import business_impact_score, compute_priorities


def item(topic_id, text, polarity):
    return {
        "topic_id": topic_id,
        "text": text,
        "polarity": polarity,
        "timestamp": "2026-01-01T00:00:00Z",
        "source": "test",
    }


class PriorityScoringTests(unittest.TestCase):
    def setUp(self):
        self.items = [
            item(0, "The app crashes and is unusable", -0.9),
            item(0, "Another crash caused data loss", -0.8),
            item(1, "The new design is nice", 0.6),
        ]
        self.topics = {
            0: {"label": "Reliability", "keywords": ["crash"], "size": 2},
            1: {"label": "Design", "keywords": ["design"], "size": 1},
        }

    def test_score_stays_bounded_when_ui_weights_exceed_one(self):
        results = compute_priorities(
            self.items,
            self.topics,
            {"frequency": 1, "sentiment": 1, "impact": 1},
        )
        self.assertTrue(all(0 <= row["priority_score"] <= 100 for row in results))
        self.assertAlmostEqual(sum(results[0]["normalized_weights"].values()), 1, places=2)

    def test_score_exposes_explainable_contributions(self):
        result = compute_priorities(self.items, self.topics)[0]
        self.assertEqual(
            result["priority_score"],
            round(sum(result["score_contributions"].values()), 1),
        )
        self.assertEqual(
            set(result["score_components"]),
            {"frequency", "sentiment", "impact", "trend", "customer_value"},
        )

    def test_zero_weights_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            compute_priorities(
                self.items,
                self.topics,
                {"frequency": 0, "sentiment": 0, "impact": 0, "trend": 0, "customer_value": 0},
            )

    def test_impact_keywords_match_whole_terms(self):
        self.assertEqual(business_impact_score("The workflow is smooth"), 0.15)
        self.assertEqual(business_impact_score("The app is slow"), 0.6)
        self.assertEqual(business_impact_score("We had an outage"), 1.0)

    def test_customer_value_surfaces_enterprise_risk(self):
        items = [
            {**item(0, "Minor issue", -0.2), "account_id": "free", "customer_tier": "free"},
            {**item(1, "Minor issue", -0.2), "account_id": "acme", "customer_tier": "enterprise", "arr": 200000},
        ]
        topics = {
            0: {"label": "Free", "keywords": ["minor"], "size": 1},
            1: {"label": "Enterprise", "keywords": ["minor"], "size": 1},
        }
        results = compute_priorities(
            items,
            topics,
            {"frequency": 0, "sentiment": 0, "impact": 0, "trend": 0, "customer_value": 1},
        )
        self.assertEqual(results[0]["label"], "Enterprise")
        self.assertEqual(results[0]["affected_arr"], 200000)

    def test_affected_arr_deduplicates_account_mentions(self):
        items = [
            {**item(0, "Crash", -0.8), "account_id": "acme", "arr": 50000},
            {**item(0, "Still crashing", -0.9), "account_id": "acme", "arr": 50000},
        ]
        topics = {0: {"label": "Crash", "keywords": ["crash"], "size": 2}}
        result = compute_priorities(items, topics)[0]
        self.assertEqual(result["affected_accounts"], 1)
        self.assertEqual(result["affected_arr"], 50000)


if __name__ == "__main__":
    unittest.main()
