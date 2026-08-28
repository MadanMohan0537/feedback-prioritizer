import unittest

from src.evaluation import ranking_metrics, sentiment_metrics, topic_metrics


class EvaluationTests(unittest.TestCase):
    def test_topic_metrics_reward_matching_clusters(self):
        metrics = topic_metrics(["a", "a", "b"], [1, 1, 0])
        self.assertEqual(metrics["adjusted_rand_index"], 1.0)

    def test_sentiment_uses_rating_as_documented_proxy(self):
        metrics = sentiment_metrics([1, 3, 5], [-0.8, 0, 0.9])
        self.assertEqual(metrics["rating_proxy_accuracy"], 1.0)

    def test_ranking_contract(self):
        metrics = ranking_metrics([
            {"priority_score": 80, "score_contributions": {"impact": 20}},
            {"priority_score": 20, "score_contributions": {"impact": 5}},
        ])
        self.assertTrue(all(metrics.values()))


if __name__ == "__main__":
    unittest.main()
