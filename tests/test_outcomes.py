import tempfile
import unittest
from pathlib import Path

from src.outcomes import OutcomeTracker


class OutcomeTests(unittest.TestCase):
    def test_tracks_sentiment_improvement_and_accounts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outcomes.json"
            tracker = OutcomeTracker(path)
            tracker.mark_released("sync", -0.8, ["acme", "acme", "globex"], "Fixed in 2.0")
            result = OutcomeTracker(path).evaluate("sync", -0.2)
            self.assertTrue(result["improved"])
            self.assertEqual(result["sentiment_delta"], 0.6)
            self.assertEqual(result["account_ids"], ["acme", "globex"])


if __name__ == "__main__":
    unittest.main()
