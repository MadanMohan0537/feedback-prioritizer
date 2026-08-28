import unittest

from src.opinion_units import expand_feedback_item, split_opinion_units


class OpinionUnitTests(unittest.TestCase):
    def test_splits_mixed_feedback_at_contrast(self):
        units = split_opinion_units("I love the design but billing keeps failing.")
        self.assertEqual(units, ["I love the design", "billing keeps failing."])

    def test_preserves_parent_provenance(self):
        rows = expand_feedback_item({"id": 42, "text": "Fast. However, sync is broken.", "source": "test"})
        self.assertTrue(all(row["parent_id"] == 42 for row in rows))
        self.assertTrue(all(row["original_text"] for row in rows))


if __name__ == "__main__":
    unittest.main()
