import unittest

from src.sentiment import SentimentAnalyzer


class SentimentFallbackTests(unittest.TestCase):
    def test_builtin_lexicon_handles_basic_polarity(self):
        analyzer = SentimentAnalyzer(prefer_transformer=False)
        analyzer._backend = "builtin_lexicon"
        self.assertEqual(analyzer.analyze("The app is great and fast").label, "positive")
        self.assertEqual(analyzer.analyze("The app is broken and unusable").label, "negative")

    def test_builtin_lexicon_handles_simple_negation(self):
        analyzer = SentimentAnalyzer(prefer_transformer=False)
        analyzer._backend = "builtin_lexicon"
        self.assertEqual(analyzer.analyze("The app is not slow").label, "positive")


if __name__ == "__main__":
    unittest.main()
