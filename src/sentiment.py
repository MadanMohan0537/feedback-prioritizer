"""
Sentiment analysis with a graceful-degradation strategy:

1. Preferred: cardiffnlp/twitter-roberta-base-sentiment-latest via
   transformers -- a RoBERTa model fine-tuned on ~124M tweets, robust to
   informal text, negation, emoji-adjacent phrasing, and short reviews.
   Returns calibrated positive/neutral/negative probabilities.

2. Fallback: VADER (vaderSentiment) -- a lexicon + rule-based analyzer
   that needs no model download or GPU. It handles negation ("not good"),
   intensifiers ("very slow"), and punctuation/caps emphasis reasonably
   well, which is why it's a sane fallback for a real-time demo rather
   than something purely random.

Edge cases considered (documented here and in the README):
- Negation: both backends handle simple negation ("not happy") correctly;
  neither reliably handles double negation or sarcasm ("oh great, ANOTHER
  crash") -- flagged as a known limitation.
- Mixed sentiment in one message ("love the design but it keeps crashing")
  is resolved to a single dominant label + a continuous polarity score
  (-1..1) so downstream prioritization isn't forced into a false binary.
- Very short text ("meh", "trash") is handled fine by both backends since
  neither depends on sentence structure.
"""

from functools import lru_cache

from . import config


class SentimentResult:
    __slots__ = ("label", "score", "polarity", "backend")

    def __init__(self, label, score, polarity, backend):
        self.label = label            # "positive" | "neutral" | "negative"
        self.score = score            # confidence of the winning label, 0..1
        self.polarity = polarity      # continuous -1 (very negative) .. +1 (very positive)
        self.backend = backend        # which model actually produced this

    def to_dict(self):
        return {
            "label": self.label,
            "score": self.score,
            "polarity": self.polarity,
            "backend": self.backend,
        }


class SentimentAnalyzer:
    """
    Lazily loads the transformer pipeline on first use; falls back to VADER
    if transformers/torch aren't available or the model can't be fetched
    (e.g. no network). The choice is made once and cached for the process.
    """

    def __init__(self, prefer_transformer=True):
        self.prefer_transformer = prefer_transformer
        self._backend = None       # "transformer" | "vader"
        self._pipeline = None
        self._vader = None

    def _init_backend(self):
        if self._backend is not None:
            return
        if self.prefer_transformer:
            try:
                from transformers import pipeline
                self._pipeline = pipeline(
                    "sentiment-analysis",
                    model=config.SENTIMENT_MODEL_NAME,
                    tokenizer=config.SENTIMENT_MODEL_NAME,
                    top_k=None,
                )
                self._backend = "transformer"
                return
            except Exception:
                pass
        # Fallback
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self._vader = SentimentIntensityAnalyzer()
        self._backend = "vader"

    def analyze(self, text: str) -> SentimentResult:
        self._init_backend()
        text = (text or "").strip()
        if not text:
            return SentimentResult("neutral", 1.0, 0.0, self._backend)

        if self._backend == "transformer":
            return self._analyze_transformer(text)
        return self._analyze_vader(text)

    def _analyze_transformer(self, text: str) -> SentimentResult:
        try:
            # top_k=None returns all class probabilities
            results = self._pipeline(text[:512])[0]
            best = max(results, key=lambda r: r["score"])
            label = best["label"].lower()
            score_map = {r["label"].lower(): r["score"] for r in results}
            pos = score_map.get("positive", 0.0)
            neg = score_map.get("negative", 0.0)
            polarity = pos - neg  # -1..1
            return SentimentResult(label, best["score"], polarity, "transformer")
        except Exception:
            # runtime failure (e.g. OOM) -- degrade to VADER for this call
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            if self._vader is None:
                self._vader = SentimentIntensityAnalyzer()
            return self._analyze_vader(text)

    def _analyze_vader(self, text: str) -> SentimentResult:
        scores = self._vader.polarity_scores(text)
        compound = scores["compound"]  # -1..1
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        return SentimentResult(label, abs(compound), compound, "vader")

    def analyze_batch(self, texts):
        return [self.analyze(t) for t in texts]


@lru_cache(maxsize=1)
def get_analyzer() -> SentimentAnalyzer:
    """Process-wide singleton so the model is loaded once."""
    return SentimentAnalyzer()
