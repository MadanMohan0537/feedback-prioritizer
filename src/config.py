"""
Central configuration for the Feedback Analyzer pipeline.
Tweak these to change stream speed, model choices, window sizes, and
prioritization weights without touching pipeline code.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SAMPLE_CSV = os.path.join(DATA_DIR, "sample_reviews.csv")

# ---------------------------------------------------------------------------
# Streaming / ingestion
# ---------------------------------------------------------------------------
DEFAULT_DELAY_SECONDS = 0.0   # delay between yielded messages in the raw generator
                              # (the Streamlit app controls actual pacing itself)

# ---------------------------------------------------------------------------
# Rolling window + retraining cadence
# ---------------------------------------------------------------------------
ROLLING_WINDOW_SIZE = 250     # how many recent feedback items topic modeling considers
RETRAIN_EVERY_N = 20          # re-run topic modeling every N new messages
MIN_MESSAGES_TO_MODEL = 15    # need at least this many messages before modeling

# ---------------------------------------------------------------------------
# Sentiment analysis
# ---------------------------------------------------------------------------
SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# If the transformer model can't be loaded (no network / no torch), we fall
# back automatically to VADER (rule-based, always available).

# ---------------------------------------------------------------------------
# Topic modeling
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
MIN_TOPIC_SIZE = 5            # BERTopic / HDBSCAN minimum cluster size
N_FALLBACK_TOPICS = 8         # number of clusters for the TF-IDF/KMeans fallback

# ---------------------------------------------------------------------------
# Prioritization
# ---------------------------------------------------------------------------
# Priority Score = frequency_weight * freq_norm
#                + sentiment_weight * negativity_norm
#                + impact_weight    * business_impact
#
# All components are normalized to [0, 1] before combining so the weights
# are directly interpretable as "how much this factor matters."
PRIORITY_WEIGHTS = {
    "frequency": 0.40,
    "sentiment": 0.35,
    "impact": 0.25,
}

# Keywords that signal high business impact when present in feedback text.
# Score 1.0 = severe (outages, billing, data loss), 0.6 = moderate, 0.3 = minor.
BUSINESS_IMPACT_KEYWORDS = {
    1.0: ["crash", "crashed", "crashing", "data loss", "lost my data", "billed twice",
          "overcharged", "can't login", "cannot log in", "security", "hacked",
          "payment failed", "double charged", "refund", "unusable", "down", "outage"],
    0.6: ["slow", "lag", "laggy", "freezes", "freezing", "bug", "broken", "error",
          "billing", "subscription", "confusing", "difficult", "support", "waiting",
          "sync", "notification", "battery"],
    0.3: ["ui", "design", "color", "font", "minor", "small", "suggestion", "would be nice",
          "wish", "nitpick"],
}

# Phrases that flag a message as a feature request rather than a complaint.
FEATURE_REQUEST_PATTERNS = [
    "i wish", "please add", "would be nice", "can you add", "could you add",
    "feature request", "add support for", "it would be great if", "please support",
    "hope you add", "would love to see", "any plans to add", "please implement",
    "missing feature", "needs a", "should have",
]

# Default business-impact weight applied when no keyword matches (neutral feedback).
DEFAULT_IMPACT_SCORE = 0.15

SOURCES = ["app_store", "google_play", "twitter", "support_ticket", "in_app_survey"]
