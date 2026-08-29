"""Deterministic synthetic feedback for demos and CI."""

from datetime import datetime, timedelta, timezone
import random

from .base import FetchResult
from ..models import FeedbackEntry


TEMPLATES = [
    ("The mobile app crashes whenever I try to sign in.", 1, "authentication"),
    ("Please add dark mode to the dashboard.", 4, "interface"),
    ("Exporting a large report is extremely slow.", 2, "exports"),
    ("I was charged twice for my subscription.", 1, "billing"),
    ("The new collaboration experience is excellent.", 5, "collaboration"),
    ("Notifications arrive hours after the event.", 2, "notifications"),
]


class SyntheticConnector:
    name = "synthetic"

    def __init__(self, count=100, seed=42):
        self.count, self.seed = count, seed

    def fetch(self, checkpoint=""):
        rng = random.Random(self.seed)
        start = datetime.now(timezone.utc) - timedelta(days=14)
        records = []
        for index in range(self.count):
            text, rating, area = rng.choice(TEMPLATES)
            raw = {"id": f"synthetic-{self.seed}-{index}", "text": text, "rating": rating, "product_area": area}
            entry = FeedbackEntry(
                text=text,
                source=self.name,
                source_type="simulated_review",
                timestamp=(start + timedelta(minutes=index * 17)).isoformat(),
                external_id=raw["id"],
                rating=rating,
                metadata={"product_area": area, "synthetic": True},
            )
            records.append((entry, raw))
        return FetchResult(records, str(self.count))
