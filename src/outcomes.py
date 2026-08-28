"""Close the feedback loop and measure sentiment after a topic is addressed."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path

from . import config


class OutcomeTracker:
    def __init__(self, path=None):
        default = Path(config.TOPIC_REGISTRY_PATH).with_name("outcomes.json")
        self.path = Path(path or default)
        self.data = {"version": 1, "outcomes": {}}
        self.load()

    def load(self):
        if not self.path.exists():
            return
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self.data.update(loaded)
        except (OSError, ValueError):
            return

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.path)

    def mark_released(self, stable_key, baseline_polarity, account_ids=(), note=""):
        self.data["outcomes"][stable_key] = {
            "released_at": datetime.now(timezone.utc).isoformat(),
            "baseline_polarity": float(baseline_polarity),
            "account_ids": sorted({str(value) for value in account_ids if value}),
            "note": str(note).strip(),
        }
        self.save()

    def evaluate(self, stable_key, current_polarity):
        record = self.data["outcomes"].get(stable_key)
        if not record:
            return None
        current = float(current_polarity)
        baseline = float(record["baseline_polarity"])
        return {
            **record,
            "current_polarity": round(current, 3),
            "sentiment_delta": round(current - baseline, 3),
            "improved": current > baseline,
        }
