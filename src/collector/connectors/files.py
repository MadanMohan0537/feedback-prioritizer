"""CSV and JSONL imports with configurable field mappings."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from .base import FetchResult
from ..models import FeedbackEntry


class FileConnector:
    name = "file"

    def __init__(self, path, source="import", mapping=None):
        self.path = Path(path)
        self.source = source
        self.mapping = {"text": "text", "timestamp": "timestamp", "user_id": "user_id", "rating": "rating", **(mapping or {})}

    def _rows(self):
        if self.path.suffix.lower() == ".jsonl":
            with self.path.open(encoding="utf-8") as handle:
                return [json.loads(line) for line in handle if line.strip()]
        with self.path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))

    def fetch(self, checkpoint=""):
        records = []
        for index, row in enumerate(self._rows()):
            text = row.get(self.mapping["text"], "")
            timestamp = row.get(self.mapping["timestamp"], "") or datetime.now(timezone.utc).isoformat()
            external_id = str(row.get("external_id") or row.get("id") or f"{self.path.name}:{index}")
            entry = FeedbackEntry(
                text=text,
                source=self.source,
                source_type="file_import",
                timestamp=timestamp,
                external_id=external_id,
                user_id=str(row.get(self.mapping["user_id"], "")),
                rating=float(row[self.mapping["rating"]]) if row.get(self.mapping["rating"]) else None,
                metadata={key: value for key, value in row.items() if key not in self.mapping.values()},
            )
            records.append((entry, row))
        return FetchResult(records, str(len(records)))
