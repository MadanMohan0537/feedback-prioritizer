"""Canonical, versioned feedback contract shared by every connector."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Optional
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class FeedbackEntry:
    text: str
    source: str
    timestamp: str
    external_id: str
    id: str = field(default_factory=lambda: str(uuid4()))
    source_type: str = "feedback"
    ingested_at: str = field(default_factory=utc_now)
    updated_at: str = ""
    user_id: str = ""
    language: str = ""
    rating: Optional[float] = None
    url: str = ""
    product: str = ""
    product_version: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"
    content_hash: str = ""

    def __post_init__(self) -> None:
        self.text = " ".join(str(self.text).split())
        self.source = str(self.source).strip().lower()
        self.external_id = str(self.external_id).strip()
        if not self.text:
            raise ValueError("Feedback text cannot be empty.")
        if not self.source:
            raise ValueError("Feedback source cannot be empty.")
        if not self.external_id:
            raise ValueError("Feedback external_id cannot be empty.")
        self.content_hash = self.content_hash or self.calculate_hash()

    def calculate_hash(self) -> str:
        normalized = " ".join(self.text.casefold().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FeedbackEntry":
        fields = cls.__dataclass_fields__
        return cls(**{key: val for key, val in value.items() if key in fields})
