"""Per-run collection metrics."""

from dataclasses import asdict, dataclass


@dataclass
class IngestionReport:
    source: str
    fetched: int = 0
    accepted: int = 0
    duplicates: int = 0
    invalid: int = 0
    redacted: int = 0
    errors: int = 0

    def to_dict(self):
        return asdict(self)
