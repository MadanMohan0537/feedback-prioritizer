"""Collection orchestration shared by CLI, cron, and future webhooks."""

import os

from .models import utc_now
from .privacy import hash_identifier, redact_pii
from .quality import IngestionReport


def collect(connector, store, redact=True, hash_users=True, salt=None):
    report = IngestionReport(source=connector.name)
    checkpoint = store.checkpoint(connector.name)
    result = connector.fetch(checkpoint)
    report.fetched = len(result.records)
    salt = salt or os.getenv("PULSE_HASH_SALT", "pulse-local")

    for entry, raw in result.records:
        try:
            if redact:
                entry.text, count = redact_pii(entry.text)
                report.redacted += count
                entry.content_hash = entry.calculate_hash()
            if hash_users and entry.user_id:
                entry.user_id = hash_identifier(entry.user_id, salt)
            if store.save(entry, raw):
                report.accepted += 1
            else:
                report.duplicates += 1
        except Exception as error:
            report.invalid += 1
            store.dead_letter(connector.name, raw, str(error))

    store.set_checkpoint(connector.name, result.next_cursor or checkpoint, utc_now())
    return report
