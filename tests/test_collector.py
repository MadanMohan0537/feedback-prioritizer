import json
import tempfile
from pathlib import Path
import unittest

from src.collector.connectors import FileConnector, SyntheticConnector
from src.collector.models import FeedbackEntry
from src.collector.pipeline import collect
from src.collector.privacy import hash_identifier, redact_pii
from src.collector.storage import FeedbackStore


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.database = Path(self.temp.name) / "feedback.db"
        self.store = FeedbackStore(self.database)

    def tearDown(self):
        self.temp.cleanup()

    def test_collect_is_idempotent_and_preserves_raw_payload(self):
        connector = SyntheticConnector(count=8, seed=7)
        first = collect(connector, self.store)
        second = collect(connector, self.store)
        self.assertEqual(first.accepted, 8)
        self.assertEqual(second.duplicates, 8)
        self.assertEqual(self.store.count(), 8)
        self.assertEqual(self.store.checkpoint("synthetic"), "8")

    def test_upserts_a_changed_external_record(self):
        original = FeedbackEntry("Export fails", "zendesk", "2026-01-01T00:00:00Z", "ticket-1")
        changed = FeedbackEntry("Export is fixed", "zendesk", "2026-01-01T00:00:00Z", "ticket-1")
        self.assertTrue(self.store.save(original))
        self.assertFalse(self.store.save(changed))
        self.assertEqual(self.store.list()[0]["text"], "Export is fixed")

    def test_file_import_and_jsonl_export(self):
        csv_path = Path(self.temp.name) / "feedback.csv"
        csv_path.write_text("id,timestamp,text,rating\n1,2026-01-01T00:00:00Z,Great app,5\n", encoding="utf-8")
        report = collect(FileConnector(csv_path, source="survey"), self.store)
        output = Path(self.temp.name) / "feedback.jsonl"
        self.assertEqual(report.accepted, 1)
        self.assertEqual(self.store.export_jsonl(output), 1)
        self.assertEqual(json.loads(output.read_text())["source"], "survey")

    def test_privacy_helpers(self):
        redacted, count = redact_pii("Email me at person@example.com or 415-555-1212")
        self.assertEqual(count, 2)
        self.assertNotIn("person@example.com", redacted)
        self.assertEqual(hash_identifier("user-1", "salt"), hash_identifier("user-1", "salt"))

    def test_delete_user(self):
        user_id = hash_identifier("customer-7", "salt")
        self.store.save(FeedbackEntry("Hello", "survey", "2026-01-01T00:00:00Z", "1", user_id=user_id))
        self.assertEqual(self.store.delete_user(user_id), 1)
        self.assertEqual(self.store.count(), 0)


if __name__ == "__main__":
    unittest.main()
