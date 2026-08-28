import unittest

from src.connectors import normalize_item, normalize_payload


class ConnectorTests(unittest.TestCase):
    def test_normalizes_github_issue_with_customer_metadata(self):
        row = normalize_item("github", {
            "number": 7,
            "title": "Sync fails",
            "body": "Enterprise accounts cannot sync.",
            "created_at": "2026-01-01T00:00:00Z",
            "account": {"id": "acme", "tier": "enterprise", "arr": 120000},
        })
        self.assertIn("Sync fails", row["text"])
        self.assertEqual(row["account_id"], "acme")
        self.assertEqual(row["arr"], 120000)

    def test_drops_empty_connector_records(self):
        rows = normalize_payload("zendesk", {"tickets": [{"id": 1}, {"id": 2, "description": "Help"}]}, ("tickets",))
        self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
