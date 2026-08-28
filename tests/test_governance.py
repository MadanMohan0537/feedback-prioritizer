import tempfile
import unittest
from pathlib import Path

from src.governance import TopicRegistry, topic_key


class TopicGovernanceTests(unittest.TestCase):
    def test_overrides_survive_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "topics.json"
            info = {0: {"label": "sync, broken", "keywords": ["sync", "broken"], "size": 3}}
            key = topic_key(info[0])
            registry = TopicRegistry(path)
            registry.update(key, label="Sync reliability", owner="Platform", status="planned")
            governed = TopicRegistry(path).apply(info)[0]
            self.assertEqual(governed["label"], "Sync reliability")
            self.assertEqual(governed["owner"], "Platform")
            self.assertEqual(governed["status"], "planned")

    def test_similar_keywords_keep_existing_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "topics.json"
            original = {0: {"label": "sync", "keywords": ["sync", "broken", "mobile"], "size": 3}}
            registry = TopicRegistry(path)
            first = registry.apply(original)[0]
            registry.update(first["stable_key"], label="Sync reliability")
            shifted = {2: {"label": "sync", "keywords": ["sync", "broken", "devices"], "size": 4}}
            second = TopicRegistry(path).apply(shifted)[2]
            self.assertEqual(second["stable_key"], first["stable_key"])
            self.assertEqual(second["label"], "Sync reliability")


if __name__ == "__main__":
    unittest.main()
