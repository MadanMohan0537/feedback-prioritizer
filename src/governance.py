"""Persistent, human-controlled topic names, ownership, status, and merges."""

import hashlib
import json
import os
from pathlib import Path

from . import config


def topic_key(info):
    keywords = sorted(str(value).strip().lower() for value in info.get("keywords", [])[:6])
    basis = "|".join(keywords) or str(info.get("label", "unknown")).strip().lower()
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


class TopicRegistry:
    def __init__(self, path=None):
        self.path = Path(path or config.TOPIC_REGISTRY_PATH)
        self.data = {"version": 1, "topics": {}, "merges": {}}
        self.load()

    def load(self):
        if not self.path.exists():
            return
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self.data.update(loaded)
        except (OSError, ValueError):
            # A damaged optional registry must not stop analysis.
            return

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.path)

    def update(self, key, *, label=None, owner=None, status=None):
        record = self.data["topics"].setdefault(key, {})
        if label is not None:
            record["label"] = str(label).strip()
        if owner is not None:
            record["owner"] = str(owner).strip()
        if status is not None:
            record["status"] = str(status).strip().lower()
        self.save()

    def merge(self, source_key, target_key):
        if source_key != target_key:
            self.data["merges"][source_key] = target_key
            self.save()

    def resolve_key(self, info):
        candidate_key = topic_key(info)
        if candidate_key in self.data["topics"] or candidate_key in self.data["merges"]:
            return self.data["merges"].get(candidate_key, candidate_key)
        candidate_words = {str(value).lower() for value in info.get("keywords", [])[:6]}
        best_key, best_score = candidate_key, 0.0
        for existing_key, record in self.data["topics"].items():
            existing_words = {str(value).lower() for value in record.get("keywords", [])}
            union = candidate_words | existing_words
            score = len(candidate_words & existing_words) / len(union) if union else 0.0
            if score > best_score:
                best_key, best_score = existing_key, score
        return best_key if best_score >= 0.5 else candidate_key

    def apply(self, topic_info):
        governed = {}
        changed = False
        for topic_id, info in topic_info.items():
            target_key = self.resolve_key(info)
            override = self.data["topics"].setdefault(target_key, {})
            keywords = [str(value).lower() for value in info.get("keywords", [])[:6]]
            if keywords and override.get("keywords") != keywords:
                override["keywords"] = keywords
                changed = True
            governed[topic_id] = {
                **info,
                "stable_key": target_key,
                "label": override.get("label") or info.get("label", f"Topic {topic_id}"),
                "owner": override.get("owner", "Unassigned"),
                "status": override.get("status", "new"),
            }
        if changed:
            self.save()
        return governed
