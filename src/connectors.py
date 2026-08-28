"""Opt-in connectors that normalize external feedback to one safe contract."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from urllib.request import Request, urlopen

from . import config


@dataclass(frozen=True)
class ConnectorConfig:
    name: str
    url: str
    token: str = ""
    items_path: tuple = ()


def _dig(payload, path):
    value = payload
    for key in path:
        if not isinstance(value, dict):
            return []
        value = value.get(key, [])
    return value if isinstance(value, list) else []


def _text(item, *keys):
    for key in keys:
        value = item.get(key)
        if value:
            if isinstance(value, dict):
                value = value.get("body") or value.get("plain_body") or value.get("text")
            if value:
                return str(value).strip()
    return ""


def normalize_item(source, item, index=0):
    """Normalize GitHub, Slack, Zendesk, Intercom, and app-review records."""
    source = source.lower()
    text = _text(item, "text", "body", "description", "content", "title", "review")
    if source == "github":
        title = _text(item, "title")
        body = _text(item, "body")
        text = ". ".join(part for part in (title, body) if part)
    elif source == "intercom":
        source_record = item.get("source") if isinstance(item.get("source"), dict) else {}
        text = text or _text(source_record, "body")
    timestamp = (
        item.get("timestamp") or item.get("created_at") or item.get("created")
        or item.get("updated_at") or datetime.now(timezone.utc).isoformat()
    )
    author = item.get("author") if isinstance(item.get("author"), dict) else {}
    account = item.get("account") if isinstance(item.get("account"), dict) else {}
    return {
        "id": str(item.get("id") or item.get("number") or f"{source}-{index}"),
        "timestamp": str(timestamp),
        "text": text,
        "source": source,
        "account_id": str(item.get("account_id") or account.get("id") or author.get("id") or ""),
        "customer_tier": str(item.get("customer_tier") or account.get("tier") or "unknown"),
        "arr": item.get("arr") or account.get("arr") or 0,
        "churn_risk": item.get("churn_risk") or account.get("churn_risk") or "",
        "product_area": item.get("product_area") or "",
        "external_url": item.get("html_url") or item.get("url") or item.get("permalink") or "",
    }


def normalize_payload(source, payload, items_path=()):
    items = _dig(payload, items_path) if items_path else payload
    if isinstance(items, dict):
        for key in ("items", "results", "tickets", "conversations", "reviews", "issues", "messages"):
            if isinstance(items.get(key), list):
                items = items[key]
                break
    if not isinstance(items, list):
        raise ValueError("Connector response does not contain a list of feedback items.")
    rows = [normalize_item(source, item, index) for index, item in enumerate(items[:config.MAX_CONNECTOR_ITEMS])]
    return [row for row in rows if row["text"]]


def fetch_connector(connector, timeout=20):
    headers = {"Accept": "application/json", "User-Agent": "feedback-prioritizer/2"}
    if connector.token:
        headers["Authorization"] = f"Bearer {connector.token}"
    request = Request(connector.url, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return normalize_payload(connector.name, payload, connector.items_path)


def configured_connectors():
    """Load opt-in connector endpoints without exposing credentials in code."""
    definitions = {
        "github": ("FEEDBACK_GITHUB_URL", "FEEDBACK_GITHUB_TOKEN", ()),
        "slack": ("FEEDBACK_SLACK_URL", "FEEDBACK_SLACK_TOKEN", ("messages",)),
        "zendesk": ("FEEDBACK_ZENDESK_URL", "FEEDBACK_ZENDESK_TOKEN", ("tickets",)),
        "intercom": ("FEEDBACK_INTERCOM_URL", "FEEDBACK_INTERCOM_TOKEN", ("conversations",)),
        "app_reviews": ("FEEDBACK_APP_REVIEWS_URL", "FEEDBACK_APP_REVIEWS_TOKEN", ()),
    }
    connectors = []
    for name, (url_key, token_key, path) in definitions.items():
        url = os.getenv(url_key, "").strip()
        if url:
            connectors.append(ConnectorConfig(name, url, os.getenv(token_key, "").strip(), path))
    return connectors
