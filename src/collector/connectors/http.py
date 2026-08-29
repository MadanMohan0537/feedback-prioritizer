"""Generic, Typeform, and Zendesk API connectors."""

from datetime import datetime, timezone
from urllib.parse import urlencode, urlsplit, urlunsplit, parse_qsl

from .base import FetchResult, get_json
from ..models import FeedbackEntry


def _with_query(url, **params):
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query))
    query.update({key: value for key, value in params.items() if value not in (None, "")})
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


class GenericJSONConnector:
    def __init__(self, name, url, token="", items_key="items"):
        self.name, self.url, self.token, self.items_key = name, url, token, items_key

    def fetch(self, checkpoint=""):
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        payload, _ = get_json(_with_query(self.url, since=checkpoint), headers)
        items = payload.get(self.items_key, []) if isinstance(payload, dict) else payload
        records = []
        for index, item in enumerate(items):
            text = item.get("text") or item.get("body") or item.get("description") or item.get("review") or ""
            entry = FeedbackEntry(
                text=text,
                source=self.name,
                source_type=str(item.get("type") or "api_feedback"),
                timestamp=str(item.get("timestamp") or item.get("created_at") or datetime.now(timezone.utc).isoformat()),
                updated_at=str(item.get("updated_at") or ""),
                external_id=str(item.get("id") or f"{self.name}-{index}"),
                user_id=str(item.get("user_id") or ""),
                rating=float(item["rating"]) if item.get("rating") is not None else None,
                url=str(item.get("url") or item.get("html_url") or ""),
                metadata={key: value for key, value in item.items() if key not in {"text", "body", "description"}},
            )
            records.append((entry, item))
        next_cursor = str(payload.get("next_cursor") or payload.get("cursor") or "") if isinstance(payload, dict) else ""
        return FetchResult(records, next_cursor)


class TypeformConnector:
    name = "typeform"

    def __init__(self, form_id, token, base_url="https://api.typeform.com"):
        self.form_id, self.token, self.base_url = form_id, token, base_url.rstrip("/")

    def fetch(self, checkpoint=""):
        url = _with_query(f"{self.base_url}/forms/{self.form_id}/responses", since=checkpoint, page_size=1000)
        payload, _ = get_json(url, {"Authorization": f"Bearer {self.token}"})
        records = []
        for response in payload.get("items", []):
            answers = []
            for answer in response.get("answers", []):
                value = answer.get(answer.get("type", ""))
                if isinstance(value, dict):
                    value = value.get("label") or value.get("other")
                if value not in (None, ""):
                    answers.append(str(value))
            entry = FeedbackEntry(
                text=" | ".join(answers), source=self.name, source_type="survey_response",
                timestamp=response.get("submitted_at") or response.get("landed_at"),
                external_id=response.get("response_id") or response.get("token"),
                user_id=str(response.get("hidden", {}).get("user_id", "")),
                metadata={"form_id": self.form_id, "hidden": response.get("hidden", {})},
            )
            records.append((entry, response))
        cursor = max((entry.timestamp for entry, _ in records), default=checkpoint)
        return FetchResult(records, cursor)


class ZendeskConnector:
    name = "zendesk"

    def __init__(self, subdomain, token, email=""):
        self.subdomain, self.token, self.email = subdomain, token, email

    def fetch(self, checkpoint=""):
        start = checkpoint or "1"
        url = f"https://{self.subdomain}.zendesk.com/api/v2/incremental/tickets.json?{urlencode({'start_time': start, 'include': 'comment_events'})}"
        headers = {"Authorization": f"Bearer {self.token}"}
        payload, _ = get_json(url, headers)
        records = []
        for ticket in payload.get("tickets", []):
            entry = FeedbackEntry(
                text=ticket.get("description") or ticket.get("subject") or "",
                source=self.name, source_type="support_ticket",
                timestamp=ticket.get("created_at") or datetime.now(timezone.utc).isoformat(),
                updated_at=ticket.get("updated_at") or "",
                external_id=str(ticket.get("id")),
                user_id=str(ticket.get("requester_id") or ""),
                url=ticket.get("url") or "",
                metadata={"status": ticket.get("status"), "priority": ticket.get("priority"), "tags": ticket.get("tags", [])},
            )
            records.append((entry, ticket))
        return FetchResult(records, str(payload.get("end_time") or start))


class AppleReviewsConnector:
    name = "apple_app_store"

    def __init__(self, app_id, token):
        self.app_id, self.token = app_id, token

    def fetch(self, checkpoint=""):
        url = f"https://api.appstoreconnect.apple.com/v1/apps/{self.app_id}/customerReviews?limit=200&sort=-createdDate"
        payload, _ = get_json(url, {"Authorization": f"Bearer {self.token}"})
        records = []
        for review in payload.get("data", []):
            attributes = review.get("attributes", {})
            entry = FeedbackEntry(
                text=attributes.get("reviewBody") or attributes.get("title") or "",
                source=self.name, source_type="app_review",
                timestamp=attributes.get("createdDate") or datetime.now(timezone.utc).isoformat(),
                external_id=str(review.get("id")),
                user_id=str(attributes.get("reviewerNickname") or ""),
                rating=float(attributes["rating"]) if attributes.get("rating") is not None else None,
                product=self.app_id,
                metadata={"title": attributes.get("title"), "territory": attributes.get("territory")},
            )
            records.append((entry, review))
        cursor = max((entry.timestamp for entry, _ in records), default=checkpoint)
        return FetchResult(records, cursor)


class GooglePlayConnector:
    name = "google_play"

    def __init__(self, package_name, access_token):
        self.package_name, self.access_token = package_name, access_token

    def fetch(self, checkpoint=""):
        url = f"https://androidpublisher.googleapis.com/androidpublisher/v3/applications/{self.package_name}/reviews?maxResults=100"
        payload, _ = get_json(url, {"Authorization": f"Bearer {self.access_token}"})
        records = []
        for review in payload.get("reviews", []):
            comments = review.get("comments", [])
            comment = next((value.get("userComment", {}) for value in comments if value.get("userComment")), {})
            modified = comment.get("lastModified", {}).get("seconds", "")
            timestamp = datetime.fromtimestamp(int(modified), timezone.utc).isoformat() if modified else datetime.now(timezone.utc).isoformat()
            entry = FeedbackEntry(
                text=comment.get("text") or "", source=self.name, source_type="app_review",
                timestamp=timestamp, external_id=str(review.get("reviewId")),
                user_id=str(review.get("authorName") or ""),
                rating=float(comment["starRating"]) if comment.get("starRating") is not None else None,
                product=self.package_name,
                product_version=str(comment.get("appVersionName") or ""),
                metadata={"device": comment.get("device"), "language": comment.get("reviewerLanguage")},
            )
            records.append((entry, review))
        token = payload.get("tokenPagination", {}).get("nextPageToken") or checkpoint
        return FetchResult(records, str(token))


class IntercomConnector:
    name = "intercom"

    def __init__(self, token):
        self.token = token

    def fetch(self, checkpoint=""):
        url = _with_query("https://api.intercom.io/conversations", per_page=150, starting_after=checkpoint)
        payload, _ = get_json(url, {"Authorization": f"Bearer {self.token}", "Intercom-Version": "2.11"})
        records = []
        for conversation in payload.get("conversations", []):
            source = conversation.get("source", {})
            contacts = conversation.get("contacts", {}).get("contacts", [])
            entry = FeedbackEntry(
                text=source.get("body") or conversation.get("title") or "",
                source=self.name, source_type="support_conversation",
                timestamp=datetime.fromtimestamp(int(conversation.get("created_at", 0)), timezone.utc).isoformat(),
                updated_at=datetime.fromtimestamp(int(conversation.get("updated_at", 0)), timezone.utc).isoformat(),
                external_id=str(conversation.get("id")),
                user_id=str(contacts[0].get("id") if contacts else ""),
                url=str(conversation.get("url") or ""),
                metadata={"state": conversation.get("state"), "priority": conversation.get("priority")},
            )
            records.append((entry, conversation))
        cursor = payload.get("pages", {}).get("next", {}).get("starting_after") or checkpoint
        return FetchResult(records, str(cursor))
