"""Connector protocol and resilient JSON HTTP client."""

from dataclasses import dataclass
import json
import random
import time
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..models import FeedbackEntry


@dataclass
class FetchResult:
    records: list[tuple[FeedbackEntry, dict]]
    next_cursor: str = ""


class Connector(Protocol):
    name: str

    def fetch(self, checkpoint: str = "") -> FetchResult: ...


def get_json(url: str, headers=None, timeout: int = 30, retries: int = 3):
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "pulse/1.0", **(headers or {})})
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8")), dict(response.headers)
        except HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == retries:
                raise
            retry_after = error.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt + random.random()
        except URLError:
            if attempt == retries:
                raise
            delay = 2**attempt + random.random()
        time.sleep(delay)
    raise RuntimeError("HTTP retry loop exhausted")
