"""Small, dependency-free privacy guard for the ingestion boundary."""

import hashlib
import re

EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d .()\-]{7,}\d)(?!\w)")
IP_ADDRESS = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def redact_pii(text: str) -> tuple[str, int]:
    redacted, count = EMAIL.subn("[EMAIL]", text)
    redacted, phone_count = PHONE.subn("[PHONE]", redacted)
    redacted, ip_count = IP_ADDRESS.subn("[IP_ADDRESS]", redacted)
    return redacted, count + phone_count + ip_count


def hash_identifier(value: str, salt: str = "pulse-local") -> str:
    if not value:
        return ""
    return hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()
