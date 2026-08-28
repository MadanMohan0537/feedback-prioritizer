"""Split multi-issue feedback into independently analyzable opinion units."""

import re


BOUNDARY_RE = re.compile(
    r"(?<=[.!?])\s+|\s+(?:but|however|although|yet|while)\s+",
    flags=re.IGNORECASE,
)


def split_opinion_units(text, min_chars=12):
    """Conservatively split sentences and contrast clauses.

    This lightweight fallback makes aspect-level sentiment possible without an
    LLM. Short fragments are joined back to their nearest unit.
    """
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return []
    raw = [part.strip(" ,;:-") for part in BOUNDARY_RE.split(cleaned) if part.strip(" ,;:-")]
    units = []
    for part in raw:
        if len(part) < min_chars and units:
            units[-1] = f"{units[-1]}; {part}"
        else:
            units.append(part)
    return units or [cleaned]


def expand_feedback_item(item):
    """Create one enriched record per opinion while preserving provenance."""
    units = split_opinion_units(item.get("text", ""))
    parent_id = item.get("id")
    return [
        {
            **item,
            "id": f"{parent_id}:{index}",
            "parent_id": parent_id,
            "original_text": item.get("text", ""),
            "text": unit,
            "opinion_index": index,
            "opinion_count": len(units),
        }
        for index, unit in enumerate(units)
    ]
