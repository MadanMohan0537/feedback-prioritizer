"""
Data ingestion: simulates a real-time feedback stream by replaying a CSV
of pre-collected reviews/tickets with configurable pacing.

In production this generator's interface (yielding dicts with timestamp,
text, source) would be swapped for a Kafka consumer, a webhook handler, or
polling loop against Twitter/Reddit/Zendesk APIs -- nothing downstream of
`stream_feedback()` needs to change.
"""

import csv
import time
from datetime import datetime

from . import config


def load_feedback_rows(csv_path=None):
    """Load all feedback rows from CSV into memory, sorted by timestamp."""
    csv_path = csv_path or config.SAMPLE_CSV
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError("Feedback CSV must contain a 'text' column.")
        for row in reader:
            if str(row.get("text", "")).strip():
                rows.append(row)
    rows.sort(key=lambda r: r["timestamp"])
    return rows


def stream_feedback(csv_path=None, delay_seconds=None, loop=False, start_index=0):
    """
    Generator that yields one feedback item at a time, simulating arrival
    in near real-time. Each yielded item is a dict:
        {timestamp, text, source, rating, id}

    Parameters
    ----------
    delay_seconds : float or None
        Seconds to sleep between yields. None uses config.DEFAULT_DELAY_SECONDS.
        The Streamlit app typically sets this to 0 and paces itself via
        st_autorefresh instead, so the generator can be pulled from on demand.
    loop : bool
        If True, restarts from the beginning after exhausting the CSV
        (useful for a long-running demo).
    start_index : int
        Index to resume streaming from (used by the Streamlit app to keep
        its place across reruns).
    """
    delay = config.DEFAULT_DELAY_SECONDS if delay_seconds is None else delay_seconds
    rows = load_feedback_rows(csv_path)

    idx = start_index
    while True:
        if idx >= len(rows):
            if loop:
                idx = 0
            else:
                return
        row = rows[idx]
        item = {
            **row,
            "id": idx,
            "timestamp": row["timestamp"],
            "text": row["text"],
            "source": row.get("source", "unknown"),
            "rating": row.get("rating"),
        }
        yield item
        idx += 1
        if delay > 0:
            time.sleep(delay)


def get_batch(rows, start_index, batch_size):
    """
    Pull a fixed-size batch starting at start_index from an already-loaded
    row list. Returns (items, next_index). Used by the Streamlit app which
    advances the stream one batch per refresh tick instead of using the
    blocking generator directly.
    """
    end_index = min(start_index + batch_size, len(rows))
    batch = []
    for idx in range(start_index, end_index):
        row = rows[idx]
        batch.append({
            **row,
            "id": idx,
            "timestamp": row["timestamp"],
            "text": row["text"],
            "source": row.get("source", "unknown"),
            "rating": row.get("rating"),
        })
    return batch, end_index
