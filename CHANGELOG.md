# Changelog

## v1.1
- Added a sidebar CSV uploader so the dashboard can run on real feedback data, not just the bundled demo set.
- Added a minimum-priority-score filter on the Issues tab to cut through low-priority noise.
- Rewrote README with a sharper framing, a "what's new" section, and an honest limitations/roadmap split.
- Added `.gitattributes` to normalize line endings across contributors on different OSes.

## v1.0
- Initial pipeline: simulated streaming ingestion, transformer sentiment with VADER fallback, BERTopic topic modeling with TF-IDF/KMeans fallback, weighted prioritization engine, four-tab Streamlit dashboard.
