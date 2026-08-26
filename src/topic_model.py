"""
Topic modeling with graceful degradation, same philosophy as sentiment.py.

Preferred backend: BERTopic
    sentence-transformers (all-MiniLM-L6-v2) embeddings -> UMAP dimensionality
    reduction -> HDBSCAN density clustering -> c-TF-IDF keyword extraction per
    cluster. This handles short, noisy text well and produces interpretable
    topics (a ranked keyword list) without needing a predefined topic count.

Fallback backend: TF-IDF + MiniBatchKMeans
    Used when bertopic/umap/hdbscan/sentence-transformers aren't installed,
    torch isn't available, or the environment has no GPU/network for model
    downloads. Produces a fixed number of clusters with top TF-IDF terms as
    the topic label -- less flexible (topic count is fixed, not discovered)
    but fully interpretable and dependency-light.

Both backends expose the same interface: `fit_transform(docs) -> (topic_ids,
topic_info)`, so the rest of the pipeline (prioritize.py, app.py) doesn't
care which one is active. `topic_info` is a dict: {topic_id: {"label": str,
"keywords": [str,...], "size": int}}.

Retraining strategy: the app re-fits the model on the last N messages
(config.ROLLING_WINDOW_SIZE) every config.RETRAIN_EVERY_N new messages,
rather than incrementally updating -- simpler, avoids topic drift/ID
instability, and is cheap enough at this data volume to run synchronously.
In production this would move to an async batch job.
"""

import re
from collections import Counter

import numpy as np

from . import config

STOPWORDS = set("""
a an the this that these those is are was were be been being have has had
do does did will would shall should may might must can could i you he she
it we they me him her us them my your his its our their and or but if
then so because as until while of at by for with about against between
into through during before after above below to from up down in out on
off over under again further here there when where why how all any both
each few more most other some such no nor not only own same than too very
s t just don now app it's i'm i've
""".split())

TOKEN_RE = re.compile(r"[a-zA-Z']+")


def _tokenize(text):
    return [w.lower() for w in TOKEN_RE.findall(text) if w.lower() not in STOPWORDS and len(w) > 2]


class BaseTopicModel:
    name = "base"

    def fit_transform(self, docs):
        raise NotImplementedError


class BERTopicModel(BaseTopicModel):
    name = "bertopic"

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        from bertopic import BERTopic
        from umap import UMAP
        from hdbscan import HDBSCAN

        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        umap_model = UMAP(
            n_neighbors=10, n_components=5, min_dist=0.0,
            metric="cosine", random_state=42,
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=max(config.MIN_TOPIC_SIZE, 3),
            metric="euclidean", cluster_selection_method="eom",
            prediction_data=True,
        )
        self.model = BERTopic(
            embedding_model=self.embedder,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            calculate_probabilities=False,
            verbose=False,
        )

    def fit_transform(self, docs):
        embeddings = self.embedder.encode(docs, show_progress_bar=False)
        topic_ids, _ = self.model.fit_transform(docs, embeddings)
        topic_ids = list(topic_ids)

        info_df = self.model.get_topic_info()
        topic_info = {}
        for _, row in info_df.iterrows():
            tid = int(row["Topic"])
            if tid == -1:
                label = "Outliers / Unclustered"
                keywords = []
            else:
                pairs = self.model.get_topic(tid) or []
                keywords = [w for w, _ in pairs[:6]]
                label = ", ".join(keywords[:3]) if keywords else f"Topic {tid}"
            topic_info[tid] = {
                "label": label,
                "keywords": keywords,
                "size": int(row["Count"]),
            }
        return topic_ids, topic_info


class SimpleTopicModel(BaseTopicModel):
    """TF-IDF + MiniBatchKMeans fallback. No heavy deps beyond scikit-learn."""
    name = "tfidf_kmeans"

    def __init__(self, n_topics=None):
        self.n_topics = n_topics or config.N_FALLBACK_TOPICS

    def fit_transform(self, docs):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans

        n_topics = min(self.n_topics, max(2, len(docs) // 5))
        vectorizer = TfidfVectorizer(
            tokenizer=_tokenize, preprocessor=lambda x: x, lowercase=False,
            max_df=0.9, min_df=1,
        )
        X = vectorizer.fit_transform(docs)
        km = KMeans(n_clusters=n_topics, random_state=42, n_init=10)
        labels = km.fit_predict(X)

        terms = np.array(vectorizer.get_feature_names_out())
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]

        topic_info = {}
        counts = Counter(labels)
        for tid in range(n_topics):
            top_terms = [terms[i] for i in order_centroids[tid, :6] if i < len(terms)]
            topic_info[tid] = {
                "label": ", ".join(top_terms[:3]) if top_terms else f"Topic {tid}",
                "keywords": top_terms,
                "size": int(counts.get(tid, 0)),
            }
        return list(labels), topic_info


def build_topic_model():
    """
    Attempts the BERTopic pipeline; falls back to TF-IDF/KMeans on any
    import or initialization failure (missing deps, no network for the
    sentence-transformer weights, etc.). Logs which backend was chosen.
    """
    try:
        model = BERTopicModel()
        return model
    except Exception as e:
        print(f"[topic_model] BERTopic unavailable ({e!r}), falling back to TF-IDF/KMeans.")
        return SimpleTopicModel()
