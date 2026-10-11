"""
nlp/topic_model.py — TF-IDF vectorization and KMeans topic clustering.

Discovers technology topics from document text using unsupervised NLP.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from config import (
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE,
    TFIDF_MIN_DF, TFIDF_MAX_DF,
    KMEANS_CLUSTER_RANGE, KMEANS_RANDOM_STATE,
)

logger = logging.getLogger(__name__)


class TopicModel:
    """
    TF-IDF + KMeans topic discovery pipeline.

    Usage:
        model = TopicModel()
        model.fit(documents_df)
        topics_df = model.get_topics()
        labels = model.predict(new_texts)
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            min_df=TFIDF_MIN_DF,
            max_df=TFIDF_MAX_DF,
            stop_words="english",
        )
        self.kmeans = None
        self.tfidf_matrix = None
        self.best_k = None
        self.feature_names = None

    def fit(self, df, text_column="text_for_nlp"):
        """
        Fit TF-IDF and KMeans on document texts.

        Tries multiple K values and selects the one with
        the highest silhouette score.

        Args:
            df: DataFrame with a text column
            text_column: Name of the column containing text

        Returns:
            self (for chaining)
        """
        texts = df[text_column].fillna("").values
        logger.info("Fitting TF-IDF on %d documents...", len(texts))

        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        logger.info("TF-IDF matrix shape: %s", self.tfidf_matrix.shape)

        # Try different K values, pick best silhouette
        best_score = -1
        best_model = None

        for k in KMEANS_CLUSTER_RANGE:
            if k >= len(texts):
                continue
            km = KMeans(
                n_clusters=k,
                random_state=KMEANS_RANDOM_STATE,
                n_init=10,
                max_iter=300,
            )
            labels = km.fit_predict(self.tfidf_matrix)
            score = silhouette_score(
                self.tfidf_matrix, labels, sample_size=min(5000, len(texts))
            )
            logger.info("  K=%d, silhouette=%.4f", k, score)

            if score > best_score:
                best_score = score
                best_model = km
                self.best_k = k

        self.kmeans = best_model
        logger.info("Selected K=%d (silhouette=%.4f)", self.best_k, best_score)
        return self

    def get_cluster_labels(self):
        """Return cluster labels for all fitted documents."""
        if self.kmeans is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.kmeans.labels_

    def get_top_terms(self, n_terms=10):
        """
        Get the top TF-IDF terms for each cluster.

        Returns:
            Dict mapping cluster_id -> list of top terms
        """
        if self.kmeans is None:
            raise ValueError("Model not fitted. Call fit() first.")

        top_terms = {}
        centroids = self.kmeans.cluster_centers_

        for cluster_id in range(self.best_k):
            centroid = centroids[cluster_id]
            top_indices = centroid.argsort()[-n_terms:][::-1]
            terms = [self.feature_names[i] for i in top_indices]
            top_terms[cluster_id] = terms

        return top_terms

    def get_topics(self, n_terms=10):
        """
        Get a DataFrame of discovered topics with their top terms.

        Returns:
            DataFrame with columns: cluster_id, top_terms, document_count
        """
        labels = self.get_cluster_labels()
        top_terms = self.get_top_terms(n_terms)

        topics = []
        for cluster_id in range(self.best_k):
            count = int(np.sum(labels == cluster_id))
            topics.append({
                "cluster_id": cluster_id,
                "top_terms": ", ".join(top_terms[cluster_id]),
                "document_count": count,
                "name": f"Topic {cluster_id}",  # Can be manually refined later
            })

        return pd.DataFrame(topics)

    def search(self, query, top_n=10):
        """
        Find documents most similar to a search query using cosine similarity.

        Args:
            query: Search string
            top_n: Number of results to return

        Returns:
            Array of indices of most similar documents
        """
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-top_n:][::-1]
        return top_indices, similarities[top_indices]
