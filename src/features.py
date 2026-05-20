"""
src/features.py
Feature Engineering: TF-IDF, Bag-of-Words
Python 3.14 compatible.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Builds and saves TF-IDF and BoW feature matrices."""

    def __init__(self, max_features: int = 10000, ngram_range: tuple[int, int] = (1, 2)) -> None:
        self.max_features = max_features
        self.ngram_range  = ngram_range

        self.tfidf_vec = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            min_df=2,
        )
        self.bow_vec = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
        )
        self._fitted = False

    def fit_transform(self, texts: list[str]) -> tuple:
        """Fit both vectorizers and return (tfidf_matrix, bow_matrix)."""
        logger.info("Fitting TF-IDF and BoW on %d texts", len(texts))
        X_tfidf = self.tfidf_vec.fit_transform(texts)
        X_bow   = self.bow_vec.fit_transform(texts)
        self._fitted = True
        logger.info("TF-IDF shape: %s  |  BoW shape: %s", X_tfidf.shape, X_bow.shape)
        return X_tfidf, X_bow

    def transform(self, texts: list[str]) -> tuple:
        """Transform new texts using fitted vectorizers."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform() first.")
        return self.tfidf_vec.transform(texts), self.bow_vec.transform(texts)

    def transform_single(self, text: str) -> tuple:
        """Transform a single text string."""
        return self.transform([text])

    def save(self, models_dir: str) -> None:
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(models_dir) / "tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(self.tfidf_vec, f)
        with open(Path(models_dir) / "bow_vectorizer.pkl", "wb") as f:
            pickle.dump(self.bow_vec, f)
        logger.info("Vectorizers saved to %s", models_dir)

    def load(self, models_dir: str) -> None:
        with open(Path(models_dir) / "tfidf_vectorizer.pkl", "rb") as f:
            self.tfidf_vec = pickle.load(f)
        with open(Path(models_dir) / "bow_vectorizer.pkl", "rb") as f:
            self.bow_vec   = pickle.load(f)
        self._fitted = True
        logger.info("Vectorizers loaded from %s", models_dir)

    def get_top_features(self, n: int = 20) -> list[str]:
        """Return top-n feature names from TF-IDF vocabulary."""
        vocab = self.tfidf_vec.get_feature_names_out()
        return list(vocab[:n])
