"""
Core recommendation logic, independent of Streamlit so it can be unit-tested.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_FILE = DATA_DIR / "movies.csv.gz"

_stemmer = PorterStemmer()


def stem_text(text: str) -> str:
    """Apply Porter stemming to every whitespace-separated token."""
    return " ".join(_stemmer.stem(token) for token in text.split())


def load_movies(path: Path = PROCESSED_FILE) -> pd.DataFrame:
    """Load the processed movie table (movie_id, title, tags)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. Run `python build_model.py` first."
        )
    df = pd.read_csv(path)
    expected = {"movie_id", "title", "tags"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Processed dataset is missing columns: {sorted(missing)}")
    df["tags"] = df["tags"].fillna("").astype(str)
    df["movie_id"] = df["movie_id"].astype(int)
    return df.reset_index(drop=True)


def build_similarity(movies: pd.DataFrame, max_features: int = 5000) -> np.ndarray:
    """Bag-of-words cosine similarity between every pair of movies."""
    vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
    vectors = vectorizer.fit_transform(movies["tags"])
    return cosine_similarity(vectors).astype(np.float32)


class Recommender:
    """Content-based recommender backed by a precomputed similarity matrix."""

    def __init__(self, movies: pd.DataFrame, similarity: np.ndarray | None = None):
        if similarity is None:
            similarity = build_similarity(movies)
        if similarity.shape != (len(movies), len(movies)):
            raise ValueError(
                f"Similarity shape {similarity.shape} does not match {len(movies)} movies"
            )
        self.movies = movies
        self.similarity = similarity
        # Map title -> first row index (titles are not guaranteed unique in TMDB).
        self._title_index = {
            title: idx for idx, title in reversed(list(enumerate(movies["title"])))
        }

    @property
    def titles(self) -> list[str]:
        return sorted(self._title_index)

    def recommend(self, title: str, n: int = 5) -> pd.DataFrame:
        """Return the `n` most similar movies (excluding the movie itself)."""
        if title not in self._title_index:
            raise KeyError(f"Movie '{title}' not found")
        idx = self._title_index[title]
        scores = self.similarity[idx]
        # argsort descending; skip the query movie itself.
        order = np.argsort(-scores, kind="stable")
        order = order[order != idx][:n]
        result = self.movies.iloc[order][["movie_id", "title"]].copy()
        result["score"] = scores[order]
        return result.reset_index(drop=True)
