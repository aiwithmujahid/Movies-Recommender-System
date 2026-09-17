"""
Build the recommender artifacts from the raw TMDB 5000 dataset.

Input  (place in ./data/raw/):
    tmdb_5000_movies.csv
    tmdb_5000_credits.csv
    (Kaggle: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

Output (written to ./data/):
    movies.csv.gz  -> columns: movie_id, title, tags

The similarity matrix is intentionally NOT stored on disk: a 4806 x 4806
float matrix is ~185 MB, which is too large for Git / Streamlit Cloud.
Instead the app recomputes it at start-up from the compact `tags` column
(takes well under a second) and caches it in memory.

Usage:
    python build_model.py
"""

from __future__ import annotations

import ast
import sys

import pandas as pd

from recommender import DATA_DIR, PROCESSED_FILE, stem_text

RAW_DIR = DATA_DIR / "raw"
MOVIES_CSV = RAW_DIR / "tmdb_5000_movies.csv"
CREDITS_CSV = RAW_DIR / "tmdb_5000_credits.csv"


def _parse_names(value: str) -> list[str]:
    """Extract the `name` field from a JSON-like list stored as a string."""
    try:
        return [item["name"] for item in ast.literal_eval(value)]
    except (ValueError, SyntaxError, TypeError, KeyError):
        return []


def _top_cast(value: str, limit: int = 3) -> list[str]:
    return _parse_names(value)[:limit]


def _director(value: str) -> list[str]:
    try:
        return [item["name"] for item in ast.literal_eval(value) if item.get("job") == "Director"]
    except (ValueError, SyntaxError, TypeError):
        return []


def _collapse(tokens: list[str]) -> list[str]:
    """Remove spaces inside multi-word tokens so 'Sam Worthington' -> 'SamWorthington'."""
    return [token.replace(" ", "") for token in tokens]


def build() -> pd.DataFrame:
    for path in (MOVIES_CSV, CREDITS_CSV):
        if not path.exists():
            sys.exit(
                f"Missing {path}.\n"
                "Download the TMDB 5000 dataset from Kaggle and place both CSV files "
                f"in {RAW_DIR}/"
            )

    movies = pd.read_csv(MOVIES_CSV)
    credits = pd.read_csv(CREDITS_CSV)

    movies = movies.merge(credits, on="title")
    movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]
    movies = movies.dropna().drop_duplicates(subset="movie_id").reset_index(drop=True)

    movies["genres"] = movies["genres"].apply(_parse_names).apply(_collapse)
    movies["keywords"] = movies["keywords"].apply(_parse_names).apply(_collapse)
    movies["cast"] = movies["cast"].apply(_top_cast).apply(_collapse)
    movies["crew"] = movies["crew"].apply(_director).apply(_collapse)
    movies["overview"] = movies["overview"].apply(lambda text: str(text).split())

    movies["tags"] = (
        movies["overview"] + movies["genres"] + movies["keywords"] + movies["cast"] + movies["crew"]
    )
    movies["tags"] = movies["tags"].apply(lambda tokens: stem_text(" ".join(tokens).lower()))

    result = movies[["movie_id", "title", "tags"]].copy()
    result["movie_id"] = result["movie_id"].astype(int)
    return result


def main() -> None:
    df = build()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_FILE, index=False, compression="gzip")
    print(f"Wrote {len(df):,} movies to {PROCESSED_FILE} ({PROCESSED_FILE.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
