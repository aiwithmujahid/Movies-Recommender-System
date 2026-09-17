# 🎬 Movies Recommender System

A content-based movie recommender built on the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) and served with Streamlit. Pick a movie you like and it returns the five most similar titles (with posters from TMDB).

**How it works**

1. For every movie, the overview, genres, keywords, top‑3 cast and director are merged into a single `tags` document and Porter‑stemmed (`build_model.py`).
2. The documents are vectorised with a bag‑of‑words `CountVectorizer` (5 000 features, English stop‑words removed).
3. Cosine similarity between all movie vectors is computed once at start‑up and cached in memory; recommendations are the top‑N nearest neighbours.

## Project layout

```
.
├── app.py                    # Streamlit UI
├── recommender.py            # Data loading + similarity + Recommender class
├── build_model.py            # Regenerates data/movies.csv.gz from the raw Kaggle CSVs
├── data/
│   ├── movies.csv.gz         # Processed dataset (committed, ~0.9 MB)
│   └── raw/                  # tmdb_5000_movies.csv / tmdb_5000_credits.csv (git‑ignored)
├── tests/test_recommender.py
├── requirements.txt
├── runtime.txt
└── .streamlit/
    ├── config.toml
    └── secrets.toml.example
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional – enables posters
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
#   then paste your TMDB API key into that file

streamlit run app.py
```

Run the test-suite with `python -m pytest`.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to <https://share.streamlit.io>, click **New app**, choose the repo/branch and set **Main file path** to `app.py`.
3. (Optional) In **Advanced settings → Secrets** add:

   ```toml
   TMDB_API_KEY = "your_tmdb_api_key"
   ```

   Get a free key at <https://www.themoviedb.org/settings/api>. Without a key the app still works; posters are replaced with a placeholder.
4. Click **Deploy**.

No large binary files are required: the processed dataset is under 1 MB and the similarity matrix is rebuilt in memory on first load (≈1 s).

## Rebuilding the dataset

```bash
# download tmdb_5000_movies.csv and tmdb_5000_credits.csv from Kaggle into data/raw/
python build_model.py
```
