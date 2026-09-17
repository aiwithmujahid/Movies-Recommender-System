import os

import requests
import streamlit as st

from recommender import PROCESSED_FILE, Recommender, build_similarity, load_movies

st.set_page_config(page_title="Movie Recommender System", page_icon="🎬", layout="wide")

TMDB_API_URL = "https://api.themoviedb.org/3/movie/{movie_id}"
TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w500{poster_path}"
PLACEHOLDER_POSTER = "https://placehold.co/500x750/1f1f1f/ffffff?text=No+Poster"
NUM_RECOMMENDATIONS = 5


def get_tmdb_api_key() -> str | None:
    """Read the TMDB key from Streamlit secrets or the environment (optional)."""
    try:
        key = st.secrets.get("TMDB_API_KEY")  # type: ignore[attr-defined]
        if key:
            return str(key)
    except (FileNotFoundError, KeyError, AttributeError):
        pass
    return os.environ.get("TMDB_API_KEY") or None


@st.cache_resource(show_spinner="Loading movie data…")
def get_recommender() -> Recommender:
    movies = load_movies(PROCESSED_FILE)
    similarity = build_similarity(movies)
    return Recommender(movies, similarity)


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def fetch_poster(movie_id: int, api_key: str | None) -> str:
    """Return a poster URL for the TMDB movie id, falling back to a placeholder."""
    if not api_key:
        return PLACEHOLDER_POSTER
    try:
        response = requests.get(
            TMDB_API_URL.format(movie_id=movie_id),
            params={"api_key": api_key, "language": "en-US"},
            timeout=8,
        )
        response.raise_for_status()
        poster_path = response.json().get("poster_path")
        if poster_path:
            return TMDB_IMAGE_URL.format(poster_path=poster_path)
    except (requests.RequestException, ValueError):
        pass
    return PLACEHOLDER_POSTER


def main() -> None:
    st.title("🎬 Movie Recommender System")
    st.caption(
        "Content-based recommendations built from the TMDB 5000 dataset "
        "(genres, keywords, cast, crew and overview)."
    )

    try:
        recommender = get_recommender()
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Could not load the movie dataset: {exc}")
        st.stop()

    api_key = get_tmdb_api_key()
    if not api_key:
        st.info(
            "Posters are disabled. Add a `TMDB_API_KEY` to Streamlit secrets "
            "or the environment to show movie posters.",
            icon="ℹ️",
        )

    selected_title = st.selectbox(
        "Choose a movie you like:",
        recommender.titles,
        index=None,
        placeholder="Start typing a movie title…",
    )

    if st.button("Recommend", type="primary", disabled=selected_title is None):
        try:
            results = recommender.recommend(selected_title, n=NUM_RECOMMENDATIONS)
        except KeyError as exc:
            st.error(str(exc))
            return

        if results.empty:
            st.warning("No recommendations found for this movie.")
            return

        st.subheader(f"Because you liked **{selected_title}**")
        columns = st.columns(len(results))
        for column, row in zip(columns, results.itertuples(index=False)):
            with column:
                st.image(fetch_poster(int(row.movie_id), api_key), width="stretch")
                st.markdown(f"**{row.title}**")
                st.caption(f"Similarity: {row.score:.0%}")


if __name__ == "__main__":
    main()
