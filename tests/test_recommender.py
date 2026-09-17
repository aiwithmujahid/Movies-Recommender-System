import numpy as np
import pandas as pd
import pytest

from recommender import PROCESSED_FILE, Recommender, build_similarity, load_movies, stem_text


@pytest.fixture(scope="module")
def toy_movies() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movie_id": [1, 2, 3, 4],
            "title": ["Space War", "Space Battle", "Romance Story", "Space War"],
            "tags": [
                "space alien war laser",
                "space alien battle laser",
                "love kiss wedding",
                "space war reboot alien",
            ],
        }
    )


def test_stem_text():
    assert stem_text("running runs runner") == "run run runner"


def test_similarity_shape(toy_movies):
    sim = build_similarity(toy_movies)
    assert sim.shape == (4, 4)
    assert np.allclose(np.diag(sim), 1.0)


def test_recommend_excludes_self_and_ranks(toy_movies):
    rec = Recommender(toy_movies)
    result = rec.recommend("Space Battle", n=2)
    assert list(result["title"]) == ["Space War", "Space War"]
    assert "Space Battle" not in result["title"].values
    assert result["score"].is_monotonic_decreasing


def test_duplicate_titles_map_to_first_row(toy_movies):
    rec = Recommender(toy_movies)
    assert rec.titles == ["Romance Story", "Space Battle", "Space War"]
    assert rec._title_index["Space War"] == 0


def test_unknown_title_raises(toy_movies):
    with pytest.raises(KeyError):
        Recommender(toy_movies).recommend("Nope")


def test_mismatched_similarity_raises(toy_movies):
    with pytest.raises(ValueError):
        Recommender(toy_movies, np.zeros((2, 2)))


@pytest.mark.skipif(not PROCESSED_FILE.exists(), reason="processed dataset not built")
def test_real_dataset_end_to_end():
    movies = load_movies()
    assert len(movies) > 4000
    rec = Recommender(movies)
    result = rec.recommend("Avatar")
    assert len(result) == 5
    assert "Avatar" not in result["title"].values
