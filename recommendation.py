"""Movie-to-movie recommendations shared by Streamlit and the notebook."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, normalize

ARTIFACT_VERSION = 1


def validate_movies(movies):
    required = {"movieId", "title", "genres"}
    if not required.issubset(movies.columns):
        raise ValueError(f"Movie data requires {sorted(required)}")
    result = movies.copy().sort_values("movieId").reset_index(drop=True)
    if result[list(required)].isna().any().any():
        raise ValueError("Movie IDs, titles and genres cannot be missing")
    if result.movieId.duplicated().any():
        raise ValueError("Movie IDs must be unique")
    numeric = pd.to_numeric(result.movieId, errors="raise")
    if not np.isfinite(numeric).all() or not (numeric > 0).all() or not (numeric == np.floor(numeric)).all():
        raise ValueError("Movie IDs must be positive integers")
    result["movieId"] = numeric.astype(int)
    return result


def catalog_fingerprint(movies):
    frame = validate_movies(movies)[["movieId", "title", "genres"]]
    return hashlib.sha256(frame.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


class ContentFeatures:
    def __init__(self, movies, metadata=None):
        self.movies = validate_movies(movies)
        self.genre_sets = [set(value.split("|")) for value in self.movies.genres]
        self.encoder = MultiLabelBinarizer(sparse_output=True)
        binary = self.encoder.fit_transform(self.genre_sets)
        # Each genre remains one atomic feature (including Sci-Fi and Film-Noir).
        self.genres = TfidfTransformer().fit_transform(binary)
        self.details = None
        self.has_details = np.zeros(len(self.movies), dtype=bool)
        if metadata is not None and len(metadata):
            if metadata.movieId.duplicated().any():
                raise ValueError("Metadata movie IDs must be unique")
            merged = self.movies[["movieId"]].merge(metadata, on="movieId", how="left")
            documents = []
            for _, row in merged.iterrows():
                overview = str(row.get("overview", "") or "")
                if overview == "nan":
                    overview = ""
                parts = [overview]
                for field in ["keywords", "cast", "director"]:
                    value = row.get(field, "")
                    if pd.notna(value):
                        parts.extend(f"{field}_{name.strip().replace(' ', '_')}"
                                     for name in str(value).split("|") if name.strip())
                documents.append(" ".join(parts))
            self.has_details = np.array([bool(doc.strip()) for doc in documents])
            if self.has_details.any():
                vectorizer = TfidfVectorizer(stop_words="english", max_features=12000)
                try:
                    self.details = vectorizer.fit_transform(documents)
                except ValueError:  # Only empty/stop-word documents: keep the genre fallback.
                    self.has_details[:] = False

    def scores(self, source):
        genre = (self.genres @ self.genres[source].T).toarray().ravel()
        content = genre.copy()
        if self.details is not None and self.has_details[source]:
            detail = (self.details @ self.details[source].T).toarray().ravel()
            eligible = self.has_details
            content[eligible] = .65 * genre[eligible] + .35 * detail[eligible]
        return np.clip(content, 0, 1), np.clip(genre, 0, 1)


def fit_item_neighbors(movies, ratings, top_k=40, min_support=3, shrinkage=20):
    """Train-only positive-rating cosine neighbors with co-like support shrinkage."""
    movies = validate_movies(movies)
    user_ids = np.sort(ratings.userId.unique())
    movie_pos = pd.Index(movies.movieId).get_indexer(ratings.movieId)
    if np.any(movie_pos < 0):
        raise ValueError("Ratings contain IDs outside the movie catalog")
    user_pos = pd.Index(user_ids).get_indexer(ratings.userId)
    positive = ratings.rating.to_numpy() >= 4
    liked = csr_matrix((ratings.rating.to_numpy()[positive] - 3,
                        (user_pos[positive], movie_pos[positive])),
                       shape=(len(user_ids), len(movies)), dtype=np.float64)
    binary = liked.copy()
    binary.data[:] = 1
    item_vectors = normalize(liked.T, norm="l2")
    neighbors = {}
    # Block computation bounds working memory; no catalog-squared matrix is saved.
    for start in range(0, len(movies), 128):
        stop = min(start + 128, len(movies))
        cosines = (item_vectors[start:stop] @ item_vectors.T).toarray()
        support = (binary[:, start:stop].T @ binary).toarray()
        scores = cosines * support / (support + shrinkage)
        scores[support < min_support] = 0
        for offset, source in enumerate(range(start, stop)):
            scores[offset, source] = 0
            ordered = np.lexsort((movies.movieId.to_numpy(), -scores[offset]))
            ordered = ordered[scores[offset, ordered] > 0][:top_k]
            neighbors[int(movies.movieId.iloc[source])] = [
                [int(movies.movieId.iloc[target]), round(float(scores[offset, target]), 7),
                 int(support[offset, target])] for target in ordered
            ]
    return neighbors


def movie_statistics(movies, ratings, prior_count=50):
    movies = validate_movies(movies)
    global_mean = float(ratings.rating.mean()) if len(ratings) else 3.0
    stats = ratings.groupby("movieId").rating.agg(["mean", "count"])
    result = movies[["movieId"]].merge(stats, left_on="movieId", right_index=True, how="left")
    result["count"] = result["count"].fillna(0).astype(int)
    result["mean"] = result["mean"].fillna(global_mean)
    result["quality"] = (result["mean"] * result["count"] + prior_count * global_mean) / (result["count"] + prior_count)
    return result.rename(columns={"mean": "avg_rating", "count": "rating_count"})


class RecommendationEngine:
    def __init__(self, movies, stats=None, neighbors=None, metadata=None):
        self.movies = validate_movies(movies)
        self.positions = {int(mid): i for i, mid in enumerate(self.movies.movieId)}
        self.features = ContentFeatures(self.movies, metadata)
        self.neighbors = neighbors or {}
        self.artifact_status = "Genre fallback"
        self.stats = self.movies[["movieId"]].copy()
        if stats is not None:
            if stats.movieId.duplicated().any():
                raise ValueError("Statistics movie IDs must be unique")
            self.stats = self.stats.merge(stats, on="movieId", how="left")
        for column, default in [("quality", 3.0), ("rating_count", 0), ("avg_rating", 3.0)]:
            if column not in self.stats:
                self.stats[column] = default
            self.stats[column] = self.stats[column].fillna(default)
        self.quality = np.clip((self.stats.quality.to_numpy() - 1) / 4, 0, 1)

    @classmethod
    def from_directory(cls, path):
        path = Path(path)
        movies = pd.read_csv(path / "movies_app.csv")
        metadata_path = path / "movie_metadata.csv"
        metadata = pd.read_csv(metadata_path).fillna("") if metadata_path.exists() else None
        manifest_path = path / "model_manifest.json"
        if not manifest_path.exists():
            return cls(movies, metadata=metadata)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("version") != ARTIFACT_VERSION or manifest.get("catalog_sha256") != catalog_fingerprint(movies):
            raise ValueError("Model artifacts do not match this catalog; rebuild app data")
        for filename, checksum in manifest["files"].items():
            if hashlib.sha256((path / filename).read_bytes()).hexdigest() != checksum:
                raise ValueError(f"Model artifact checksum mismatch: {filename}")
        stats = pd.read_csv(path / "movie_stats.csv")
        with gzip.open(path / "item_neighbors.json.gz", "rt", encoding="utf-8") as stream:
            neighbors = {int(mid): rows for mid, rows in json.load(stream).items()}
        engine = cls(movies, stats, neighbors, metadata)
        engine.artifact_status = "Genres + viewer-rating patterns"
        return engine

    def find_movie_id(self, title):
        matches = self.movies.loc[self.movies.title.str.casefold() == str(title).strip().casefold(), "movieId"]
        if len(matches) == 0:
            raise ValueError(f"Film not found: {title}")
        if len(matches) > 1:
            raise ValueError("Title is ambiguous; select by movie ID")
        return int(matches.iloc[0])

    def collaborative_scores(self, movie_id):
        values = np.zeros(len(self.movies))
        for target, score, _ in self.neighbors.get(int(movie_id), []):
            if target in self.positions:
                values[self.positions[target]] = score
        if values.max() > 0:
            values /= values.max()
        return values

    def recommend(self, movie_id, k=10, mode="hybrid", diversity=.08):
        if int(movie_id) not in self.positions:
            raise ValueError(f"Unknown movie ID: {movie_id}")
        if not isinstance(k, int) or k < 1:
            raise ValueError("Recommendation count must be a positive integer")
        if mode not in {"hybrid", "content", "genres", "collaborative"}:
            raise ValueError("Unknown recommendation mode")
        if not 0 <= diversity <= .25:
            raise ValueError("Diversity must be between 0 and .25")
        source = self.positions[int(movie_id)]
        content, genre = self.features.scores(source)
        if mode == "genres":
            content = genre
        collab = self.collaborative_scores(movie_id)
        if mode == "collaborative":
            base = collab
            eligible = collab > 0
        elif mode in {"content", "genres"} or not collab.any():
            base = .95 * content + .05 * self.quality
            eligible = content > 0
        else:
            base = .4 * content + .55 * collab + .05 * self.quality
            eligible = (genre > 0) | (collab > 0)
        eligible[source] = False
        # Tie breakers depend on movie data/ID, never CSV row position.
        order = np.lexsort((self.movies.movieId, -self.stats.rating_count,
                            -self.quality, -np.round(base, 12)))
        pool = order[eligible[order]][:max(200, k)]
        chosen, ranking_scores = [], []
        remaining = pool.tolist()
        while remaining and len(chosen) < min(k, len(pool)):
            penalties = np.zeros(len(remaining))
            if chosen and diversity and mode != "collaborative":
                similarities = (self.features.genres[remaining] @ self.features.genres[chosen].T).toarray()
                penalties = similarities.max(axis=1)
            adjusted = base[remaining] - diversity * penalties
            # np.argmax preserves our deterministic quality/ID tie order.
            best = int(np.argmax(adjusted))
            chosen.append(remaining.pop(best))
            ranking_scores.append(float(adjusted[best]))
        result = self.movies.iloc[chosen][["movieId", "title", "genres"]].copy()
        result["content_score"] = content[chosen]
        result["collab_score"] = collab[chosen]
        result["score"] = base[chosen]
        result["ranking_score"] = ranking_scores
        result["avg_rating"] = self.stats.avg_rating.iloc[chosen].to_numpy()
        result["rating_count"] = self.stats.rating_count.iloc[chosen].to_numpy()
        reasons = []
        for target in chosen:
            shared = sorted(self.features.genre_sets[source] & self.features.genre_sets[target])
            signals = []
            if mode not in {"content", "genres"} and collab[target] > 0:
                signals.append("Liked by some of the same viewers")
            if shared:
                signals.append("Shared genres: " + " · ".join(shared))
            if self.features.has_details[source] and self.features.has_details[target] and mode not in {"collaborative", "genres"}:
                signals.append("Includes available story and credit metadata")
            reasons.append(". ".join(signals) or "Similar viewer-rating patterns")
        result["reason"] = reasons
        return result.reset_index(drop=True)


def export_app_data(movies, ratings, directory, neighbors_k=40):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    movies = validate_movies(movies)[["movieId", "title", "genres"]]
    movies.to_csv(directory / "movies_app.csv", index=False)
    movie_statistics(movies, ratings).to_csv(directory / "movie_stats.csv", index=False, float_format="%.8f")
    neighbors = fit_item_neighbors(movies, ratings, top_k=neighbors_k)
    payload = json.dumps(neighbors, separators=(",", ":")).encode()
    # Deterministic compressed artifacts, no raw user IDs/ratings in the app bundle.
    (directory / "item_neighbors.json.gz").write_bytes(gzip.compress(payload, mtime=0))
    files = ["movie_stats.csv", "item_neighbors.json.gz"]
    manifest = {"version": ARTIFACT_VERSION, "catalog_sha256": catalog_fingerprint(movies),
                "training_scope": "All available ratings for serving; evaluation uses train-only artifacts",
                "ratings": len(ratings), "users": int(ratings.userId.nunique()),
                "neighbor_count": neighbors_k, "min_co_likes": 3, "support_shrinkage": 20,
                "quality_prior_count": 50,
                "training_ratings_sha256": hashlib.sha256(
                    pd.util.hash_pandas_object(
                        ratings[["userId", "movieId", "rating", "timestamp"]].sort_values(["userId", "movieId"]),
                        index=False).to_numpy().tobytes()
                ).hexdigest(),
                "files": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in files}}
    (directory / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
