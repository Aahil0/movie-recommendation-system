"""Explicit-rating baselines and observed-entry regularized matrix factorization."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


class BiasModel:
    def __init__(self, regularization=10, iterations=10):
        self.regularization, self.iterations = regularization, iterations

    def fit(self, ratings):
        if ratings.empty:
            raise ValueError("Cannot train on empty ratings")
        if ratings[["userId", "movieId", "rating"]].isna().any().any():
            raise ValueError("Training ratings cannot contain missing values")
        if ratings.duplicated(["userId", "movieId"]).any() or not ratings.rating.between(1, 5).all():
            raise ValueError("Training data requires unique user/movie pairs and ratings within 1–5")
        self.users = pd.Index(np.sort(ratings.userId.unique()))
        self.movies = pd.Index(np.sort(ratings.movieId.unique()))
        u = self.users.get_indexer(ratings.userId)
        i = self.movies.get_indexer(ratings.movieId)
        r = ratings.rating.to_numpy(dtype=float)
        self.global_mean = float(r.mean())
        self.user_bias, self.movie_bias = np.zeros(len(self.users)), np.zeros(len(self.movies))
        nu = np.bincount(u, minlength=len(self.users)) + self.regularization
        ni = np.bincount(i, minlength=len(self.movies)) + self.regularization
        for _ in range(self.iterations):
            self.user_bias = np.bincount(u, weights=r-self.global_mean-self.movie_bias[i], minlength=len(self.users)) / nu
            self.movie_bias = np.bincount(i, weights=r-self.global_mean-self.user_bias[u], minlength=len(self.movies)) / ni
        return self

    def predict(self, user_ids, movie_ids):
        u, i = self.users.get_indexer(user_ids), self.movies.get_indexer(movie_ids)
        ub = np.zeros(len(u)); ib = np.zeros(len(i))
        ub[u >= 0] = self.user_bias[u[u >= 0]]
        ib[i >= 0] = self.movie_bias[i[i >= 0]]
        return np.clip(self.global_mean + ub + ib, 1, 5)

    def score_user(self, user_id, movie_ids):
        return self.predict(np.repeat(user_id, len(movie_ids)), movie_ids)


class ObservedALS(BiasModel):
    """Ridge ALS minimizes error only on observed training ratings."""
    def __init__(self, factors=24, regularization=15, epochs=10, seed=42):
        super().__init__()
        self.factors, self.factor_regularization = factors, regularization
        self.epochs, self.seed = epochs, seed

    def fit(self, ratings):
        super().fit(ratings)
        u, i = self.users.get_indexer(ratings.userId), self.movies.get_indexer(ratings.movieId)
        residual = ratings.rating.to_numpy() - self.global_mean - self.user_bias[u] - self.movie_bias[i]
        matrix = csr_matrix((residual, (u, i)), shape=(len(self.users), len(self.movies)))
        transposed = matrix.T.tocsr()
        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0, .1, (len(self.users), self.factors))
        self.movie_factors = rng.normal(0, .1, (len(self.movies), self.factors))
        ridge = self.factor_regularization * np.eye(self.factors)
        for _ in range(self.epochs):
            for target, fixed, observed in [
                (self.user_factors, self.movie_factors, matrix),
                (self.movie_factors, self.user_factors, transposed),
            ]:
                for row in range(observed.shape[0]):
                    start, end = observed.indptr[row:row+2]
                    features = fixed[observed.indices[start:end]]
                    values = observed.data[start:end]
                    target[row] = solve(features.T @ features + ridge, features.T @ values,
                                        assume_a="pos", check_finite=False)
        return self

    def predict(self, user_ids, movie_ids):
        # Use unbounded bias sum before the final rating clamp.
        u, i = self.users.get_indexer(user_ids), self.movies.get_indexer(movie_ids)
        values = np.full(len(u), self.global_mean)
        values[u >= 0] += self.user_bias[u[u >= 0]]
        values[i >= 0] += self.movie_bias[i[i >= 0]]
        known = (u >= 0) & (i >= 0)
        values[known] += np.einsum("ij,ij->i", self.user_factors[u[known]], self.movie_factors[i[known]])
        return np.clip(values, 1, 5)


class LegacySVD(BiasModel):
    """Corrected, bounded legacy reconstruction retained as an honest comparator."""
    def fit(self, ratings):
        super().fit(ratings)
        matrix = ratings.pivot(index="userId", columns="movieId", values="rating").reindex(index=self.users, columns=self.movies)
        self.means = matrix.mean(axis=1).to_numpy()
        factors = min(50, min(matrix.shape)-1)
        self.svd = TruncatedSVD(n_components=max(1, factors), random_state=42)
        uf = self.svd.fit_transform(matrix.sub(self.means, axis=0).fillna(0))
        self.predictions = uf @ self.svd.components_ + self.means[:, None]
        return self

    def predict(self, user_ids, movie_ids):
        values = super().predict(user_ids, movie_ids)
        u, i = self.users.get_indexer(user_ids), self.movies.get_indexer(movie_ids)
        known = (u >= 0) & (i >= 0)
        values[known] = self.predictions[u[known], i[known]]
        return np.clip(values, 1, 5)
