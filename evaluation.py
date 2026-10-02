"""Train-only evaluation, cold-start coverage, and computed result exports."""
from __future__ import annotations

import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from rating_models import BiasModel, LegacySVD, ObservedALS
from recommendation import movie_statistics, validate_movies


def load_movielens(directory):
    directory = Path(directory)
    movies = pd.read_csv(directory / "movies.dat", sep="::", engine="python",
                         encoding="latin-1", names=["movieId", "title", "genres"])
    ratings = pd.read_csv(directory / "ratings.dat", sep="::", engine="python",
                          names=["userId", "movieId", "rating", "timestamp"])
    movies = validate_movies(movies)
    if ratings.isna().any().any() or ratings.duplicated(["userId", "movieId"]).any():
        raise ValueError("Ratings contain missing values or duplicate user/movie pairs")
    if not ratings.rating.between(1, 5).all() or not ratings.movieId.isin(movies.movieId).all():
        raise ValueError("Invalid ratings or movie IDs")
    return movies, ratings


def split_ratings(ratings, protocol="random", seed=42, fraction=.2):
    if protocol == "random":
        train, test = [], []
        for _, group in ratings.groupby("userId"):
            if len(group) < 2:
                train.append(group)
                continue
            a, b = train_test_split(group, test_size=fraction, random_state=seed)
            train.append(a); test.append(b)
        if not test:
            raise ValueError("No eligible test interactions")
        return pd.concat(train), pd.concat(test), {"protocol": "per-user random", "seed": seed}
    if protocol != "temporal":
        raise ValueError("Unknown evaluation protocol")
    # A single global cutoff keeps equal-time groups together and prevents learning
    # from other users' future interactions; cold users/items are scored, not dropped.
    cutoff = int(ratings.timestamp.quantile(1-fraction, interpolation="lower"))
    train, test = ratings[ratings.timestamp <= cutoff], ratings[ratings.timestamp > cutoff]
    if train.empty or test.empty:
        raise ValueError("Timestamp groups cannot produce a nonempty strict holdout")
    assert train.timestamp.max() < test.timestamp.min()
    return train.copy(), test.copy(), {"protocol": "strict global timestamp", "cutoff": cutoff}


class MeanBaseline:
    def __init__(self, train, catalog, kind="movie"):
        self.global_mean = float(train.rating.mean())
        self.kind = kind
        self.means = train.groupby("userId" if kind == "user" else "movieId").rating.mean()
        self.stats = movie_statistics(catalog, train).set_index("movieId")

    def predict(self, users, movies):
        if self.kind == "global":
            return np.full(len(users), self.global_mean)
        key = users if self.kind == "user" else movies
        return pd.Series(key).map(self.means).fillna(self.global_mean).to_numpy()

    def score_user(self, user, movies):
        if self.kind == "popularity":
            return self.stats.rating_count.reindex(movies).to_numpy()
        if self.kind == "bayesian":
            return self.stats.quality.reindex(movies).to_numpy()
        return self.predict(np.repeat(user, len(movies)), movies)


class ValidatedRanker:
    """Choose relevance/popularity balance on an inner training-only holdout."""
    def __init__(self, model, train, catalog, weight):
        self.model, self.weight = model, weight
        self.counts = train.groupby("movieId").size().reindex(catalog.movieId,fill_value=0)
        self.maximum = max(1,float(np.log1p(self.counts.max())))

    def predict(self, users, movies):
        return self.model.predict(users,movies)

    def score_user(self, user, movies):
        popularity=np.log1p(self.counts.reindex(movies,fill_value=0).to_numpy())/self.maximum
        if user not in self.model.users:
            return popularity
        relevance=(self.model.score_user(user,movies)-1)/4
        return self.weight*relevance+(1-self.weight)*popularity


def select_ranker(training, catalog, protocol, seed, factors=24, epochs=10):
    inner_train, validation, split_info=split_ratings(training,protocol,seed+1000)
    seen=inner_train.groupby("userId").movieId.apply(set).to_dict()
    relevant=validation[validation.rating>=4].groupby("userId").movieId.apply(set)
    # Fixed validation sample, never sampled from the outer test.
    users=relevant.index.to_numpy()
    if len(users)>1000:
        users=np.sort(np.random.default_rng(seed).choice(users,1000,replace=False))
    candidates=catalog.movieId.to_numpy()
    popularity=MeanBaseline(inner_train,catalog,"popularity")
    choices=[]
    for name,model in [
        ("Legacy SVD",LegacySVD().fit(inner_train)),
        ("Observed ALS",ObservedALS(factors=factors,epochs=epochs,seed=seed).fit(inner_train)),
    ]:
        measurements={weight:[] for weight in [0,.25,.5,.75,1]}
        counts=inner_train.groupby("movieId").size().reindex(candidates,fill_value=0)
        maximum=max(1,float(np.log1p(counts.max())))
        for user in users:
            eligible=candidates[~np.isin(candidates,list(seen.get(user,set())))]
            pop=np.log1p(counts.reindex(eligible).to_numpy())/maximum
            values=(model.score_user(user,eligible)-1)/4
            for weight in measurements:
                scores=weight*values+(1-weight)*pop if user in model.users else pop
                measurements[weight].append(ranking_metrics(scores,eligible,relevant[user])[3])
        for weight,values in measurements.items():
            choices.append({"model":name,"weight":weight,"validation_NDCG_at_10":float(np.mean(values))})
    # Stable preference for the first model/weight if validation metrics tie.
    best=max(choices,key=lambda row:row["validation_NDCG_at_10"])
    return {**best,"validation_users":len(users),"inner_train_ratings":len(inner_train),
            "inner_validation_ratings":len(validation),"split":split_info,
            "selection_objective":"Mean validation NDCG@10; outer test not used",
            "all_choices":choices}


def ranking_metrics(scores, candidate_ids, relevant, k=10):
    order = np.lexsort((candidate_ids, -scores))[:k]
    hits = np.isin(candidate_ids[order], list(relevant)).astype(float)
    precision = float(hits.sum() / k)
    recall = float(hits.sum() / len(relevant))
    discount = np.log2(np.arange(len(hits)) + 2)
    dcg = float((hits / discount).sum())
    ideal = float((1 / np.log2(np.arange(min(k, len(relevant))) + 2)).sum())
    return precision, recall, float(hits.any()), dcg/ideal


def evaluate_split(movies, ratings, protocol="random", seed=42, factors=24, epochs=10):
    train, test, info = split_ratings(ratings, protocol, seed)
    seen = train.groupby("userId").movieId.apply(set).to_dict()
    relevant = test[test.rating >= 4].groupby("userId").movieId.apply(set)
    ids = movies.movieId.to_numpy()
    cold_user = ~test.userId.isin(train.userId)
    cold_movie = ~test.movieId.isin(train.movieId)
    info.update({"train_ratings": len(train), "test_ratings": len(test),
                 "cold_user_ratings": int(cold_user.sum()), "cold_movie_ratings": int(cold_movie.sum()),
                 "ranking_users": len(relevant), "test_users_without_relevant_movies": int(test.userId.nunique()-len(relevant)),
                 "candidate_policy": "Full catalog excluding only training-rated items",
                 "rating_policy": "All test ratings; explicit bias/global fallback for cold IDs",
                 "als": {"factors": factors, "epochs": epochs, "regularization": 15},
                 "hyperparameter_policy": "ALS defaults fixed; ranking blend selected on inner training-only validation"})
    models = {
        "Global mean": MeanBaseline(train,movies,"global"),
        "User mean": MeanBaseline(train,movies,"user"),
        "Movie mean": MeanBaseline(train,movies),
        "Bias": BiasModel().fit(train),
        "Legacy SVD": LegacySVD().fit(train),
        "Observed ALS": ObservedALS(factors=factors,epochs=epochs,seed=seed).fit(train),
    }
    selected=select_ranker(train,movies,protocol,seed,factors,epochs)
    info["validated_ranker"]=selected
    models["Validated blend"]=ValidatedRanker(models[selected["model"]],train,movies,selected["weight"])
    rows, per_user = [], []
    for name, model in models.items():
        prediction = model.predict(test.userId.to_numpy(), test.movieId.to_numpy())
        assert np.isfinite(prediction).all() and len(prediction) == len(test)
        row = {"protocol": protocol, "seed": seed, "model": name,
               "RMSE": float(np.sqrt(np.mean((test.rating.to_numpy()-prediction)**2))),
               "MAE": float(np.mean(np.abs(test.rating.to_numpy()-prediction))),
               "rating_coverage": 1.0,
               "warm_RMSE": float(np.sqrt(np.mean((test.rating.to_numpy()[~(cold_user|cold_movie)]-prediction[~(cold_user|cold_movie)])**2)))}
        if name in {"Global mean", "User mean"}:
            rows.append(row)
            continue
        measurements, exposed = [], set()
        for user, truth in relevant.items():
            eligible = ids[~np.isin(ids, list(seen.get(user, set())))]
            scores = model.score_user(user, eligible)
            values = ranking_metrics(scores, eligible, truth)
            measurements.append(values)
            ranked = np.lexsort((eligible, -scores))[:10]
            exposed.update(eligible[ranked])
            per_user.append({"protocol": protocol,"seed": seed,"model": name,"userId": int(user),
                             "Precision@10": values[0],"Recall@10":values[1],"HitRate@10":values[2],"NDCG@10":values[3]})
        row.update(dict(zip(["Precision@10","Recall@10","HitRate@10","NDCG@10"], np.mean(measurements,axis=0).tolist())))
        row["catalog_coverage"] = len(exposed)/len(ids)
        rows.append(row)
    for name, kind in [("Popularity","popularity"),("Bayesian quality","bayesian")]:
        model = MeanBaseline(train,movies,kind)
        measurements, exposed = [], set()
        for user, truth in relevant.items():
            eligible = ids[~np.isin(ids,list(seen.get(user,set())))]
            scores = model.score_user(user,eligible)
            values = ranking_metrics(scores,eligible,truth)
            measurements.append(values)
            exposed.update(eligible[np.lexsort((eligible,-scores))[:10]])
            per_user.append({"protocol":protocol,"seed":seed,"model":name,"userId":int(user),
                             "Precision@10":values[0],"Recall@10":values[1],"HitRate@10":values[2],"NDCG@10":values[3]})
        row = {"protocol":protocol,"seed":seed,"model":name,"catalog_coverage":len(exposed)/len(ids)}
        row.update(dict(zip(["Precision@10","Recall@10","HitRate@10","NDCG@10"],np.mean(measurements,axis=0).tolist())))
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(per_user), info


def export_evaluation(movies, ratings, output, seeds=(42,7,2026), factors=24, epochs=10):
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    tables, users, protocols = [], [], []
    for protocol, seed in [("random",s) for s in seeds] + [("temporal",42)]:
        print(f"Evaluating {protocol} / seed {seed}",flush=True)
        table, per_user, info = evaluate_split(movies,ratings,protocol,seed,factors,epochs)
        tables.append(table);users.append(per_user);protocols.append(info)
    metrics = pd.concat(tables,ignore_index=True)
    user_metrics = pd.concat(users,ignore_index=True)
    metrics.to_csv(output/"evaluation_results.csv",index=False,float_format="%.8f")
    user_metrics.to_csv(output/"per_user_metrics.csv.gz",index=False,compression={"method":"gzip","mtime":0})
    summary = metrics.groupby(["protocol","model"]).agg(
        RMSE=("RMSE","mean"),RMSE_std=("RMSE","std"),
        Precision_at_10=("Precision@10","mean"),Recall_at_10=("Recall@10","mean"),
        NDCG_at_10=("NDCG@10","mean"),catalog_coverage=("catalog_coverage","mean")).reset_index()
    summary.to_csv(output/"evaluation_summary.csv",index=False,float_format="%.8f")
    confidence = []
    rng=np.random.default_rng(123)
    for protocol,seed in [("random",s) for s in seeds] + [("temporal",42)]:
        group=user_metrics[(user_metrics.protocol==protocol)&(user_metrics.seed==seed)]
        wide=group.pivot(index="userId",columns="model",values="Precision@10")
        delta=(wide["Validated blend"]-wide["Popularity"]).to_numpy()
        # Chunked bootstrap bounds RAM; confidence is conditional on each trained split.
        samples=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(1000)])
        lo,hi=np.quantile(samples,[.025,.975])
        confidence.append({"protocol":protocol,"seed":seed,"comparison":"Validated blend minus Popularity Precision@10",
                           "mean_difference":float(delta.mean()),"95pct_user_bootstrap":[float(lo),float(hi)]})
    report={"schema_version":1,"versions":{"python":platform.python_version(),"numpy":np.__version__,
            "pandas":pd.__version__,"sklearn":sklearn.__version__},
            "protocols":protocols,"bootstrap":confidence,
            "limitations":["MovieLens is a historical catalog; no recent releases.",
                           "Unobserved ratings are not confirmed dislikes.",
                           "Movie similarity requires human judgments; user ranking metrics do not establish semantic similarity.",
                           "Fixed ALS defaults; ranking blend selected on an inner holdout, not the test."]}
    (output/"evaluation_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return metrics
