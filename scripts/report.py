"""Charts and exports always come from computed evaluation/model variables."""
from pathlib import Path
import argparse
import json
import os
import sys
import subprocess

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
# Keep plotting caches inside a selected, writable report directory.
import numpy as np
import pandas as pd
from evaluation import load_movielens, ValidatedRanker
from rating_models import ObservedALS, LegacySVD
from recommendation import RecommendationEngine
from threadpoolctl import threadpool_limits


def generate_report(data_dir, root=ROOT):
    root=Path(root)
    os.environ.setdefault("MPLCONFIGDIR",str(root/".cache"/"matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.hashsalt":"frame","font.family":"DejaVu Sans","axes.spines.top":False,
                         "axes.spines.right":False,"figure.facecolor":"#f7f4ee","axes.facecolor":"#f7f4ee"})
    movies,ratings=load_movielens(data_dir)
    results=root/"results";images=root/"images"
    images.mkdir(parents=True,exist_ok=True)
    metrics=pd.read_csv(results/"evaluation_results.csv")
    base=metrics[(metrics.protocol=="random")&(metrics.seed==42)]
    base[["model","RMSE"]].dropna().rename(columns={"model":"Model"}).to_csv(results/"model_results.csv",index=False,float_format="%.8f")
    base[["model","Precision@10","Recall@10","NDCG@10"]].dropna().melt(
        id_vars="model",var_name="Metric",value_name="Score").rename(columns={"model":"Model"}).to_csv(results/"top_k_results.csv",index=False,float_format="%.8f")
    figure,ax=plt.subplots(figsize=(9,4.5))
    table=metrics[metrics.RMSE.notna()].groupby(["model","protocol"]).RMSE.mean().unstack()
    table.plot.bar(ax=ax,color=["#8b343b","#8f9a82"],rot=25)
    ax.set_title("Rating prediction: random vs strict global-time holdout")
    ax.set_ylabel("RMSE (lower is better)")
    figure.tight_layout()
    figure.savefig(images/"model_comparison.svg",metadata={"Date":None});plt.close(figure)
    figure,ax=plt.subplots(figsize=(7,4))
    ratings.rating.value_counts().sort_index().plot.bar(ax=ax,color="#8b343b",rot=0)
    ax.set_title("MovieLens 1M rating distribution");ax.set_xlabel("Rating");ax.set_ylabel("Count")
    figure.tight_layout();figure.savefig(images/"rating_distribution.svg",metadata={"Date":None});plt.close(figure)
    with threadpool_limits(limits=2):
        report=json.loads((results/"evaluation_report.json").read_text(encoding="utf-8"))
        selected=report["protocols"][0]["validated_ranker"]
        model=(ObservedALS() if selected["model"]=="Observed ALS" else LegacySVD()).fit(ratings)
        ranker=ValidatedRanker(model,ratings,movies,selected["weight"])
    seen=set(ratings.loc[ratings.userId==100,"movieId"])
    candidates=movies[~movies.movieId.isin(seen)].copy()
    candidates["predicted_rating"]=model.score_user(100,candidates.movieId.to_numpy())
    candidates["ranking_score"]=ranker.score_user(100,candidates.movieId.to_numpy())
    candidates=candidates.sort_values(["ranking_score","movieId"],ascending=[False,True]).head(10)
    candidates.to_csv(results/"user_100_recommendations.csv",index=False,float_format="%.8f")
    figure,ax=plt.subplots(figsize=(10,5))
    ax.barh(candidates.title.iloc[::-1],candidates.predicted_rating.iloc[::-1],color="#8b343b")
    ax.set_xlim(0,5);ax.set_title("User 100: validated ranking blend, full-data serving fit")
    ax.set_xlabel("Predicted rating (bounded to 1–5)")
    figure.tight_layout();figure.savefig(images/"recommendation_chart.svg",metadata={"Date":None});plt.close(figure)
    engine=RecommendationEngine.from_directory(root/"app_data")
    samples=[]
    exposure=np.zeros(len(engine.movies),dtype=int)
    errors=[]
    for mid in engine.movies.movieId:
        rec=engine.recommend(int(mid),10)
        if len(rec)!=10 or mid in set(rec.movieId) or rec.movieId.duplicated().any():errors.append(int(mid))
        for target in rec.movieId:exposure[engine.positions[int(target)]]+=1
    for title in ["Toy Story (1995)","Matrix, The (1999)","Shawshank Redemption, The (1994)","Godfather, The (1972)","Alien (1979)"]:
        rec=engine.recommend(engine.find_movie_id(title),10)
        rec.insert(0,"query",title);rec.insert(1,"rank",range(1,len(rec)+1))
        samples.append(rec)
    samples=pd.concat(samples,ignore_index=True)
    samples.to_csv(results/"similar_film_examples.csv",index=False,float_format="%.7f")
    judgments=samples[["query","rank","movieId","title","reason"]].copy()
    judgments["human_relevance_0_to_2"]=""
    judgments["reviewer_notes"]=""
    judgments.to_csv(results/"human_relevance_template.csv",index=False)
    audit={"queries":len(engine.movies),"k":10,"self_count_or_duplicate_failures":errors,
           "unique_recommended_movies":int((exposure>0).sum()),
           "catalog_coverage":float((exposure>0).mean()),
           "human_relevance_status":"Not measured; template provided for blind human judgments.",
           "metadata_coverage":float(engine.features.has_details.mean())}
    (results/"app_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    assert not errors,audit
    subprocess.run([sys.executable,str(Path(__file__).with_name("write_docs.py")),str(root.resolve())],check=True)
    print("Reports and full-catalog recommendation audit generated.",audit,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,default=ROOT)
    args=parser.parse_args()
    generate_report(args.data_dir,args.output_dir)
