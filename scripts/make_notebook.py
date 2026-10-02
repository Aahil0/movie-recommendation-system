"""Generate the executable notebook from shared project modules."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
cells=[]
def markdown(text):
    cells.append({"cell_type":"markdown","metadata":{},"id":f"cell-{len(cells):02d}","source":text.splitlines(keepends=True)})
def code(text):
    cells.append({"cell_type":"code","metadata":{},"id":f"cell-{len(cells):02d}","source":text.splitlines(keepends=True),
                  "execution_count":None,"outputs":[]})

markdown("""# FRAME: movie discovery and reproducible evaluation

The app recommends related films with genres and aggregate co-like patterns. The rating model and personalized ranking are evaluated separately; rating RMSE does not measure story similarity.

Run this notebook from the repository root or notebooks/. First install requirements-dev.txt and download official MovieLens 1M with scripts/download_data.py. Set MOVIELENS_DIR to use an existing dataset.

Default execution reuses the checked-in multi-seed evaluation and regenerates examples/charts. Set RUN_FULL_EVALUATION=True to retrain every benchmark. No cell writes requirements, deletes datasets, or hard-codes accuracy scores.""")
code("""from pathlib import Path
import os
import sys
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path.cwd() if (Path.cwd() / "recommendation.py").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
from evaluation import load_movielens, export_evaluation
from recommendation import RecommendationEngine, catalog_fingerprint, export_app_data
from scripts.report import generate_report
DATA_DIR = Path(os.environ.get("MOVIELENS_DIR", str(ROOT / "data" / "ml-1m")))
RUN_FULL_EVALUATION = False
""")
code("""movies, ratings = load_movielens(DATA_DIR)
print(f"{len(movies):,} films; {len(ratings):,} ratings; {ratings.userId.nunique():,} users")
print("Ratings by value:", ratings.rating.value_counts().sort_index().to_dict())
movies.head()
""")
markdown("""## Film-to-film recommendations

Serving artifacts are fit on all available ratings. Offline evaluation fits models only on training interactions. Artifacts contain aggregate neighbor IDs/scores/support and movie statistics, not raw user records.

Genres are atomic TF-IDF features. Optional verified metadata adds overview, keywords, cast and director features. Co-like cosine scores are shrunk by support; hybrid ranking mixes both signals and a small Bayesian quality prior, then reduces repetitive genre matches. Self-exclusion uses movie IDs.""")
code("""if not (ROOT / "app_data" / "model_manifest.json").exists():
    with threadpool_limits(limits=2):
        export_app_data(movies, ratings, ROOT / "app_data")
engine = RecommendationEngine.from_directory(ROOT / "app_data")
assert catalog_fingerprint(movies) == catalog_fingerprint(engine.movies)
print(engine.artifact_status)
print("Optional metadata coverage:", float(engine.features.has_details.mean()))
""")
code("""def recommend_movies(title, num_recommendations=10):
    return engine.recommend(engine.find_movie_id(title), num_recommendations, mode="content")

def collaborative_recommend(movie_id, num_recommendations=10):
    return engine.recommend(movie_id, num_recommendations, mode="collaborative", diversity=0)

def hybrid_recommend(title, movie_id, num_recommendations=10):
    if engine.find_movie_id(title) != movie_id:
        raise ValueError("Title and movie ID refer to different films")
    return engine.recommend(movie_id, num_recommendations, mode="hybrid")

for mode in ["content", "collaborative", "hybrid"]:
    result = engine.recommend(2571, 10, mode=mode)
    assert 2571 not in set(result.movieId)
    assert not result.movieId.duplicated().any()
    print(mode, result.title.tolist())
""")
code("""hybrid_recommend("Matrix, The (1999)", 2571)
""")
code("""hybrid_recommend("Shawshank Redemption, The (1994)", 318)
""")
markdown("""## Rating prediction and personalized ranking

Observed-entry ridge ALS with biases predicts explicit ratings and clips predictions to 1–5. A corrected legacy SVD remains a comparator. Bias, movie mean, global/user mean, popularity and Bayesian quality baselines make improvement claims testable.

Personalized ranking uses a model/popularity blend chosen on an inner training-only holdout by NDCG@10. Cold users use popularity. Outer test data never selects model weights. The app's film-to-film hybrid is a separate task and does not predict your personal rating.

Benchmarks use random per-user splits with seeds 42, 7 and 2026, plus a strict global timestamp cutoff. Equal timestamp groups cannot straddle the cutoff. All test ratings are scored with explicit cold-start fallbacks; ranking excludes training-rated items only.""")
code("""if RUN_FULL_EVALUATION:
    with threadpool_limits(limits=2):
        export_evaluation(movies, ratings, ROOT / "results")
metrics = pd.read_csv(ROOT / "results" / "evaluation_results.csv")
summary = pd.read_csv(ROOT / "results" / "evaluation_summary.csv")
summary
""")
code("""report = json.loads((ROOT / "results" / "evaluation_report.json").read_text())
for protocol in report["protocols"]:
    print(protocol["protocol"], protocol.get("seed", protocol.get("cutoff")),
          "cold-user ratings:", protocol["cold_user_ratings"],
          "cold-movie ratings:", protocol["cold_movie_ratings"],
          "ranking choice:", protocol["validated_ranker"]["model"],
          "relevance weight:", protocol["validated_ranker"]["weight"])
report["bootstrap"]
""")
markdown("""## Computed exports and figures

Charts and CSV outputs are generated from computed metrics and the fitted serving model. User 100's example is a full-data serving fit and excludes all of that user's recorded ratings. It is an example, not an evaluation test case.""")
code("""generate_report(DATA_DIR, ROOT)
pd.read_csv(ROOT / "results" / "user_100_recommendations.csv")
""")
code("""from IPython.display import SVG, display
display(SVG(filename=str(ROOT / "images" / "model_comparison.svg")))
display(SVG(filename=str(ROOT / "images" / "rating_distribution.svg")))
display(SVG(filename=str(ROOT / "images" / "recommendation_chart.svg")))
""")
markdown("""## Limits and human validation

Unobserved films are not confirmed dislikes. MovieLens 1M ends at 2000. Collaborative similarity is evidence of shared viewers, not proof of similar stories.

The current repository has no populated plot/cast metadata. Run the optional TMDB importer with your own locally configured read token; it rejects ambiguous title/year matches. Until then, the app clearly reports genre/viewer signals.

Use results/human_relevance_template.csv for blind 0–2 relevance judgments before claiming semantic film-similarity accuracy. Rank metrics and catalog coverage measure different aspects of performance.""")
code("""audit = json.loads((ROOT / "results" / "app_audit.json").read_text())
assert not audit["self_count_or_duplicate_failures"]
print("Full-catalog audit:", audit)
""")
notebook={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                                  "language_info":{"name":"python","version":"3.12"}},
          "nbformat":4,"nbformat_minor":5}
path=ROOT/"notebooks"/"Movie_Recommendation_System_MovieLens_1M.ipynb"
path.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+"\n",encoding="utf-8")
print(f"Generated {len(cells)} notebook cells.")

if __name__=="__main__":pass
