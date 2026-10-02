"""Keep documented metric tables synchronized with actual generated results."""
import json
from pathlib import Path
import sys
import pandas as pd

ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parents[1]
summary=pd.read_csv(ROOT/"results/evaluation_summary.csv")
audit=json.loads((ROOT/"results/app_audit.json").read_text())
report=json.loads((ROOT/"results/evaluation_report.json").read_text())
rows=["| Protocol | Model | RMSE ↓ | Precision@10 ↑ | NDCG@10 ↑ |",
      "|---|---|---:|---:|---:|"]
for _,row in summary[summary.model.isin(["Bias","Legacy SVD","Observed ALS","Validated blend","Popularity"])].iterrows():
    rmse=f"{row.RMSE:.4f}" if pd.notna(row.RMSE) else "—"
    rows.append(f"| {row.protocol} | {row.model} | {rmse} | {row.Precision_at_10:.2%} | {row.NDCG_at_10:.4f} |")
table="\n".join(rows)
readme=f"""# FRAME · Movie discovery

A Streamlit movie discovery app, backed by reproducible MovieLens experiments.

**Existing app:** https://movie-recommendation-system-uvth6veeegbjuu4kzv4c8k.streamlit.app/

## What the application actually does

Start with a film you like. **Similar films** combines atomic genre TF-IDF, aggregate co-like patterns, and a small Bayesian quality signal, then reduces repetitive genre matches. **Genre matches** uses genres with a rating-quality tie-breaker. Ranking preserves scores and excludes the source by movie ID. Results stay visible between interactions.

The historical catalog contains 3,883 films released through 2000. The app does not claim to know your personal rating or compare plots without metadata. Optional verified overview, keyword, cast and director metadata can enrich content features.

The serving bundle contains aggregate movie statistics and compact top-neighbor scores, with checksums and a data fingerprint. It contains no raw user IDs or histories and requires no API key or training download at startup. Genre fallback remains available if derived artifacts are missing.

## Run the app

Use Python 3.12:

```bash
python -m venv .venv
# Activate .venv using the command for your shell.
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit Community Cloud should deploy repository branch main, file app.py. Pinning dependencies makes local and cloud runs consistent.

## Rebuild and evaluate

```bash
python scripts/download_data.py
python scripts/train.py --data-dir data/ml-1m --evaluate
python -m unittest discover -s tests -v
```

Training regenerates app data, three random-seed benchmarks, a strict global-time benchmark, computed CSV outputs, figures, User 100's serving example, and a full-catalog app audit. Without --evaluate it rebuilds only aggregate serving artifacts.

Run the notebook after installing requirements-dev.txt. Its default Run all loads the checked-in multi-seed results and regenerates examples/charts; set RUN_FULL_EVALUATION=True to retrain every benchmark. It does not overwrite dependencies or delete data.

## Current computed results

{table}

Random results are means across seeds 42, 7 and 2026. Temporal uses one global timestamp cutoff, so equal-time groups cannot leak across the boundary. All test ratings are included with explicit cold-user/item fallbacks. Ranking excludes training-rated films, uses the full catalog, and evaluates users with at least one held-out rating of 4 or 5.

Observed-entry ALS improves rating prediction, but rating RMSE alone is insufficient for recommendation ranking. The **validated blend** selects a model/popularity balance on an inner training-only holdout by NDCG@10; the outer test never selects weights. Cold users receive popularity-based ranking.

These protocols differ from the original notebook's dropped-cold-item/per-user timestamp checks; temporal values are not directly comparable with those earlier numbers.

App audit: {audit['queries']:,} inputs at K={audit['k']}; no self/count/duplicate failures; {audit['catalog_coverage']:.2%} catalog coverage ({audit['unique_recommended_movies']:,} films). The earlier genre-only implementation exposed 29.41%. Greater coverage does not prove semantic relevance.

Full per-split results, uncertainty intervals, coverage and model-selection settings are in results/evaluation_results.csv and results/evaluation_report.json. A blank human relevance template is provided for blind judgments of actual film similarity; no human-judgment score is fabricated.

## Optional story and credit metadata

MovieLens does not contain plots, cast or directors. Obtain a TMDB API read-access token and set TMDB_READ_TOKEN in your local environment; never commit it. Then run:

```bash
python scripts/enrich_metadata.py
```

The importer accepts only a unique exact title/year match, rejects ambiguous matches, resumes existing output, retries transient API failures, and records provenance. For skipped films, supply a manually checked CSV with movieId,tmdbId using --mapping. The app reads movie_metadata.csv automatically after restart. No credentials are required for the existing genre/viewer app.

TMDB metadata must be used with appropriate attribution and terms: https://developer.themoviedb.org/docs/faq

## Layout and files

- app.py: Streamlit presentation; theme in .streamlit/config.toml.
- recommendation.py: shared content, collaborative and film-to-film hybrid ranking.
- rating_models.py: bias baseline, legacy comparator, observed-entry regularized ALS.
- evaluation.py: splits, train-only validation, rating/ranking metrics, cold starts and bootstrap.
- scripts/: data download, metadata enrichment, training and computed reports.
- notebooks/: executable analysis using those same modules.
- tests/: recommendation, leakage, cold-start, artifact and Streamlit regressions.
- app_data/: checksummed aggregate serving bundle.
- results/: computed evaluation results and relevance-review template.

## Figures

![Rating prediction comparison](images/model_comparison.svg)
![Rating distribution](images/rating_distribution.svg)
![User 100 serving example](images/recommendation_chart.svg)

## Limitations

This is a historical educational recommender. Unobserved films are not confirmed dislikes; implicit ranking metrics depend on the candidate policy. The strict-time split includes cold users, which makes its task different from per-user random validation. Story similarity still needs populated metadata and human evaluation. The app's film-to-film recommendations and the notebook's personalized rating model are separate tasks.

Dataset: [GroupLens MovieLens 1M](https://grouplens.org/datasets/movielens/1m/). Author: Aahil.
"""
(ROOT/"README.md").write_text(readme,encoding="utf-8")
details=f"""# Evaluation protocol and results

This file is generated from computed result tables. It is not a hard-coded score source.

{table}

## Method

- Random per-user 80/20 holdouts: seeds 42, 7, 2026.
- Strict global timestamp: ratings at or before the cutoff train the model; strictly later ratings are test. Equal-time groups stay together.
- Rating error includes every test rating; cold IDs use training-only global/bias fallback. No silent dropping.
- Candidate universe is the full 3,883-film catalog minus training-rated items.
- Relevant test films have ratings ≥4. Users without relevant held-out films are reported separately.
- Inner validation chooses the ranking model and blend weight using NDCG@10 on up to 1,000 sampled validation users. Outer test data never chooses these values.
- RMSE and MAE evaluate bounded 1–5 predictions. Precision, recall, hit rate, NDCG and catalog coverage evaluate ranking.
- Paired user bootstrap intervals are conditional on each trained split; multi-seed results report split variation.
- App artifacts use all ratings for serving, separately from train-only benchmark fits.

## Provenance

Versions: {report['versions']}.

See evaluation_report.json for the exact cutoff, training/test sizes, cold-start counts, all validation candidates and bootstrap intervals.

## Film similarity review

Offline user ranking is not a semantic similarity label. human_relevance_template.csv contains five query films with ten actual app results each. Have reviewers grade each result 0=unrelated, 1=somewhat related, 2=strongly related, without showing model scores. Preserve the blank template and save judgments separately. Compare blinded old/new rankings before claiming story-level improvement.

Current app metadata coverage: {audit['metadata_coverage']:.2%}. Additional plot/cast features are supported but not populated without external credentials or a supplied metadata file.
"""
(ROOT/"results/EVALUATION.md").write_text(details,encoding="utf-8")
print("Documentation synchronized with generated metrics.")
