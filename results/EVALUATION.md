# Evaluation protocol and results

This file is generated from computed result tables. It is not a hard-coded score source.

| Protocol | Model | RMSE ↓ | Precision@10 ↑ | NDCG@10 ↑ |
|---|---|---:|---:|---:|
| random | Bias | 0.9102 | 5.31% | 0.0589 |
| random | Legacy SVD | 0.9659 | 17.91% | 0.2166 |
| random | Observed ALS | 0.8636 | 10.41% | 0.1237 |
| random | Popularity | — | 14.81% | 0.1744 |
| random | Validated blend | 0.9659 | 18.64% | 0.2251 |
| temporal | Bias | 0.9499 | 12.21% | 0.1156 |
| temporal | Legacy SVD | 1.0165 | 18.98% | 0.1958 |
| temporal | Observed ALS | 0.9382 | 15.76% | 0.1633 |
| temporal | Popularity | — | 26.05% | 0.2795 |
| temporal | Validated blend | 0.9382 | 27.96% | 0.3001 |

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

Versions: {'python': '3.12.14', 'numpy': '2.5.3', 'pandas': '3.0.6', 'sklearn': '1.7.2'}.

See evaluation_report.json for the exact cutoff, training/test sizes, cold-start counts, all validation candidates and bootstrap intervals.

## Film similarity review

Offline user ranking is not a semantic similarity label. human_relevance_template.csv contains five query films with ten actual app results each. Have reviewers grade each result 0=unrelated, 1=somewhat related, 2=strongly related, without showing model scores. Preserve the blank template and save judgments separately. Compare blinded old/new rankings before claiming story-level improvement.

Current app metadata coverage: 0.00%. Additional plot/cast features are supported but not populated without external credentials or a supplied metadata file.
