# Evaluation data

The original MovieLens 1M ratings are not committed. Download from the official GroupLens source:

```bash
python scripts/download_data.py
```

This writes movies.dat and ratings.dat to data/ml-1m/. Alternatively, download and extract the official archive manually.

The serving app uses derived aggregate statistics and top-neighbor scores in app_data/. These contain no raw user IDs or viewing histories.

MovieLens 1M is a historical dataset with 1,000,209 ratings, 6,040 users, and 3,883 films released through 2000. Review the official dataset README for provenance and usage conditions: https://files.grouplens.org/datasets/movielens/ml-1m-README.txt
