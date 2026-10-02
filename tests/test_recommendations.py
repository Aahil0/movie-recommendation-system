import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from recommendation import (ContentFeatures, RecommendationEngine, export_app_data,
                            fit_item_neighbors, movie_statistics)
from rating_models import BiasModel, ObservedALS
from evaluation import ranking_metrics, split_ratings, ValidatedRanker
from scripts.enrich_metadata import select_match, split_title


class Fixture(unittest.TestCase):
    def setUp(self):
        self.movies=pd.DataFrame({"movieId":[1,2,3,4,5],"title":["One (2000)","Two (2000)","Three (2000)","Four (2000)","Five (2000)"],
                                  "genres":["Drama","Drama","Drama|Sci-Fi","Sci-Fi","Film-Noir"]})
        self.ratings=pd.DataFrame([(u,i,4 if i!=4 else 2,100+u) for u in range(1,7) for i in range(1,5)],
                                  columns=["userId","movieId","rating","timestamp"])
        self.stats=movie_statistics(self.movies,self.ratings)


class RecommendationTests(Fixture):

    def test_source_excluded_even_when_identical_vectors_tie(self):
        engine=RecommendationEngine(self.movies,self.stats)
        for mid in self.movies.movieId:
            result=engine.recommend(mid,k=20)
            self.assertNotIn(mid,set(result.movieId))
            self.assertEqual(len(result),len(set(result.movieId)))

    def test_row_order_cannot_change_rank(self):
        a=RecommendationEngine(self.movies,self.stats).recommend(1)
        b=RecommendationEngine(self.movies.sample(frac=1,random_state=8),self.stats.sample(frac=1,random_state=9)).recommend(1)
        self.assertEqual(a.movieId.tolist(),b.movieId.tolist())

    def test_collaborative_order_is_preserved_and_affects_hybrid(self):
        neighbors={1:[[3,.9,20],[2,.1,5]]}
        engine=RecommendationEngine(self.movies,self.stats,neighbors)
        collab=engine.recommend(1,mode="collaborative")
        self.assertEqual(collab.movieId.tolist(),[3,2])
        hybrid=engine.recommend(1,mode="hybrid",diversity=0)
        self.assertEqual(hybrid.movieId.iloc[0],3)
        content=engine.recommend(1,mode="content",diversity=0)
        self.assertEqual(content.movieId.iloc[0],2)

    def test_atomic_genres_and_metadata_change_relevance(self):
        features=ContentFeatures(self.movies)
        self.assertIn("Sci-Fi",features.encoder.classes_)
        self.assertIn("Film-Noir",features.encoder.classes_)
        self.assertNotIn("Sci",features.encoder.classes_)
        metadata=pd.DataFrame({"movieId":[1,2,3],"overview":["prison friendship escape","royal palace monarch","prison friendship escape"],
                               "cast":["","", ""],"keywords":["","",""],"director":["","",""]})
        engine=RecommendationEngine(self.movies,metadata=metadata)
        scores,_=engine.features.scores(0)
        self.assertGreater(scores[2],scores[1])
        self.assertTrue(np.isfinite(scores).all())
        genres_only=engine.recommend(1,mode="genres",diversity=0)
        self.assertEqual(genres_only.movieId.iloc[0],2)
        self.assertNotIn("story",genres_only.reason.iloc[0])

    def test_artifact_integrity_and_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            export_app_data(self.movies,self.ratings,directory)
            engine=RecommendationEngine.from_directory(directory)
            self.assertEqual(engine.movies.movieId.tolist(),self.movies.movieId.tolist())
            path=Path(directory)/"movie_stats.csv"
            path.write_text(path.read_text()+"\n",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"checksum"):
                RecommendationEngine.from_directory(directory)
        with tempfile.TemporaryDirectory() as directory:
            self.movies.to_csv(Path(directory)/"movies_app.csv",index=False)
            self.assertEqual(RecommendationEngine.from_directory(directory).artifact_status,"Genre fallback")

    def test_invalid_ids_counts_and_ambiguous_titles(self):
        engine=RecommendationEngine(self.movies)
        with self.assertRaises(ValueError):engine.recommend(999)
        with self.assertRaises(ValueError):engine.recommend(1,k=0)
        with self.assertRaises(ValueError):engine.find_movie_id("missing")
        duplicate=self.movies.copy();duplicate.loc[1,"title"]=duplicate.loc[0,"title"]
        with self.assertRaisesRegex(ValueError,"ambiguous"):
            RecommendationEngine(duplicate).find_movie_id("One (2000)")

    def test_train_only_neighbors_do_not_use_held_out_preferences(self):
        # User 7 likes only films 1 and 4 in the hypothetical held-out interactions.
        neighbors=fit_item_neighbors(self.movies,self.ratings,min_support=1)
        self.assertFalse(any(row[0]==4 for row in neighbors[1]))
        held_out=pd.DataFrame([(7,1,5,200),(7,4,5,200)],columns=self.ratings.columns)
        changed=fit_item_neighbors(self.movies,pd.concat([self.ratings,held_out]),min_support=1)
        self.assertTrue(any(row[0]==4 for row in changed[1]))


class EvaluationTests(Fixture):
    def test_global_time_split_keeps_equal_timestamps_together(self):
        data=self.ratings.copy()
        data["timestamp"]=[10]*12+[20]*6+[30]*6
        train,test,_=split_ratings(data,"temporal",fraction=.5)
        self.assertLess(train.timestamp.max(),test.timestamp.min())
        self.assertFalse(set(train.timestamp)&set(test.timestamp))
        self.assertEqual(len(train)+len(test),len(data))

    def test_cold_start_predictions_are_finite_and_bounded(self):
        for model in [BiasModel(),ObservedALS(factors=2,epochs=2)]:
            model.fit(self.ratings)
            values=model.predict(np.array([1,999,1,999]),np.array([1,1,999,999]))
            self.assertTrue(np.isfinite(values).all())
            self.assertTrue(((values>=1)&(values<=5)).all())
            self.assertAlmostEqual(values[-1],model.global_mean)

    def test_ranking_metrics_known_fixture(self):
        result=ranking_metrics(np.array([.9,.8,.7]),np.array([1,2,3]),{1,3},k=2)
        self.assertEqual(result[:3],(.5,.5,1.0))
        self.assertAlmostEqual(result[3],1/(1+1/np.log2(3)))

    def test_metadata_matching_never_picks_ambiguous_title_year(self):
        self.assertEqual(split_title("Matrix, The (1999)"),("The Matrix",1999))
        exact={"id":1,"title":"The Matrix","release_date":"1999-03-31"}
        self.assertEqual(select_match("Matrix, The (1999)",[exact])["id"],1)
        self.assertIsNone(select_match("Matrix, The (1999)",[exact,dict(exact,id=2)]))
        self.assertIsNone(select_match("Matrix, The (1999)",[dict(exact,release_date="2000-01-01")]))

    def test_cold_user_ranking_uses_training_popularity(self):
        train=self.ratings[self.ratings.movieId!=3]
        model=BiasModel().fit(train)
        ranker=ValidatedRanker(model,train,self.movies,.75)
        ids=self.movies.movieId.to_numpy()
        scores=ranker.score_user(999,ids)
        self.assertEqual(scores[self.movies.movieId.tolist().index(3)],0)
        self.assertGreater(scores[0],scores[2])

    def test_invalid_training_data_is_rejected(self):
        duplicate=pd.concat([self.ratings,self.ratings.iloc[:1]])
        with self.assertRaises(ValueError):ObservedALS().fit(duplicate)
        invalid=self.ratings.copy();invalid.loc[0,"rating"]=9
        with self.assertRaises(ValueError):BiasModel().fit(invalid)


if __name__=="__main__":
    unittest.main()
