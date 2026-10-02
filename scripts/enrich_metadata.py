"""Optional TMDB enrichment; unmatched or ambiguous films are never guessed."""
import argparse
from difflib import SequenceMatcher
import os
from pathlib import Path
import re
import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def split_title(title):
    match=re.fullmatch(r"(.*)\s+\((\d{4})\)",title)
    if not match:
        raise ValueError(f"Title has no release year: {title}")
    name=re.sub(r"\s*\([^)]*\)", "",match.group(1)).strip()
    article=re.fullmatch(r"(.*), (The|An|A)",name)
    if article: name=f"{article.group(2)} {article.group(1)}"
    return name,int(match.group(2))


def select_match(title, candidates):
    name,year=split_title(title)
    normalize=lambda value:re.sub(r"[^\w]+"," ",value.casefold()).strip()
    expected=normalize(name)
    matches=[item for item in candidates if str(item.get("release_date",""))[:4]==str(year)
             and any(normalize(item.get(field,""))==expected for field in ["title","original_title"])]
    return matches[0] if len(matches)==1 else None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--catalog",type=Path,default=Path("app_data/movies_app.csv"))
    parser.add_argument("--output",type=Path,default=Path("app_data/movie_metadata.csv"))
    parser.add_argument("--limit",type=int,default=None)
    parser.add_argument("--mapping",type=Path,help="Optional manually checked movieId,tmdbId CSV")
    args=parser.parse_args()
    token=os.environ.get("TMDB_READ_TOKEN")
    if not token:
        raise SystemExit("Set TMDB_READ_TOKEN locally to your API read-access token; do not commit it.")
    session=requests.Session()
    session.headers["Authorization"]=f"Bearer {token}"
    session.mount("https://",HTTPAdapter(max_retries=Retry(total=3,backoff_factor=1,status_forcelist=[429,500,502,503,504],respect_retry_after_header=True)))
    def get(endpoint,params=None):
        response=session.get(f"https://api.themoviedb.org/3/{endpoint}",params=params,timeout=30)
        if response.status_code!=200:
            raise RuntimeError(f"TMDB returned HTTP {response.status_code}; no credentials logged")
        return response.json()
    movies=pd.read_csv(args.catalog)
    existing=pd.read_csv(args.output).fillna("").to_dict("records") if args.output.exists() else []
    done={int(row["movieId"]) for row in existing}
    mappings=pd.read_csv(args.mapping).set_index("movieId").tmdbId.to_dict() if args.mapping else {}
    skipped=[]
    for _,row in movies.head(args.limit or len(movies)).iterrows():
        mid=int(row.movieId)
        if mid in done: continue
        query,year=split_title(row.title)
        if mid in mappings:
            tmdb_id=int(mappings[mid]);matched_by="manually checked mapping"
        else:
            candidates=get("search/movie",{"query":query,"primary_release_year":year,"include_adult":"false"}).get("results",[])
            match=select_match(row.title,candidates)
            if match is None:
                skipped.append({"movieId":mid,"title":row.title,"reason":"No unique exact-title/year match"})
                continue
            tmdb_id=int(match["id"]);matched_by="unique exact title and release year"
        details=get(f"movie/{tmdb_id}",{"append_to_response":"credits,keywords"})
        cast=details.get("credits",{}).get("cast",[])[:5]
        crew=details.get("credits",{}).get("crew",[])
        existing.append({"movieId":mid,"tmdbId":tmdb_id,"overview":details.get("overview",""),
                         "keywords":"|".join(item["name"] for item in details.get("keywords",{}).get("keywords",[])),
                         "cast":"|".join(person["name"] for person in cast),
                         "director":"|".join(person["name"] for person in crew if person.get("job")=="Director"),
                         "source":"TMDB","matched_by":matched_by})
        args.output.parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(existing).sort_values("movieId").to_csv(args.output,index=False)
        time.sleep(.15)
    pd.DataFrame(skipped,columns=["movieId","title","reason"]).to_csv(args.output.with_name("metadata_unmatched.csv"),index=False)
    print(f"Metadata saved for {len(existing)} films; {len(skipped)} skipped for manual review.")


if __name__=="__main__":
    main()
