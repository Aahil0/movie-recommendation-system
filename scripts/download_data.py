"""Download the official historical MovieLens dataset into an ignored data folder."""
import argparse
from io import BytesIO
from pathlib import Path
import zipfile
import requests

URL="https://files.grouplens.org/datasets/movielens/ml-1m.zip"

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--directory",type=Path,default=Path("data"))
    args=parser.parse_args()
    response=requests.get(URL,timeout=60)
    response.raise_for_status()
    archive=zipfile.ZipFile(BytesIO(response.content))
    # Only fixed official dataset members are extracted.
    for filename in ["movies.dat","ratings.dat","README"]:
        member=f"ml-1m/{filename}"
        destination=args.directory/"ml-1m"/filename
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(archive.read(member))
    print(f"Official MovieLens data ready in {args.directory/'ml-1m'}")

if __name__=="__main__":main()
