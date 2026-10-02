"""Regenerate serving data and optionally the complete evaluation."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluation import export_evaluation, load_movielens
from recommendation import export_app_data
from threadpoolctl import threadpool_limits


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--evaluate",action="store_true")
    parser.add_argument("--seeds",type=int,nargs="+",default=[42,7,2026])
    args=parser.parse_args()
    movies,ratings=load_movielens(args.data_dir)
    with threadpool_limits(limits=2):
        if args.evaluate:
            export_evaluation(movies,ratings,args.output_dir/"results",seeds=args.seeds)
        export_app_data(movies,ratings,args.output_dir/"app_data")
        if args.evaluate:
            from scripts.report import generate_report
            generate_report(args.data_dir,args.output_dir)
    print("Generated app artifacts; no raw user data is included in the serving bundle.")


if __name__=="__main__":
    main()
