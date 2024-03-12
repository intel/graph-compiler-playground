import argparse
import numpy as np
import pandas as pd

from dl_bench.report import BenchmarkDb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path',
        default="results.csv",
        help="Path to results.csv",
    )
    # Backend
    parser.add_argument(
        "--host",
        default=None,
        help="Name of the host machine",
    )
    # Reporting
    parser.add_argument("-t", "--tag", default=None, help="Tag to mark this result in DB")
    parser.add_argument(
        "-u",
        "--url",
        default=None,
        help="Database url in sqlalchemy format, like sqlite://./database.db'",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.path)
    if args.host is not None:
        df['host'] = args.host

    if args.tag is not None:
        df['tag'] = args.tag

    db = args.url and BenchmarkDb(args.url)

    for i, row in df.drop(['id', 'date'], axis=1).iterrows():
        if db is None:
            print(row, end='\n\n')
        else:
            db.report(**row)

if __name__ == '__main__':
    main()
