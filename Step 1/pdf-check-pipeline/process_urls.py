from argparse import ArgumentParser
from datetime import datetime

from pipeline import process_urls

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("date", help="Execution date")

    args = parser.parse_args()

    date = args.date

    if not date:
        raise Exception("Execution date missing")

    print(f"ANALYZING DATE: {date}")
    date = str(datetime.strptime(str(date), '%Y-%m-%d')).split(" ")[0]

    process_urls(date)
