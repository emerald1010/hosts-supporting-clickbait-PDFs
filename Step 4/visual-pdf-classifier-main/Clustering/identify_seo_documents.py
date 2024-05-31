import os
import pathlib
import sys
from argparse import ArgumentParser
from datetime import datetime

from Data.Datalake.Phishing.Phishing import prepare_phishing_datalake
from Utilities.Confs.Configs import Configs
from pipeline import identify_seo_documents

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("date", help="Execution date")

    args = parser.parse_args()

    date = args.date

    if not date:
        raise Exception("Execution date missing")

    print(f"ANALYZING DATE: {date}")
    date = datetime.strptime(str(date), '%Y-%m-%d')

    # Load configs
    title = "Daily seo metric run"
    configs = Configs(os.path.join(pathlib.Path(__file__).parent.resolve(), "config.yaml"), title, debug_folder=False)

    # Load the Phishing datalake management class
    phishing_dataset = prepare_phishing_datalake(configs)

    identify_seo_documents(phishing_dataset, from_date=datetime(2020, 12, 16).date(), to_date=date)
