import os
import pathlib
from argparse import ArgumentParser
from datetime import datetime

from Utilities.Confs.Configs import Configs
from pipeline import send_notification

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

    send_notification(configs, date)
