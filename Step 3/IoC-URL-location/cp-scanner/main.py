from Confs.Configs import Configs
from pipeline import load_urls_to_process, process_urls, save_results
import datetime, time

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load URLs to process.')
    parser.add_argument('--conf', dest='conf_fname', nargs='?', action='store',
                        const=sum, default="/path/to/conf/config.yaml",
                        help='Pipeline configuration file . (default: %(default)s)')
    args = parser.parse_args()



    config = Configs(config_file_path=args.conf_fname,debug_folder=False)

    print("LOADING urls ...")
    load_urls_to_process(config)

    print("PROCESSING urls ...")
    process_urls(config, str(datetime.datetime.today()))

    print("SAVING RESULTS urls ...")
    save_results(config)

    time.sleep (5*60)
    print("LOADING urls ...")
    load_urls_to_process(config, use_proxy=True)

    print("PROCESSING urls ...")
    process_urls(config, str(datetime.datetime.today()), use_proxy=True)

    print("SAVING RESULTS urls ...")
    save_results(config, use_proxy=True)
