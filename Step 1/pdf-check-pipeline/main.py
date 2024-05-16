from Confs.Configs import Configs
from pipeline import load_urls_to_process, process_urls, save_results, test_vpns
import datetime
if __name__ == "__main__":
    config = Configs('config.yaml',debug_folder=False)

    print("LOADING urls ...")
    load_urls_to_process()

    print('TESTING vpns ...')
    test_vpns()

    print("PROCESSING urls ...")
    process_urls(str(datetime.datetime.today().date()))

    print("SAVING RESULTS urls ...")
    save_results(str(datetime.datetime.today().date()))
