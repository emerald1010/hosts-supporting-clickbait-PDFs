import hashlib
import json
import multiprocessing
import os
import sys
import pathlib
import random
import shutil
import subprocess
import time
import traceback
from collections import defaultdict
from datetime import datetime, timedelta, date
from pathlib import Path
import pickle
import re

import requests
from psycopg2._psycopg import ProgrammingError, InterfaceError, OperationalError, InternalError
import urllib3
from random_user_agent.params import SoftwareName, OperatingSystem
from random_user_agent.user_agent import UserAgent

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from Utilities.Confs.Configs import Configs
from Datalake.Phishing.Phishing import PhishingDataLake
from Datalake.Redis.RedisDatalake import RedisDatalake
from Utilities.Files.IO import filename_path_decode
from Utilities.Files.Screenshots import PDFPPM, produce_thumbnail

from Utilities.checkvpn import CheckVPN

MAX_WORKERS = 75
MAX_RETRIES = 2
WAIT_BEFORE_RETRY_SECONDS = 60*5

# DUMP LOGS
DUMP_LOGS = False

# User agent parameters
software_names = [SoftwareName.CHROME.value, SoftwareName.FIREFOX.value]
operating_systems = [OperatingSystem.WINDOWS.value, OperatingSystem.MACOS.value]

# Available proxy endpoints
proxies = [dict(http='http://10.0.3.1:1080', https='http://10.0.3.1:1080'),
           dict(http='http://10.0.3.1:1081', https='http://10.0.3.1:1081'),
           dict(http='http://10.0.3.1:1082', https='http://10.0.3.1:1082'),
           #dict(http='http://10.0.3.1:1083', https='http://10.0.3.1:1083'),
           #dict(http='http://10.0.3.1:1084', https='http://10.0.3.1:1084'),
           dict(http='http://10.0.3.1:1085', https='http://10.0.3.1:1085'),
           dict(http='http://10.0.3.1:1086', https='http://10.0.3.1:1086'),
           dict(http='http://10.0.3.1:1087', https='http://10.0.3.1:1087')]


def load_urls_to_process():
    """
    Retrieves form the database all the urls that have to be processed and load them into the "2-process" queue.
    :param phishing_dataset: instance of the phishing dataset management class
    :return:
    """

    print("Starting pipeline")

    # Test internet connection
    try:
        request = requests.get('https://google.com', timeout=20)
    except (requests.ConnectionError, requests.Timeout) as exception:
        raise Exception("NO INTERNET CONNECTION")

    # read the config file
    config = Configs(os.path.join(pathlib.Path(__file__).parent.resolve(), "config.yaml"), debug_folder=False)

    # Instantiate the phishing Datatlake management class
    phishing_datalake = PhishingDataLake.prepare_datalake(config)

    # Instantiate the redis Datalake management class
    redis_datalake = RedisDatalake.prepare_datalake(config)

    # Create queue object to store the urls to process
    queue_urls_to_process = redis_datalake.create_queue('urls_2_process')

    # Create queue object to store the results
    queue_results = redis_datalake.create_queue('urls_results')

    # make sure that the queues containing the urls to process and the results are initially is empty
    queue_urls_to_process.clean()
    queue_results.clean()

    # Get urls to process
    urls_to_process = phishing_datalake.get_urls_to_process()

    # shuffle list of urls to reduce the risk of resource-collisions between threads
    random.shuffle(urls_to_process)

    first_visit_count = defaultdict(lambda: 0)
    # Add the urls to process into the redis queue
    for url, first_visit in urls_to_process:
        # schedule a job to download the file and retrieve the HTTP header
        queue_urls_to_process.push((url, first_visit, config, 0, time.time()))

        first_visit_count[first_visit] += 1

    print(f"Loaded urls: {queue_urls_to_process.count()}")
    print(f"First visits: {first_visit_count[True]}")
    print(f" Others: {first_visit_count[False]}")


def test_vpns():
    check_vpn = CheckVPN()

    try:
        check_vpn.check_one_vpn(0)

        time.sleep(60)

        for i, proxy in enumerate(proxies):
            check_vpn.check_one_vpn(i, proxy)
            time.sleep(60)
    except Exception as e:
        print(e)
        sys.exit(-1)


def process_urls(date):
    # read the config file
    config = Configs(os.path.join(pathlib.Path(__file__).parent.resolve(), "config.yaml"), debug_folder=False)

    target_endpoint = config["global"]["providers"]["phishing"]["file_storage"]

    tmp_endpoint_root = os.path.join(config["global"]["providers"]["tmp"]["tmp_file_storage"], date)

    tmp_endpoint_samples = os.path.join(tmp_endpoint_root, 'samples')

    # Be sure that the tmp endpoint is located in the tmp_folder
    assert (str(tmp_endpoint_samples).startswith('/tmp/'))

    if not Path(tmp_endpoint_samples).exists():
        print(f"TMP endpoint does not exists, created it: at: {tmp_endpoint_samples}")
        os.makedirs(tmp_endpoint_samples)

    # make sure that the tmp endpoint is a valid folder
    assert (Path(tmp_endpoint_samples).is_dir())

    # Ensure that the tmp endpoint is empty
    if len(os.listdir(tmp_endpoint_samples)) > 0:
        print(f"WARNING THE TMP FOLDER ALREADY EXISTS")

    # Instantiate the redis Datalake management class
    redis_datalake = RedisDatalake.prepare_datalake(config)

    # Create queue for managing processign errors
    processing_errors = redis_datalake.create_queue('processing_errors')
    queue_logs = redis_datalake.create_queue('url_2_process_logs')

    queue_logs.clean()
    processing_errors.clean()

    queue_urls_to_process = redis_datalake.create_queue('urls_2_process')

    N_THREADS = min(queue_urls_to_process.count(), MAX_WORKERS)

    print("Processing...")
    print(f"Using {N_THREADS} threads")

    with multiprocessing.get_context("spawn").Pool(N_THREADS) as Pool:
        # todo: find an alternative to starmap
        Pool.starmap(process_urls_worker,
                     [(config, tmp_endpoint_samples, target_endpoint, i) for i in range(N_THREADS)])

    print('Processing ended!')

    n_exceptions = processing_errors.count()
    if n_exceptions > 0:
        print(f"Could not process {n_exceptions} due to the following exception")

        while processing_errors.count() > 0:
            url, e, t = processing_errors.pop()

            print(f"SKIPPING {url} for the exception: {e}, traceback:")
            print(t)

        raise Exception(f"{n_exceptions} exceptions while processing urls")
    else:
        print("No exceptions during the execution of the pipeline!")


def process_urls_worker(config, tmp_endpoint, target_endpoint, worker_id):
    # Instantiate the redis Datalake management class
    redis_datalake = RedisDatalake.prepare_datalake(config)

    # Create queue object to store the urls to process
    queue_urls_to_process = redis_datalake.create_queue('urls_2_process')

    # Create queue object to store the urls to process
    queue_results = redis_datalake.create_queue('urls_results')

    # Create queue object to store the urls to process
    queue_processing_errors = redis_datalake.create_queue('processing_errors')

    # Create queue object to store the urls to process
    queue_logs = redis_datalake.create_queue('url_2_process_logs')

    user_agent_rotator = UserAgent(software_names=software_names, operating_systems=operating_systems)

    while True:

        logs = {}

        logs["worker_id"] = worker_id
        logs["start_processing"] = time.time()

        if not queue_urls_to_process.ping() or not queue_results.ping() or not queue_processing_errors.ping():
            # if one redis connection is dead, reestablish all the queues
            del redis_datalake
            del queue_urls_to_process
            del queue_results
            del queue_processing_errors
            del queue_logs

            redis_datalake = RedisDatalake.prepare_datalake(config)

            queue_urls_to_process = redis_datalake.create_queue('urls_2_process')
            queue_results = redis_datalake.create_queue('urls_results')
            queue_processing_errors = redis_datalake.create_queue('processing_errors')
            queue_logs = redis_datalake.create_queue('url_2_process_logs')

        element = queue_urls_to_process.pop()

        # if the queue is empty kill the thread
        if not element:
            break

        url, first_visit, config, execution_count, scheduling_time = element

        time_since_last_execution = time.time() - scheduling_time

        time_to_sleep = 0

        # check if enough time has passed from the last try
        if (time_since_last_execution < WAIT_BEFORE_RETRY_SECONDS) and execution_count > 0:
            time_to_sleep = WAIT_BEFORE_RETRY_SECONDS - time_since_last_execution
            print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                  f"  WAITING FOR: {time_to_sleep} ELEMENTS_IN_QUEUE: {queue_urls_to_process.count()}")
            # if not, wait for the minimum amount and then process the sample
            time.sleep(time_to_sleep)

        result = None
        proxy = None

        # if this is the last time we are going to test the url before logging a failure use a proxy
        if execution_count == MAX_RETRIES:
            proxy = random.choice(proxies)

        # pick a user agent to utilize
        user_agent = user_agent_rotator.get_random_user_agent()
        user_agent += ' Scientific research study. Conctact yourname@yourinstitution.com'

        logs["before_fetching_data"] = time.time()

        try:
            result = process_single_url(url, first_visit, tmp_endpoint, target_endpoint, proxy, user_agent)
        except Exception as e:
            queue_processing_errors.push((url, e, traceback.format_exc()))
            continue
        logs["after_fetching_data"] = time.time()

        url, filehash, header, status_code, content_type, error, amz, processing_time, internal_logs = result

        logs = {**logs, **internal_logs}

        # if there is an error and we can still retry
        if error and execution_count < MAX_RETRIES:
            # re-push the element in the retry queue
            logs["before_rescheduling_results"] = time.time()
            queue_urls_to_process.push((url, first_visit, config, execution_count + 1, time.time()))
            logs["after_rescheduling_results"] = time.time()
            queue_logs.push(logs)
            continue

        logs["before_pushing_results"] = time.time()

        # add the results to the queue of the results to process
        queue_results.push((result, 0))

        logs["after_pushing_results"] = time.time()

        queue_logs.push(logs)

        logs["end_processing"] = time.time()

    print(f"Worker terminated: {datetime.now()}", flush=True)


def process_request(url, header_only=False, proxy=None, user_agent=None):
    response = None
    error = None

    session = requests.Session()
    try:

        # use the selected proxy for this request
        if proxy is not None:
            session.proxies.update(proxy)

        # attach specified user agent to the session
        if user_agent is not None:
            session.headers.update({'User-Agent': user_agent})

        if header_only:
            response = session.head(url, verify=False, timeout=(5, 20))
        else:
            response = session.get(url, verify=False, timeout=(5, 20))

    except requests.exceptions.Timeout as e:
        error = 'Timeout error'
    except requests.exceptions.TooManyRedirects as e:
        error = 'Too many redirects'
    except requests.exceptions.HTTPError:
        error = 'HTTP error'
    except requests.exceptions.SSLError as e:
        error = 'SSLError'
        if 'Caused by SSLError' in error:
            detailed_msg = error.split('Caused by SSLError')[1].strip()
            error = detailed_msg
    except requests.exceptions.ConnectionError as e:
        error = 'Connection error'
    except Exception as e:
        error = str(e)
    finally:
        session.close()

    return response, error


def process_single_url(url, first_visit, tmp_endpoint, target_endpoint, proxy=None, user_agent=None):
    filehash = None
    error = None
    response = None
    amz = None

    logs = {}

    logs["before_processing_head_request"] = time.time()

    response, error = process_request(url, True, proxy=proxy, user_agent=user_agent)

    logs["url"] = url
    logs["after_processing_head_request"] = time.time()

    if not error:

        # check if the url still points to a correct resource
        if 300 > response.status_code >= 200 or response.headers.get('content-type') == "application/xml":

            if response.headers.get('content-type') == "application/pdf":

                # The url still points to a valid pdf
                if first_visit:

                    # If this is the first time we visit this url and it is the first time we find this file
                    # download the sample
                    logs["before_content_request"] = time.time()
                    response, error = process_request(url, False, proxy=proxy, user_agent=user_agent)
                    logs["after_content_request"] = time.time()

                    if not error:

                        if 300 > response.status_code >= 200:

                            if response.headers.get('content-type') == "application/pdf":

                                # compute the hash of the response that will be used as filename
                                filehash = str(hashlib.sha256(response.content).hexdigest())

                                # compute the path of the folder in the file endpoint
                                dataset_dest_path = os.path.join(filename_path_decode(filehash, target_endpoint),
                                                                 filehash)

                                # If the dataset does not already contain the file download it
                                if not Path(dataset_dest_path).exists():

                                    logs["pdf_save_time_start"] = time.time()

                                    # compute the path of the folder where we should save the sample temporarly
                                    tmp_dest_folder_path = tmp_endpoint

                                    # Compute the final path of the file
                                    tmp_dest_file_path = os.path.join(tmp_dest_folder_path, filehash)

                                    # Make sure that no file already exists at this path
                                    if not Path(tmp_dest_file_path).exists():
                                        # the sample does not exist, save it
                                        with open(tmp_dest_file_path, 'wb+') as file:
                                            file.write(response.content)

                                    logs["pdf_save_time_end"] = time.time()
                            else:
                                error = "Inconsistent content_type"
                        else:
                            error = "Inconsistent status code"

            else:

                error = "Bad content type"

                if response.headers.get('content-type') == "application/xml":

                    response, error = process_request(url, False, proxy=proxy, user_agent=user_agent)

                    if not error and response.headers.get('content-type') != "application/xml":
                        error = "Inconsistent content type"

                    elif not error:
                        if "amazonaws" in str(url).lower() or "bucket" in str(response.text.lower()) or (
                                "hostid" in str(response.text.lower()) and "requestid" in str(response.text.lower())):
                            amz = str(response.text)

                        error = "Bad content type"
        else:
            error = "Bad status code"

    header = None
    status_code = None
    content_type = None
    if response is not None:
        header = dict(response.headers)
        status_code = response.status_code
        content_type = response.headers.get('Content-Type')

        content_type = None if content_type == "" else content_type
        header = header if header != 'null' else None

    # print(url, filehash, header, status_code, content_type, error, amz, datetime.now())
    logs["error"] = error

    return url, filehash, header, status_code, content_type, error, amz, datetime.now(), logs


def create_backup(date):
    config = Configs(os.path.join(pathlib.Path(__file__).parent.resolve(), "config.yaml"), debug_folder=False)

    tmp_endpoint = os.path.join(config["global"]["providers"]["tmp"]["tmp_file_storage"], date, 'samples')
    backup_path = os.path.join(config["global"]["providers"]["phishing"]["daily_backups"], date)

    shutil.make_archive(backup_path, 'zip', tmp_endpoint)


def save_results(date):
    # read the config file
    config = Configs(os.path.join(pathlib.Path(__file__).parent.resolve(), "config.yaml"), debug_folder=False)

    target_endpoint = config["global"]["providers"]["phishing"]["file_storage"]

    assert (Path(target_endpoint).exists() and Path(target_endpoint).is_dir())

    tmp_endpoint_root = os.path.join(config["global"]["providers"]["tmp"]["tmp_file_storage"], date)

    tmp_endpoint_samples = os.path.join(tmp_endpoint_root, 'samples')

    assert (Path(tmp_endpoint_samples).exists() and Path(tmp_endpoint_samples).is_dir())

    tmp_endpoint_screenshots = os.path.join(tmp_endpoint_root, 'screenshots')

    if not Path(tmp_endpoint_screenshots).exists():
        os.makedirs(tmp_endpoint_screenshots)

    redis_datalake = RedisDatalake.prepare_datalake(config)

    queue_results = redis_datalake.create_queue('urls_results')

    print(f"Saving {queue_results.count()} results")

    n_threads = 25

    saving_errors = redis_datalake.create_queue('saving_errors')
    saving_errors.clean()

    with multiprocessing.Pool(n_threads) as Pool:
        Pool.starmap(save_results_worker,
                     [(config, tmp_endpoint_samples, tmp_endpoint_screenshots, target_endpoint, date)] * n_threads)

    if saving_errors.count() > 0:

        while saving_errors.count() > 0:
            url, filehash, error, traceback_str = saving_errors.pop()
            print(f"could not process {url}, {filehash} due to the following error: {error}")
            print(traceback_str)

        raise Exception("One ore more samples have not been inserted correctly")
    else:
        print(f"ALL OK! All samples inserted successfully")

    print("Creating dump of the logs")

    queue_logs = redis_datalake.create_queue('url_2_process_logs')

    if DUMP_LOGS:

        logs_list = []

        while True:

            sample = queue_logs.pop()

            if not sample:
                break

            logs_list.append(sample)

        logs_dump_path = os.path.join(config["global"]["providers"]["tmp"]["logs"], date + ".pkl")

        if not Path(config["global"]["providers"]["tmp"]["logs"]).exists():
            os.makedirs(config["global"]["providers"]["tmp"]["logs"])

        assert not Path(logs_dump_path).exists()

        with open(logs_dump_path, 'wb') as f:
            pickle.dump(logs_list, f)


def save_results_worker(config, tmp_endpoint, tmp_endpoint_screenshots, target_endpoint, date):
    # Instantiate the redis Datalake management class
    redis_datalake = RedisDatalake.prepare_datalake(config)

    queue_results = redis_datalake.create_queue('urls_results')

    # Instantiate the phishing Datatlake management class
    phishing_datalake = PhishingDataLake.prepare_datalake(config, autocommit=False)

    # got a programming ex
    saving_errors = redis_datalake.create_queue('saving_errors')

    while True:

        # If all the urls have been processed and all the results have been saved, exit
        if queue_results.count() == 0:
            break

        results = None
        retry_attempts = 0

        # get one sample from the queue
        out = queue_results.pop()

        if not out:
            continue

        try:
            results, retry_attempts = out

            save_url_result(results, phishing_datalake, date)
            phishing_datalake.commit()

            url, filehash, header, status_code, content_type, error, amz, datetime_request, _ = results

            # The db insert has been processed correctly
            # Copy the pdf from the tmp to the destination folder

            if filehash:
                # compute the theoretical path of the sample if it was downloaded
                tmp_path = os.path.join(tmp_endpoint, filehash)

                # compute the theoretical path of the sample in the file endpoint of the dataset
                final_dest_folder_path = filename_path_decode(filehash, target_endpoint)

                if Path(tmp_path).exists() and Path(tmp_path).is_file():
                    # if the sample is saved in the temporary endpoint

                    # if no folder structure has yet been created for this sample
                    if not Path(final_dest_folder_path).exists():
                        # create the sample specific folder structure
                        os.makedirs(final_dest_folder_path, exist_ok=True)

                    try: # there are often case of same filehash seen first on the same day --> raises error for concurrency
                        # move the file from the tmp folder to the final storage endpoint
                        shutil.move(tmp_path, final_dest_folder_path)
                    except shutil.Error as e:
                        print(f'File {filehash} for url {url} already exists!')
                        print(e)

                # Extract the screenshot of the sample
                screenshot_destination_folder_path = filename_path_decode(filehash,
                                                                          phishing_datalake.screenshots_files_endpoint)
                screenshot_destination_path = os.path.join(screenshot_destination_folder_path, filehash + '.png')

                thumbnail_destination_folder_path = filename_path_decode(filehash,
                                                                         phishing_datalake.thumbnail_screenshot_entrypoint)
                thumbnail_destination_path = os.path.join(thumbnail_destination_folder_path, filehash + '.png')

                if not Path(screenshot_destination_path).exists():

                    ppm = PDFPPM(os.path.join(final_dest_folder_path, filehash), tmp_endpoint_screenshots, filehash)

                    if hasattr(ppm, 'images') and len(ppm.images) > 0:
                        ppm_img_path = ppm.images[0]

                        if not Path(screenshot_destination_folder_path).exists():
                            os.makedirs(screenshot_destination_folder_path)

                        if not Path(screenshot_destination_path).exists():
                            try:
                                shutil.move(ppm_img_path, screenshot_destination_path)
                            except shutil.Error as e:
                                print(f'Screenshot for file {filehash} already exists!')
                                print(e)

                if not Path(thumbnail_destination_path).exists() and Path(screenshot_destination_path).exists():
                    produce_thumbnail(screenshot_destination_path, thumbnail_destination_path)
        except ProgrammingError as e:
            print(e)
            # If a programming error is catched, there is a problem with the queries
            # Do not reprocess the sample
            phishing_datalake.rollback()
            saving_errors.push((results[0], results[1], e, traceback.format_exc()))
            continue
        except (InterfaceError, OperationalError, InternalError) as e:
            print(e)
            # If the problem does not depend on the queries but on some other factor of the db
            # try to insert again the samples up to 2 times before logging it as failed
            phishing_datalake.regenerate_connection()

            if retry_attempts > 1:
                saving_errors.push((results[0], results[1], e, traceback.format_exc()))
            else:
                queue_results.push((results, retry_attempts + 1))

            time.sleep(1)
            continue
        except Exception as e:
            print(e)
            # If the exception has not been generated by the db
            # Do not retry and log the sample as failed
            phishing_datalake.rollback()

            if results:
                saving_errors.push((results[0], results[1], e, traceback.format_exc()))

            time.sleep(1)
            continue


def save_url_result(sample_results, phishing_datalake, date):
    url, filehash, header, status_code, content_type, error, amz, datetime_request, _ = sample_results

    # check if an error is present
    if not error:
        # handle online resource

        if filehash:
            # if the filehash has been computed:
            # - it is the first time that we visit the url
            # - it pointed to a valid PDF

            # insert and entry in the imported_sample table linking this sample with the provider 'FromUrl'
            # if it does not already exist

            phishing_datalake.add_imported_sample(filehash, 'application/pdf', datetime_request, 'FromUrl',
                                                  datetime_request, None)

        # Update link reference in the db
        phishing_datalake.save_online_url_results(url, header, datetime_request, filehash, status_code, content_type)
    else:

        # handle offline resource
        phishing_datalake.save_offline_url_results(url, header, datetime_request, status_code, content_type,
                                                   error, amz)
