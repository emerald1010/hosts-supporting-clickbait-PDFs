import multiprocessing
import os
import random
import time
import traceback
from datetime import datetime

import requests
from psycopg2._psycopg import ProgrammingError, InterfaceError, OperationalError, InternalError
from random_user_agent.params import SoftwareName, OperatingSystem
from random_user_agent.user_agent import UserAgent


import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from Datalake.Phishing.Phishing import PhishingDataLake
from Datalake.Redis.RedisDatalake import RedisDatalake

import log

__LOGGER__ = log.getlogger('slims_scanner', level=log.INFO,
                           filename=os.path.join('/path/to/', 'logs', 'slims_scanner.log'))

MAX_WORKERS = 15
MAX_RETRIES = 0
WAIT_BEFORE_RETRY_SECONDS = 60 * 5

FLAGS_SLIMS = [
    '<meta name="description" content="',
    '<meta name="generator" content="'
]

FLAGS_ELEARN = [
    '<title>e-learning madrasah',
    'hak cipta kementerian Agama republik indonesia'
]

# User agent parameters
software_names = [SoftwareName.CHROME.value, SoftwareName.FIREFOX.value]
operating_systems = [OperatingSystem.WINDOWS.value, OperatingSystem.MACOS.value]



def load_urls_to_process(config):
    """
    Retrieves form the database all the urls that have to be processed and load them into the "2-process" queue.
    :param phishing_dataset: instance of the phishing dataset management class
    :return:
    """

    __LOGGER__.info("Starting pipeline")

    # Test internet connection
    try:
        _ = requests.get('https://google.com', timeout=20)
    except (requests.ConnectionError, requests.Timeout):
        raise Exception("NO INTERNET CONNECTION")

    # Instantiate the phishing Datatlake management class
    phishing_datalake = PhishingDataLake.prepare_datalake(config, user='user')

    # Instantiate the redis Datalake management class
    redis_datalake = RedisDatalake.prepare_datalake(config)

    # Create queue object to store the urls to process
    queue_urls_to_process = redis_datalake.create_queue('domains_slims')

    # Create queue object to store the results
    queue_results = redis_datalake.create_queue('slims_results')

    # make sure that the queues containing the urls to process and the results are initially is empty
    queue_urls_to_process.clean()
    queue_results.clean()

    urls_to_process = phishing_datalake.get_urls_to_process()
    __LOGGER__.info(f"Loaded urls: {len(urls_to_process)}")
    # shuffle list of urls to reduce the risk of resource-collisions between threads
    random.shuffle(urls_to_process)

    # Add the urls to process into the redis queue
    for url in urls_to_process:
        # schedule a job to download the file and retrieve the HTTP header
        queue_urls_to_process.push((url, ))



def process_urls(config):

    # Instantiate the redis Datalake management class
    redis_datalake = RedisDatalake.prepare_datalake(config)

    # Create queue for managing processign errors
    processing_errors = redis_datalake.create_queue('slims_processing_errors')
    processing_errors.clean()

    queue_urls_to_process = redis_datalake.create_queue('domains_slims')
    if queue_urls_to_process.count() == 0:
        __LOGGER__.info('No URL to process, returning!')
        return

    N_THREADS = min(queue_urls_to_process.count(), MAX_WORKERS)

    __LOGGER__.debug("Processing...")
    __LOGGER__.info(f"Using {N_THREADS} threads")
    with multiprocessing.get_context("spawn").Pool(N_THREADS) as Pool:
        # todo: find an alternative to starmap
        Pool.map(process_urls_worker, [config] * N_THREADS)

    n_exceptions = processing_errors.count()
    if processing_errors.count() > 0:
        __LOGGER__.warning(f"Could not process {processing_errors.count()} due to the following exception")

        while processing_errors.count() > 0:
            url, e, t = processing_errors.pop()

            __LOGGER__.info(f"SKIPPING {url} for the exception: {e}, traceback:")
            __LOGGER__.debug(t)

        raise Exception(f"{n_exceptions} exceptions while processing urls")


def process_urls_worker(config):
    # Instantiate the redis Datalake management class
    redis_datalake = RedisDatalake.prepare_datalake(config)

    # Create queue object to store the urls to process
    queue_urls_to_process = redis_datalake.create_queue('domains_slims')

    # Create queue object to store the urls to process
    queue_results = redis_datalake.create_queue('slims_results')

    # Create variable to count how many consecutive times the worker has
    # found the urls queue empty
    consecutive_empty = 0

    queue_processing_errors = redis_datalake.create_queue('slims_processing_errors')

    user_agent_rotator = UserAgent(software_names=software_names, operating_systems=operating_systems)

    while consecutive_empty < 3:

        while queue_urls_to_process.count() > 0:

            consecutive_empty = 0

            url_to_visit, = queue_urls_to_process.pop()

            if not url_to_visit:
                continue

            result = None

            # pick a user agent to utilize
            user_agent = user_agent_rotator.get_random_user_agent()
            user_agent += ' Scientific research study. Conctact yourname@yourinstitution.com'

            try:
                response, error = process_request(url_to_visit, user_agent=user_agent)

                if not error:
                    if 300 > response.status_code >= 200:

                        matches_slims = any([
                            flag for flag in FLAGS_SLIMS if flag in response.text.lower() and "slims" in response.text.lower()
                        ])
                        if matches_slims:
                            result = (url_to_visit, 'slims')
                            __LOGGER__.debug(f'Result {result} for url {url_to_visit}')

                            # add the results to the queue of the results to process
                            queue_results.push((result, 0))


                        matches_elearn = any([
                            flag for flag in FLAGS_ELEARN if flag in response.text.lower()
                                                             and "e-learning madrasah" in response.text.lower()
                        ])
                        if matches_elearn:

                            ckfinder_location = '__statics/ckdrive/ckfinder.html'
                            if url_to_visit.endswith('/'):
                                vuln_endpoint = url_to_visit + ckfinder_location
                            else:
                                vuln_endpoint = url_to_visit + '/' + ckfinder_location
                            response_vuln_endpoint, error_vuln = process_request(vuln_endpoint, user_agent=user_agent)

                            if not error_vuln and 300 > response_vuln_endpoint.status_code >= 200:
                                result = (url_to_visit, 'e-learning madrasah')
                                __LOGGER__.debug(f'Result {result} for url {url_to_visit}')
                                # add the results to the queue of the results to process
                                queue_results.push((result, 0))

                    else:
                        __LOGGER__.info(f'Response code {response.status_code} for url {url_to_visit}')
                else:
                    __LOGGER__.warning(f'Error {error} with request {url_to_visit}')

            except Exception as e:
                queue_processing_errors.push((url_to_visit, e, traceback.format_exc()))
                continue

        time.sleep(2 ** consecutive_empty)
        consecutive_empty += 1

    __LOGGER__.info("Worker terminated.")


def process_request(url, user_agent=None):
    response = None
    error = None

    session = requests.Session()
    try:
        # attach specified user agent to the session
        if user_agent is not None:
            session.headers.update({'User-Agent': user_agent})

        response = session.get(url, verify=False, timeout=(5, 15))

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


def save_results(config):
    # read the config file

    redis_datalake = RedisDatalake.prepare_datalake(config)

    queue_results = redis_datalake.create_queue('slims_results')
    if queue_results.count() == 0:
        __LOGGER__.info('No URL to save, returning!')
        return

    __LOGGER__.info(f"Saving {queue_results.count()} results")

    n_threads = 10

    saving_errors = redis_datalake.create_queue('slims_saving_errors')
    saving_errors.clean()

    with multiprocessing.Pool(n_threads) as Pool:
        Pool.map(save_results_worker,
                     [config] * n_threads)

    if saving_errors.count() > 0:

        while saving_errors.count() > 0:
            domain, error, traceback_str = saving_errors.pop()
            __LOGGER__.warning(f"could not process {domain} due to the following error: {error}")
            __LOGGER__.info(traceback_str)

        raise Exception("One ore more samples have not been inserted correctly")
    else:
        __LOGGER__.info(f"ALL OK! All samples inserted successfully")


def save_results_worker(config):
    # Instantiate the redis Datalake management class
    redis_datalake = RedisDatalake.prepare_datalake(config)

    queue_urls_to_process = redis_datalake.create_queue('domains_slims')

    queue_results = redis_datalake.create_queue('slims_results')

    # Instantiate the phishing Datatlake management class
    phishing_datalake = PhishingDataLake.prepare_datalake(config, user='user', autocommit=False)

    # got a programming ex
    saving_errors = redis_datalake.create_queue('slims_saving_errors')

    while True:

        # If all the urls have been processed and all the results have been saved, exit
        if queue_results.count() == 0 and queue_urls_to_process.count() == 0:
            break
        elif queue_results.count() == 0 and queue_urls_to_process.count() > 0:
            __LOGGER__.warning('There are still URLs in the `queue_urls_to_process`!')
            time.sleep(5)

        results = None
        retry_attempts = 0

        # get one sample from the queue
        out = queue_results.pop()

        if not out:
            continue

        try:
            results, retry_attempts = out
            __LOGGER__.debug(results)
            phishing_datalake.update(results)
            phishing_datalake.commit()

        except ProgrammingError as e:
            __LOGGER__.error(e)

            # If a programming error is catched, there is a problem with the queries
            # Do not reprocess the sample
            phishing_datalake.rollback()
            saving_errors.push((results[0], e, traceback.format_exc()))
            continue
        except (InterfaceError, OperationalError, InternalError) as e:

            # If the problem does not depend on the queries but on some other factor of the db
            # try to insert again the samples up to 2 times before logging it as failed
            phishing_datalake.regenerate_connection()

            if retry_attempts > 1:
                saving_errors.push((results[0], e, traceback.format_exc()))
            else:
                queue_results.push((results, retry_attempts + 1))

            time.sleep(1)
            continue

        except Exception as e:
            # If the exception has not been generated by the db
            # Do not retry and log the sample as failed
            phishing_datalake.rollback()

            if results:
                saving_errors.push((results[0], e, traceback.format_exc()))
            time.sleep(1)
            continue

    __LOGGER__.info(f"ALL OK! All samples processed successfully")

