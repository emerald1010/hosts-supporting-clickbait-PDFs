import hashlib
import multiprocessing
import os
import random
import shutil
import subprocess
import time
import traceback
from datetime import datetime
import re

import requests
from psycopg2._psycopg import ProgrammingError, InterfaceError, OperationalError, InternalError
from random_user_agent.params import SoftwareName, OperatingSystem
from random_user_agent.user_agent import UserAgent

import pandas as pd
from urllib.parse import urlparse

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from Datalake.Phishing.Phishing import PhishingDataLake
from Datalake.Redis.RedisDatalake import RedisDatalake

from Utilities.CompromisedPlugins.Endpoints import get_endpoint_flag_per_plugin, run_chrome, get_path

import log

__LOGGER__ = log.getlogger('cp_scanner', level=log.INFO,
                           filename=os.path.join('/path/to/', 'logs', 'cp_scanning',
                                                 datetime.today().strftime("%Y-%m-%d") + ".log"))

MAX_WORKERS = 15
MAX_RETRIES = 0
WAIT_BEFORE_RETRY_SECONDS = 60 * 5

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


def load_urls_to_process(config, use_proxy=False):
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
    queue_urls_to_process = redis_datalake.create_queue('domains_cp')

    # Create queue object to store the results
    queue_results = redis_datalake.create_queue('cp_results')

    # make sure that the queues containing the urls to process and the results are initially is empty
    queue_urls_to_process.clean()
    queue_results.clean()

    # Get urls to process
    urls_to_process = phishing_datalake.get_urls_to_process(use_proxy)

    __LOGGER__.info(f"Loaded urls: {len(urls_to_process)}")
    # shuffle list of urls to reduce the risk of resource-collisions between threads
    random.shuffle(urls_to_process)

    plugin_endpoints_per_domain = []
    for d in urls_to_process:
        plugin_endpoints_flags = get_endpoint_flag_per_plugin(d)
        for plugin in plugin_endpoints_flags:
            plugin_endpoints_per_domain.append(plugin)

    # Add the urls to process into the redis queue
    for plugin_ep in plugin_endpoints_per_domain:
        # schedule a job to download the file and retrieve the HTTP header
        queue_urls_to_process.push((plugin_ep, config))


def load_urls_to_process_guessed_cp(config, use_proxy=False):
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
    queue_urls_to_process = redis_datalake.create_queue('domains_cp')

    # Create queue object to store the results
    queue_results = redis_datalake.create_queue('cp_results')

    # make sure that the queues containing the urls to process and the results are initially is empty
    queue_urls_to_process.clean()
    queue_results.clean()

    # Get urls to process
    urls_to_process = phishing_datalake.get_urls_to_process_guessed_cp(use_proxy)

    __LOGGER__.info(f"Loaded urls: {len(urls_to_process)}")
    # shuffle list of urls to reduce the risk of resource-collisions between threads
    random.shuffle(urls_to_process)

    plugin_endpoints_per_domain = []
    for d in urls_to_process:
        plugin_endpoints_flags = get_endpoint_flag_per_plugin(d)
        for plugin in plugin_endpoints_flags:
            plugin_endpoints_per_domain.append(plugin)

    # Add the urls to process into the redis queue
    for plugin_ep in plugin_endpoints_per_domain:
        # schedule a job to download the file and retrieve the HTTP header
        queue_urls_to_process.push((plugin_ep, config))



def process_urls(config, date, use_proxy=False):
    # read the config file

    target_endpoint = config["global"]["file_storage"]

    # Instantiate the redis Datalake management class
    redis_datalake = RedisDatalake.prepare_datalake(config)

    # Create queue for managing processign errors
    processing_errors = redis_datalake.create_queue('cp_processing_errors')
    processing_errors.clean()

    queue_urls_to_process = redis_datalake.create_queue('domains_cp')
    if queue_urls_to_process.count() == 0:
        __LOGGER__.info('No URL to process, returning!')
        return

    N_THREADS = min(queue_urls_to_process.count(), MAX_WORKERS)

    __LOGGER__.debug("Processing...")
    __LOGGER__.info(f"Using {N_THREADS} threads")
    with multiprocessing.get_context("spawn").Pool(N_THREADS) as Pool:
        # todo: find an alternative to starmap
        Pool.starmap(process_urls_worker, [(config, target_endpoint, use_proxy)] * N_THREADS)

    n_exceptions = processing_errors.count()
    if processing_errors.count() > 0:
        __LOGGER__.warning(f"Could not process {processing_errors.count()} due to the following exception")

        while processing_errors.count() > 0:
            url, e, t = processing_errors.pop()

            __LOGGER__.info(f"SKIPPING {url} for the exception: {e}, traceback:")
            __LOGGER__.debug(t)

        raise Exception(f"{n_exceptions} exceptions while processing urls")


def process_urls_worker(config, target_endpoint, use_proxy=False):
    # Instantiate the redis Datalake management class
    redis_datalake = RedisDatalake.prepare_datalake(config)

    # Create queue object to store the urls to process
    queue_urls_to_process = redis_datalake.create_queue('domains_cp')

    # Create queue object to store the urls to process
    queue_results = redis_datalake.create_queue('cp_results')

    # Create variable to count how many consecutive times the worker has
    # found the urls queue empty
    consecutive_empty = 0

    queue_processing_errors = redis_datalake.create_queue('cp_processing_errors')

    user_agent_rotator = UserAgent(software_names=software_names, operating_systems=operating_systems)

    while consecutive_empty < 3:

        while queue_urls_to_process.count() > 0:

            consecutive_empty = 0

            element = queue_urls_to_process.pop()

            if not element:
                continue

            plugin_endpoints, config = element

            result = None
            proxy = None

            # if this is the second time we scan those endpoints
            if use_proxy:
                proxy = random.choice(proxies)

            # pick a user agent to utilize
            user_agent = user_agent_rotator.get_random_user_agent()
            user_agent += ' Scientific research study. Conctact yourname@yourinstitution.com'
            
            try:
                result = process_batch_endpoints(plugin_endpoints, True, target_endpoint, proxy, user_agent)

                df = pd.DataFrame(result, columns=['plugin', 'endpoints', 'iocs', 'timestamp'])
                __LOGGER__.debug(df)
            except Exception as e:
                queue_processing_errors.push((plugin_endpoints, e, traceback.format_exc()))
                continue

            df['proxy'] = [use_proxy] * df.shape[0]

            tmp_endpoints = plugin_endpoints['endpoints'][0]
            domain = set([urlparse(x).scheme + '://' + urlparse(x).netloc for x in tmp_endpoints[1]]).pop()
            df['domain'] = [domain] * df.shape[0]

            # overwrite endpoints, which we passed back here to be able to extract `domain`
            df.loc[df.iocs.isna(), 'endpoints'] = None

            df['path'] = df.endpoints.apply(get_path)
            df.drop_duplicates(['domain', 'plugin', 'endpoints'], inplace=True)
            to_insert = df.to_dict(orient='records')

            # add the results to the queue of the results to process
            queue_results.push((to_insert, 0))

        time.sleep(2 ** consecutive_empty)
        consecutive_empty += 1

    __LOGGER__.info("Worker terminated.")


def process_request(url, header_only=False, proxies=None, user_agent=None):
    response = None
    error = None

    session = requests.Session()
    try:

        # use the selected proxy for this request
        if proxies is not None:
            session.proxies.update(proxies)

        # attach specified user agent to the session
        if user_agent is not None:
            session.headers.update({'User-Agent': user_agent})

        if header_only:

            response = session.head(url, verify=False, timeout=(5, 15))
        else:
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

def process_batch_endpoints (endpoint_urls, first_visit, target_endpoint, proxies=None, user_agent=None):
    # paths for metadata
    screenshots_endpoint = os.path.join(target_endpoint, 'cp_scanner', 'screenshots')
    if not os.path.exists(screenshots_endpoint):
        os.makedirs(screenshots_endpoint)
    ioc_endpoint = os.path.join(target_endpoint, 'cp_scanner', 'manual_analysis')
    if not os.path.exists(ioc_endpoint):
        os.makedirs(ioc_endpoint)

    plugin = endpoint_urls['plugin']
    flags_endpoints = endpoint_urls['endpoints']
    results = []

    for flags, endpoints in flags_endpoints:

        for endpoint_url in endpoints:
            response, error = process_request(endpoint_url, False, proxies=proxies, user_agent=user_agent)

            if error:
                continue

            else:
                if 300 > response.status_code >= 200:

                    # print(response.status_code)
                    matches = []
                    for flag in flags:
                        matches.append(find_flag(response.text, flag))

                    if any(matches):
                        results.append((plugin, endpoint_url, matches, datetime.now()))
                        break

                        # try:
                        #     browser_config = {
                        #         'user_agent': user_agent,
                        #         'proxy': proxies,
                        #         'timeout': 10
                        #     }
                        #     run_chrome(endpoint_url, os.path.join(screenshots_endpoint, endpoint_url.replace('/', '\\') + '.png')
                        #                , browser_config)
                        #
                        #     # everything went well -- screenshot was saved etc.
                        #     results.append((plugin, endpoint_url, matches, datetime.now()))
                        #     break
                        #
                        # except Exception as e:
                        #     __LOGGER__.exception(e)
                        #     continue
                    # else:
                    #     with open(os.path.join(ioc_endpoint, endpoint_url.replace('/', '\\')), 'w+') as f_out:
                    #         f_out.write(response.text)
            time.sleep(2)

    if len(results) == 0:
        try:
            any_endpoint = flags_endpoints[0][1][0]
            results.append((plugin, any_endpoint, None, datetime.now()))
        except Exception as e:
            __LOGGER__.warning(e)
            results.append((plugin, None, None, datetime.now()))


    return results

def find_flag(text, regex):
    try:
        return regex.search(text.replace('\\', '')).group(0)
    except Exception:
        return None


def save_results(config, use_proxy=False):
    # read the config file

    redis_datalake = RedisDatalake.prepare_datalake(config)

    queue_results = redis_datalake.create_queue('cp_results')
    if queue_results.count() == 0:
        __LOGGER__.info('No URL to save, returning!')
        return

    __LOGGER__.info(f"Saving {queue_results.count()} results")

    n_threads = 10

    saving_errors = redis_datalake.create_queue('cp_saving_errors')
    saving_errors.clean()

    with multiprocessing.Pool(n_threads) as Pool:
        Pool.starmap(save_results_worker,
                     [(config, use_proxy)] * n_threads)

    if saving_errors.count() > 0:

        while saving_errors.count() > 0:
            domain, error, traceback_str = saving_errors.pop()
            __LOGGER__.warning(f"could not process {domain} due to the following error: {error}")
            __LOGGER__.info(traceback_str)

        raise Exception("One ore more samples have not been inserted correctly")
    else:
        __LOGGER__.info(f"ALL OK! All samples inserted successfully")


def save_results_guessed_cps(config, use_proxy=False):
    # read the config file

    redis_datalake = RedisDatalake.prepare_datalake(config)

    queue_results = redis_datalake.create_queue('cp_results')
    if queue_results.count() == 0:
        __LOGGER__.info('No URL to save, returning!')
        return

    __LOGGER__.info(f"Saving {queue_results.count()} results")

    n_threads = 10

    saving_errors = redis_datalake.create_queue('cp_saving_errors')
    saving_errors.clean()

    with multiprocessing.Pool(n_threads) as Pool:
        Pool.starmap(save_results_worker_guessed_cps,
                     [(config, use_proxy)] * n_threads)

    if saving_errors.count() > 0:

        while saving_errors.count() > 0:
            domain, error, traceback_str = saving_errors.pop()
            __LOGGER__.warning(f"could not process {domain} due to the following error: {error}")
            __LOGGER__.info(traceback_str)

        raise Exception("One ore more samples have not been inserted correctly")
    else:
        __LOGGER__.info(f"ALL OK! All samples inserted successfully")



def save_results_worker(config, use_proxy=False):
    # Instantiate the redis Datalake management class
    redis_datalake = RedisDatalake.prepare_datalake(config)

    queue_urls_to_process = redis_datalake.create_queue('domains_cp')

    queue_results = redis_datalake.create_queue('cp_results')

    # Instantiate the phishing Datatlake management class
    phishing_datalake = PhishingDataLake.prepare_datalake(config, user='user', autocommit=False)

    # got a programming ex
    saving_errors = redis_datalake.create_queue('cp_saving_errors')

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

            if not use_proxy:
                __LOGGER__.debug(results)
                phishing_datalake.insert(results)
            else:
                phishing_datalake.update(results)
            phishing_datalake.commit()
        except ProgrammingError as e:
            __LOGGER__.error(e)

            # If a programming error is catched, there is a problem with the queries
            # Do not reprocess the sample
            phishing_datalake.rollback()
            saving_errors.push((results[0]['domain'], e, traceback.format_exc()))
            continue
        except (InterfaceError, OperationalError, InternalError) as e:

            # If the problem does not depend on the queries but on some other factor of the db
            # try to insert again the samples up to 2 times before logging it as failed
            phishing_datalake.regenerate_connection()

            if retry_attempts > 1:
                saving_errors.push((results[0]['domain'], e, traceback.format_exc()))
            else:
                queue_results.push((results, retry_attempts + 1))

            time.sleep(1)
            continue

        except Exception as e:
            # If the exception has not been generated by the db
            # Do not retry and log the sample as failed
            phishing_datalake.rollback()

            if results:
                saving_errors.push((results[0]['domain'], e, traceback.format_exc()))
            time.sleep(1)
            continue

    __LOGGER__.debug(f"ALL OK! All samples inserted successfully")


def save_results_worker_guessed_cps(config, use_proxy=False):
    # Instantiate the redis Datalake management class
    redis_datalake = RedisDatalake.prepare_datalake(config)

    queue_urls_to_process = redis_datalake.create_queue('domains_cp')

    queue_results = redis_datalake.create_queue('cp_results')

    # Instantiate the phishing Datatlake management class
    phishing_datalake = PhishingDataLake.prepare_datalake(config, user='postgres', autocommit=False)

    # got a programming ex
    saving_errors = redis_datalake.create_queue('cp_saving_errors')

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

            if not use_proxy:
                __LOGGER__.debug(results)
                phishing_datalake.insert_guessed_cps(results)
            else:
                phishing_datalake.update_guessed_cps(results)
            phishing_datalake.commit()
        except ProgrammingError as e:
            __LOGGER__.error(e)

            # If a programming error is catched, there is a problem with the queries
            # Do not reprocess the sample
            phishing_datalake.rollback()
            saving_errors.push((results[0]['domain'], e, traceback.format_exc()))
            continue
        except (InterfaceError, OperationalError, InternalError) as e:

            # If the problem does not depend on the queries but on some other factor of the db
            # try to insert again the samples up to 2 times before logging it as failed
            phishing_datalake.regenerate_connection()

            if retry_attempts > 1:
                saving_errors.push((results[0]['domain'], e, traceback.format_exc()))
            else:
                queue_results.push((results, retry_attempts + 1))

            time.sleep(1)
            continue

        except Exception as e:
            # If the exception has not been generated by the db
            # Do not retry and log the sample as failed
            phishing_datalake.rollback()

            if results:
                saving_errors.push((results[0]['domain'], e, traceback.format_exc()))
            time.sleep(1)
            continue

    __LOGGER__.debug(f"ALL OK! All samples inserted successfully")

