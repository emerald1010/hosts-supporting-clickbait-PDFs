import psycopg2
import os, time, argparse, yaml, random, sys
import shlex
from subprocess import check_call, CalledProcessError, run, PIPE, STDOUT
from datetime import datetime, timedelta
import shutil

from collections import deque

import log

WPTAGENT_BASE_DIR = '/path/to/wptagent'
WPTAGENT_CMDS = os.path.join(WPTAGENT_BASE_DIR, 'bin/python3') + ' ' + os.path.join(WPTAGENT_BASE_DIR,  'wptagent.py') + ' ' + \
                '-vvvv --xvfb --shaper none --testurl  "{testurl}" --browser Chrome --testout json --testoutdir "{testoutdir}" --server {serverurl} --testspec {testspec}'
# four `v`s are necessary to get info from DEBUG output of the agent
WPTAGENT_SERVER_URL = 'http://{}localhost:8089/'
WPTAGENT_TESTSPEC = os.path.join(WPTAGENT_BASE_DIR, 'testspec.json')
WPTAGENT_WORKDIR = '/path/to/wptagent-workdir'


class DbWrapper:
    def __init__(self, db_bindings, database, user, autocommit):
        self.database = db_bindings['databases'][database]
        self.user = db_bindings['users'][user]
        self.password = db_bindings['passwords'][password]
        self.host = db_bindings['host']
        self.port = db_bindings['port']
        self.autocommit = autocommit

        self.conn = None
        self.cursor = None

    def _manage_connection(self):
        if self.conn == None or self.conn.closed != 0 or (self.conn.closed == 0 and self.cursor.closed):
            if self.conn and self.conn.closed == 0 and self.cursor.closed:
                self.release_cursor()

            try:
                self.conn = psycopg2.connect(dbname=self.database, user=self.user, password=self.password, host=self.host, port=self.port,
                                             keepalives=1, keepalives_idle=30, keepalives_interval=10, keepalives_count=5)
                self.conn.autocommit = self.autocommit
            except (psycopg2.errors.OperationalError, psycopg2.errors.DatabaseError) as e:
                __LOGGER__.exception(e)
                __LOGGER__.info('Retrying connection...')
                time.sleep(10)

                try:
                    self.conn = psycopg2.connect(dbname=self.database, user=self.user, password=self.password,
                                                 host=self.host, port=self.port,
                                                 keepalives=1, keepalives_idle=30, keepalives_interval=10,
                                                 keepalives_count=5)
                    self.conn.autocommit = self.autocommit
                except Exception as e:
                    __LOGGER__.exception(e)
                    sys.exit(-1)
                else:
                    self.cursor = self.conn.cursor()

            else:
                self.cursor = self.conn.cursor()

    def get_cursor(self):
        self._manage_connection()
        return self.cursor

    def release_cursor(self):
        self.cursor.close()
        self.conn.close()


def cdn_analyses(worker_name, db_bindings, excluded_domains, client_logs_path):
    logger = log.getlogger(component=f'wptagent-cdn', level=log.INFO, filename=client_logs_path)

    with psycopg2.connect(dbname=db_bindings['databases']['wptagent'], user=db_bindings['users']['user'],
                          password=db_bindings['passwords']['password'], host=db_bindings['host']) as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    """SELECT url
                       FROM assigned_urls
                       LEFT JOIN job USING(url)
                       WHERE agent = %s AND job.url IS NULL;
                    """, (worker_name, )
                )
                wptagent_inspected_urls = [x[0] for x in cur.fetchall()
                                           if not any([e for e in excluded_domains if e in x[0]])]
            except psycopg2.errors.Error as e:
                logger.exception(e)
                wptagent_inspected_urls = []
    tot_urls = len(wptagent_inspected_urls)
    logger.info("Selected {} files.".format(tot_urls))

    random.shuffle(wptagent_inspected_urls)
    for i, url in enumerate(wptagent_inspected_urls):
        logger.info(f'{i} / {tot_urls}:\tProcessing url {url}')

        disposable_dir = os.path.join(WPTAGENT_WORKDIR, f"{worker_name}-{str(time.time())}")
        os.mkdir(disposable_dir)
        cmd = WPTAGENT_CMDS.format(testurl=url, testoutdir=disposable_dir, serverurl=WPTAGENT_SERVER_URL.format('cdn_analyses.'),
                                   testspec=WPTAGENT_TESTSPEC)
        logger.debug(cmd)
        try:
            check_call(shlex.split(cmd))
        except CalledProcessError as e:
            logger.exception(e)
        finally:
            shutil.rmtree(disposable_dir)

    logger.info('Analyzed all URLs.')
    logger.debug('Now going to delete all urls analyzed today from the table...')
    time.sleep(10)

    db_wrapper = DbWrapper(db_bindings, 'wptagent', 'user', True)
    cur = db_wrapper.get_cursor()
    for url in wptagent_inspected_urls:
        try:
            cur.execute("DELETE FROM assigned_urls WHERE url = %s AND agent = %s;",
                        (url, worker_name)
                        )
        except psycopg2.errors.OperationalError as op_err:
            logger.exception(op_err)
            logger.info(f'Details: {url}, {worker_name}\nRetrying delete...')

            db_wrapper.release_cursor()
            cur = db_wrapper.get_cursor()

            try:
                cur.execute("DELETE FROM assigned_urls WHERE url = %s AND agent = %s;",
                            (url, worker_name)
                            )
            except psycopg2.errors.Error as e:
                logger.exception(e)
                logger.info('Failed in getting cursor.')
                sys.exit(-1)

        except psycopg2.errors.Error as e:
            logger.exception(e)
            sys.exit(-1)
    db_wrapper.release_cursor()
    logger.info('Delete completed successfully. Exiting...')
    return




def call_wptagent(url_rescan, logger, endpoint):
    worker_name = os.uname().nodename
    disposable_dir = os.path.join(WPTAGENT_WORKDIR, f"{worker_name}-{str(time.time())}")
    os.mkdir(disposable_dir)

    url, rescan = url_rescan

    # subdomain = 'compromised_infrastructure.'
    subdomain = endpoint
    if rescan:
        subdomain = 'rescan.' + endpoint

    cmd = WPTAGENT_CMDS.format(testurl=url, testoutdir=disposable_dir,
                               serverurl=WPTAGENT_SERVER_URL.format(subdomain),
                               testspec=WPTAGENT_TESTSPEC)
    logger.debug(cmd)

    timeout = False

    try:
        completed_process = run(shlex.split(cmd), check=True, stdout=PIPE, stderr=STDOUT)

        debug_logs = completed_process.stdout.decode('utf-8')[-300:]
        post_log_msg = f'{WPTAGENT_SERVER_URL.format(subdomain)[:-1]} "POST '
        post_address_index = debug_logs.index(post_log_msg)
        if post_address_index >= 0:
            logger.debug('Found POST string in output.')
            if '&error=net' in debug_logs[post_address_index + len(post_log_msg):]:
                logger.warning(
                    f'URL {url} returned error {debug_logs[post_address_index + len(post_log_msg):]}, thus will be inserted in the rescanning queue.')
                timeout = True
                logger.debug('Observed timeout!')
        else:
            logger.error(f'Could not parse {debug_logs}')

    except CalledProcessError as e:
        logger.exception(e)
    finally:
        shutil.rmtree(disposable_dir)
        return timeout


def compromised_infrastructure_analyses (worker_name, db_bindings, excluded_domains, client_logs_path):
    logger = log.getlogger(component=f'wptagent-cf', level=log.INFO, filename=client_logs_path)

    with psycopg2.connect(dbname=db_bindings['databases']['wptagent'], user=db_bindings['users']['user'],
                          password=db_bindings['passwords']['password'], host=db_bindings['host']) as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    """SELECT domain, rescan
                    FROM assigned_urls_cf
                    LEFT JOIN cf_analyses USING (domain)
                    WHERE agent = %s AND (
                    (cf_analyses.domain IS NULL and rescan=False) OR (cf_analyses.domain is not null and rescan=True and times_scanned_off < 3)
                    ); 
                    """, (worker_name, )
                )
                target_domains = cur.fetchall()
            except psycopg2.errors.Error as e:
                logger.exception(e)
                target_domains = []
    tot_urls = len(target_domains)
    logger.info("Selected {} domains.".format(tot_urls))

    netw_err_urls = deque()

    random.shuffle(target_domains)
    for i, t in enumerate(target_domains):

        if any([e for e in excluded_domains if e in t[0]]):
            continue

        logger.info(f'{i} / {tot_urls}:\tProcessing url {t[0]}')

        timed_out = call_wptagent(t, logger, 'compromised_infrastructure.')
        if timed_out:
            netw_err_urls.append((t, 1))

    time.sleep(60*5)
    logger.info('Proceeding with timed-out URLs.')
    while netw_err_urls:
        t, times_inspected = netw_err_urls.popleft()

        if times_inspected < 3:
            timed_out = call_wptagent(t, logger, 'compromised_infrastructure.')

            if timed_out:
                netw_err_urls.append((t, times_inspected + 1))

    logger.info('Analyzed all URLs.')
    return


def random_crawling(worker_name, db_bindings, excluded_domains, client_logs_path):
    logger = log.getlogger(component=f'wptagent-random_crawl', level=log.DEBUG, filename=client_logs_path)

    with psycopg2.connect(dbname=db_bindings['databases']['wptagent'], user=db_bindings['users']['user'],
                          password=db_bindings['passwords']['password'], host=db_bindings['host']) as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    """SELECT sampled_url
                    FROM urls_random_crawling
                    LEFT JOIN random_crawling ON urls_random_crawling.sampled_url = random_crawling.uri
                    WHERE agent = %s AND random_crawling.uri IS NULL; 
                    """, (worker_name, )
                )
                sampled_urls = [
                    x[0] for x in cur.fetchall() if not any([e for e in excluded_domains if e in x[0]])
                ]
            except psycopg2.errors.Error as e:
                logger.exception(e)
                sampled_urls = []
    tot_urls = len(sampled_urls)
    logger.info("Selected {} domains.".format(tot_urls))

    random.shuffle(sampled_urls)
    for i, url in enumerate(sampled_urls):
        logger.info(f'{i} / {tot_urls}:\tProcessing url {url}')

        _ = call_wptagent((url, False), logger, 'random_crawling.')

    logger.info('Analyzed all URLs.')
    return


def load_config_yaml(conf_fname):
    with open(conf_fname) as s:
        config = yaml.safe_load(s)
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run wptagent local analyses')
    parser.add_argument('--conf', dest='conf_fname', nargs='?', action='store',
                        const=sum, default="/path/to/conf/config.yaml",
                        help='Pipeline configuration file . (default: %(default)s)')

    args = parser.parse_args()
    config = load_config_yaml(args.conf_fname)

    worker_name = os.uname().nodename

    client_logs_path = os.path.join(config['global']['file_storage'], 'logs', 'wptagents', f'{worker_name}.log')

    exceptions = []
    with open(config['global']['excluded_domains'], 'r') as exceptions_file:
        for line in exceptions_file.read().split('\n'):
            domain = line.strip()
            if len(domain) > 0:
                exceptions.append(domain)

    #cdn_analyses(worker_name, config['global']['postgres'], exceptions, client_logs_path)
    compromised_infrastructure_analyses(worker_name, config['global']['postgres'], exceptions, client_logs_path)
    random_crawling(worker_name, config['global']['postgres'], exceptions, client_logs_path)
