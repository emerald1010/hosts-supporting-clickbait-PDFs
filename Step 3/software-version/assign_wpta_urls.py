import psycopg2
from psycopg2.extras import execute_values
import yaml
import argparse
import os
import sys
import random
import time
from datetime import datetime, timedelta
from urllib.parse import urlparse
import tldextract
import pandas as pd

import log

# object_storage_or_cdn = ['strikinglycdn', 's123-cdn-static-a', 'filesusr', 'squarespace', 'webydo', 'shopify',
#             's123-cdn-static-b', 's123-cdn-static-c', 's123-cdn-static', 'mozfiles', 's123-cdn', 's123-cdn-static-d',
#             'amazonaws', 'digitaloceanspaces',
#             'f-static', 'sqhk']


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
                logger.exception(e)
                logger.info('Retrying connection...')
                time.sleep(5)

                try:
                    self.conn = psycopg2.connect(dbname=self.database, user=self.user, password=self.password,
                                                 host=self.host, port=self.port,
                                                 keepalives=1, keepalives_idle=30, keepalives_interval=10,
                                                 keepalives_count=5)
                    self.conn.autocommit = self.autocommit
                except Exception as e:
                    logger.exception(e)
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

class HostingServices:
    def __init__(self, db_bindings):
        self._known_services = self._fetch_known_services(db_bindings)

    def _fetch_known_services(self, db_bindings):
        with psycopg2.connect(dbname=db_bindings['databases']['pipeline'], user=db_bindings['users']['user'],
                              password=db_bindings['passwords']['password'], host=db_bindings['host']) as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute("""
                        select provider, hosting_type
                        from hosting_services
                        where hosting_type = 'object_storage' or hosting_type = 'other_domain_free_webhosting' or hosting_type = 'rogue_cdn';""")
                    services = cur.fetchall()
                except psycopg2.errors.Error as e:
                    logger.exception(e)
                    services = []
        res = {}
        for p, ht in services:
            res[p] = ht

        return res

    @property
    def known_services(self):
        return self._known_services



def load_config_yaml(conf_fname):
    with open(conf_fname) as s:
        config = yaml.safe_load(s)
    return config

# https://stackoverflow.com/a/2135920
def split_equal(a, n):
    random.shuffle(a)
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def get_path(x):
    path = urlparse(x).path
    splits=[x.strip() for x in path.split('/') if len(x.strip()) > 0]
    return '/'.join(splits[:-1])

def assign_cdn_urls(db_bindings, n_agents, log_path):
    logger = log.getlogger(component='wptagent_split_urls', level=log.INFO, filename=log_path)

    db_wrapper = DbWrapper(db_bindings, 'wptagent', 'user', True)
    cur = db_wrapper.get_cursor()

    inspected_dompaths = []
    try:
        cur.execute(
            "SELECT DISTINCT domain_path FROM job;"
        )
        inspected_dompaths = [x[0] for x in cur.fetchall()]
    except psycopg2.errors.OperationalError as op_err:
        logger.exception(op_err)
        logger.info('Retrying select...')

        db_wrapper.release_cursor()
        cur = db_wrapper.get_cursor()

        try:
            cur.execute(
                "SELECT DISTINCT domain_path FROM job;"
            )
            inspected_dompaths = [x[0] for x in cur.fetchall()]
        except psycopg2.errors.Error as e:
            logger.exception(e)
            logger.info('Failed in getting cursor.')

    except psycopg2.errors.Error as e:
        logger.exception(e)
    if len(inspected_dompaths) == 0:
        sys.exit(-1)

    yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    db_wrapper_pipeline = DbWrapper(db_bindings, 'pipeline', 'user', True)
    cur_pipeline = db_wrapper_pipeline.get_cursor()
    try:
        # samples are later shuffled! This ORDER BY is to remove duplicate filehashes
        cur_pipeline.execute(
            """     SELECT DISTINCT all_urls.uri
                    FROM all_urls
                    JOIN imported_samples i USING(filehash)
                    WHERE
                        i.provider <> 'FromUrl'
                        AND i.upload_date = %s
                        AND all_urls.is_pdf = True
                        AND all_urls.has_http_scheme = True
                        AND all_urls.is_email = FALSE
                        AND all_urls.is_empty = FALSE
                        AND all_urls.is_relative_url = FALSE
                        AND all_urls.is_local = FALSE
                        AND all_urls.has_invalid_char = FALSE
                        AND all_urls.has_valid_tld = True;
            """, (yesterday, )
        )
        yesterdays_urls = [x[0] for x in cur_pipeline.fetchall()]
    except psycopg2.errors.OperationalError as op_err:
        logger.exception(op_err)
        logger.info('Retrying select...')

        db_wrapper_pipeline.release_cursor()
        cur_pipeline = db_wrapper_pipeline.get_cursor()

        try:
            cur_pipeline.execute(
                """ SELECT DISTINCT all_urls.uri
                    FROM all_urls
                    JOIN imported_samples i USING(filehash)
                    WHERE
                        i.provider <> 'FromUrl'
                        AND i.upload_date %s
                        AND all_urls.is_pdf = True
                        AND all_urls.has_http_scheme = True
                        AND all_urls.is_email = FALSE
                        AND all_urls.is_empty = FALSE
                        AND all_urls.is_relative_url = FALSE
                        AND all_urls.is_local = FALSE
                        AND all_urls.has_invalid_char = FALSE
                        AND all_urls.has_valid_tld = True;
                """, (yesterday,)
            )
            yesterdays_urls = cur_pipeline.fetchall()
        except psycopg2.errors.Error as e:
            logger.exception(e)
            yesterdays_urls = []

    except psycopg2.errors.Error as e:
        logger.exception(e)
        yesterdays_urls = []
    db_wrapper_pipeline.release_cursor()

    clean_uri_df = pd.DataFrame(yesterdays_urls, columns=['url'])
    if clean_uri_df.empty:
        logger.debug(f'Going to insert 0 URLs! No URLs were found for today!')
        return

    clean_uri_df['path'] = clean_uri_df['url'].apply(
        lambda x: urlparse(x).scheme + '://' + urlparse(x).netloc + '/' + get_path(x)
    )

    urls_per_path = clean_uri_df[~(clean_uri_df.path.isin(inspected_dompaths))].groupby('path')\
                                                                       .sample(2, replace=True)\
                                                                       .drop_duplicates('url')
    logger.debug(f'Going to insert a total of {len(urls_per_path)} URLs!')

    # the code below works but is not needed atm, logic of script changed
    # flattened_urls = urls_per_path.groupby('path').uri.apply(lambda x: x.values.flatten())
    # try:
    #     main_fallback_uri_df = pd.DataFrame(flattened_urls.to_list(), index=flattened_urls.index, columns=['url', 'fallback_url'])
    # except ValueError as e:
    #     logger.info(e)
    #     logger.debug('No fallback URL for today.')
    #     main_fallback_uri_df = pd.DataFrame(flattened_urls.to_list(), index=flattened_urls.index, columns=['url'])
    #     main_fallback_uri_df['fallback_url'] = pd.Series()
    # parts = split_equal(main_fallback_uri_df.to_dict(orient='records'), n_agents)


    parts = split_equal(urls_per_path.to_dict(orient='records'), n_agents)

    cur = db_wrapper.get_cursor()

    for agent_number, records in zip(range(n_agents), parts):

        insert_stmt = f"""
            INSERT INTO assigned_urls ( url, agent )
            VALUES %s
            ON CONFLICT DO NOTHING;
        """

        for r in records:
            r.update({
            'agent': f'wpta{agent_number}'
        })

        try:
            execute_values(cur, insert_stmt, records, template="( %(url)s, %(agent)s )", page_size=500)
            logger.debug("Insert successful")
        except psycopg2.errors.OperationalError as op_err:
            logger.exception(op_err)
            logger.info('Retrying insert...')

            db_wrapper.release_cursor()
            cur = db_wrapper.get_cursor()

            try:
                execute_values(cur, insert_stmt,  records, template="( %(url)s, %(fallback_url)s, %(agent)s )", page_size=500)
                logger.debug("Insert successful")
            except psycopg2.errors.Error as e:
                logger.exception(e)

        except psycopg2.errors.Error as e:
            logger.exception(e)

    db_wrapper.release_cursor()
    logger.info("Insert completed for all workers.")



def get_domain(url):
    scheme, netloc, _, _, _, _ = urlparse(url)
    return scheme + '://' + netloc

def select_netlocs_compromised_frameworks(db_bindings, logger):
    db_wrapper = DbWrapper(db_bindings, 'pipeline', 'user', True)
    cur = db_wrapper.get_cursor()
    try:
        cur.execute("""
            SELECT uri
            FROM url_headers
            WHERE datetime_first_online > current_date AND datetime_first_offline IS NULL;
            """)
        #   WHERE datetime_first_online >  current_date - interval '1 day' ;
        online_today = [get_domain(x[0]) for x in cur.fetchall()]
    except psycopg2.errors.Error as e:
        logger.exception(e)
        sys.exit(-1)
    db_wrapper.release_cursor()

    return remove_cdns(online_today)

def select_rescan_compromised_frameworks(db_bindings, logger):

    db_wrapper = DbWrapper(db_bindings, 'pipeline', 'user', True)
    cur = db_wrapper.get_cursor()
    try:
        cur.execute("""
            SELECT uri
            FROM url_headers
            WHERE datetime_first_offline > current_date - interval '4 day' AND datetime_first_online IS NOT NULL AND offline_count >= 3;
            """)
        offline_today = set([get_domain(x[0]) for x in cur.fetchall()])
    except psycopg2.errors.Error as e:
        logger.exception(e)
        sys.exit(-1)
    try:
        cur.execute("""
            SELECT uri
            FROM url_headers
            WHERE datetime_first_online IS NOT NULL AND datetime_first_offline IS NULL;
            """)
        still_online = set([get_domain(x[0]) for x in cur.fetchall()])
    except psycopg2.errors.Error as e:
        logger.exception(e)
        sys.exit(-1)
    db_wrapper.release_cursor()

    to_inspect = offline_today - still_online

    return remove_cdns(to_inspect)

def remove_cdns(urls):
    res = []
    for url in urls:
        tld = tldextract.extract(url).domain

        if not any([tld in hosting_provider for hosting_provider in known_service_providers.keys()]):
            scheme, netloc, _, _, _, _ = urlparse(url)
            res.append(scheme + '://' + netloc)

    return res

def assign_compromised_frameworks_urls(db_bindings, n_agents, log_path):
    logger = log.getlogger(component='wptagent_cf_analyses', level=log.INFO, filename=log_path)

    new_urls_today = [(x, False) for x in select_netlocs_compromised_frameworks(db_bindings, logger)]
    rescan_offline = [(x, True) for x in select_rescan_compromised_frameworks(db_bindings, logger)]
    all_domains = new_urls_today + rescan_offline
    random.shuffle(all_domains)
    logger.info(f'Going to insert {len(all_domains)} domains for today.')

    if len(all_domains) == 0:
        logger.info('Actually, no domain to insert. Returning...')
        return

    try:
        assert len(set(new_urls_today) & set(rescan_offline)) == 0
    except AssertionError:
        logger.error(
            f"Data inconsistency: {len(set(new_urls_today) & set(rescan_offline))} urls are both online today and offline today"
        )
        sys.exit(-1)


    parts = split_equal(all_domains, n_agents)

    db_wrapper = DbWrapper(db_bindings, 'wptagent', 'user', True)
    cur = db_wrapper.get_cursor()

    for agent_number, tuples in zip(range(n_agents), parts):

        insert_stmt = f"""
            INSERT INTO assigned_urls_cf ( domain, agent, rescan )
            VALUES %s
            ON CONFLICT DO NOTHING;
        """
        try:
            execute_values(cur, insert_stmt, tuples, template=f"( %s, 'wpta{agent_number}', %s )", page_size=500)
            logger.debug("Insert successful")
        except psycopg2.errors.OperationalError as op_err:
            logger.exception(op_err)
            logger.info('Retrying insert...')

            db_wrapper.release_cursor()
            cur = db_wrapper.get_cursor()

            try:
                execute_values(cur, insert_stmt,  tuples, template=f"( %s, 'wpta{agent_number}', %s )", page_size=500)
                logger.debug("Insert successful")
            except psycopg2.errors.Error as e:
                logger.exception(e)

        except psycopg2.errors.Error as e:
            logger.exception(e)

    db_wrapper.release_cursor()
    logger.info("Insert completed (?) for all workers.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split daily URLs for wptagent workers.')
    parser.add_argument("--n_workers", dest='n_workers', help="Number of workers. Each worker must have an own DB table.", type=int,  default=10)
    parser.add_argument('--conf', dest='conf_fname', nargs='?', action='store',
                        const=sum, default="/path/to/conf/config.yaml",
                        help='Pipeline configuration file . (default: %(default)s)')
    parser.add_argument('--assign', dest='assign', action='store', help='Invoke URL assignment for `cdn`s or `cf`s.',
                        choices=['cdn', 'cf'], required=True)

    args = parser.parse_args()
    config = load_config_yaml(args.conf_fname)

    if args.n_workers and args.assign:
        client_logs_path = os.path.join(config['global']['file_storage'], 'logs', 'wptagents', 'wptagent_split_urls.log')

        hs = HostingServices(config['global']['postgres'])
        global known_service_providers
        known_service_providers = hs.known_services

        if args.assign == 'cdn':
            assign_cdn_urls(config['global']['postgres'], args.n_workers, client_logs_path)
        elif args.assign == 'cf':
            assign_compromised_frameworks_urls(config['global']['postgres'], args.n_workers, client_logs_path)
        else:
            raise ValueError('Wrong invocation argument: either `cdn` or `cf`.')
