import requests
import json
import yaml
import argparse
import time
from datetime import datetime
import psycopg2
from psycopg2.extras import execute_values, Json
import sys
from collections import defaultdict
import pandas as pd

import log
logger = log.getlogger("wpscan_db_queries", level=log.DEBUG, filename='/path/to/logs/wpscanDB_scrape.log')


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
                self.conn = psycopg2.connect(dbname=self.database, user=self.user, password=self.password, host=self.host, port=self.port)
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

def load_config_yaml(conf_fname):
    with open(conf_fname) as s:
        config = yaml.safe_load(s)
    return config


def perform_request(slug, token, plugin):
    endpoint = "https://wpscan.com/api/v3"

    plugins_path = f"/plugins/{slug}"
    themes_path = f"/themes/{slug}"

    headers_auth_str = f"Token token={token}"

    if plugin:
        url = endpoint + plugins_path.format(slug=slug)
    else:
        url = endpoint + themes_path.format(slug=slug)

    response = requests.get(url, headers={'Authorization': headers_auth_str})

    if response.status_code == 200:
        return response.json()
    elif response.status_code == 401:
        logger.warning('Unauthorized')
        raise Exception
    elif response.status_code == 403:
        logger.warning("Authentication failure")
        raise Exception
    elif response.status_code == 429:
        logger.warning("Quota Exceeded")
        raise Exception
    elif response.status_code == 500:
        logger.warning("Internal Server Error")
        raise Exception



def add_rc_components (db_bindings, plugin):
    query_sw_vers = f"""select distinct what 
                from support_cfan_random_components 
                where {"label='WordPress plugins'" if plugin
                    else "label='WordPress themes'"};"""
    query_known_comp = "select distinct what from scrape_wpscandb;"

    with psycopg2.connect(dbname=db_bindings['databases']['pipeline'], user=db_bindings['users']['user'],
                          password=db_bindings['passwords']['password'],
                          host=db_bindings['host'], port=db_bindings['port']) as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(query_known_comp)  # or support_cfan_components.label='WordPress plugins' or support_cfan_components.label='WordPress themes'
                known_components = [x[0] for x in cur.fetchall()]
            except psycopg2.errors.Error as e:
                logger.exception(e)
                sys.exit(-1)
    with psycopg2.connect(dbname=db_bindings['databases']['wptagent'], user=db_bindings['users']['user'],
                          password=db_bindings['passwords']['password'],
                          host=db_bindings['host'], port=db_bindings['port']) as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(query_sw_vers)  # or support_cfan_components.label='WordPress plugins' or support_cfan_components.label='WordPress themes'
                all_components = set([x[0] for x in cur.fetchall()])
                components_to_lookup = all_components - set(known_components)
                logger.debug(f"Selected {len(components_to_lookup)} records from random crawling.")
            except psycopg2.errors.Error as e:
                logger.exception(e)
                sys.exit(-1)
    return list(components_to_lookup)

def select_data(db_bindings, plugin):

    query = f"""select distinct support_cfan_components.what 
                from support_cfan_components 
                left join scrape_wpscandb USING(what)
                where scrape_wpscandb.what is null AND
                {"support_cfan_components.label='WordPress plugins'" if plugin
                    else "support_cfan_components.label='WordPress themes'"};"""

    with psycopg2.connect(dbname=db_bindings['databases']['pipeline'], user=db_bindings['users']['user'],
                          password=db_bindings['passwords']['password'],
                          host=db_bindings['host'], port=db_bindings['port']) as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(query)  # or support_cfan_components.label='WordPress plugins' or support_cfan_components.label='WordPress themes'
                components_to_lookup = [x[0] for x in cur.fetchall()]
                logger.debug(f"Selected {len(components_to_lookup)} records.")
            except psycopg2.errors.Error as e:
                logger.exception(e)
                sys.exit(-1)

    return components_to_lookup

def persist_data(data, db_bindings):
    db_wrapper = DbWrapper(db_bindings, 'pipeline', 'postgres', True)
    cur = db_wrapper.get_cursor()

    insert_stmt = f"""
        INSERT INTO scrape_wpscandb ( what, friendly_name, latest_version, last_updated, popular, vulnerabilities, scrape_timestamp )
        VALUES %s
        ON CONFLICT DO NOTHING;
    """
    try:
        execute_values(cur, insert_stmt, data,
                       template="( %(what)s, %(friendly_name)s, %(latest_version)s, %(last_updated)s, %(popular)s, %(vulnerabilities)s::json[], %(scrape_timestamp)s )",
                       page_size=500)
        logger.debug("Insert successful")
    except psycopg2.errors.OperationalError as op_err:
        logger.exception(op_err)
        logger.info('Retrying insert...')

        db_wrapper.release_cursor()
        cur = db_wrapper.get_cursor()

        try:
            execute_values(cur, insert_stmt, data,
                           template="( %(what)s, %(friendly_name)s, %(latest_version)s, %(last_updated)s, %(popular)s, %(vulnerabilities)s::json[], %(scrape_timestamp)s )",
                           page_size=500)
            logger.debug("Insert successful")
        except psycopg2.errors.Error as e:
            logger.exception(e)

    except psycopg2.errors.Error as e:
        logger.exception(e)

    db_wrapper.release_cursor()
    logger.info("Persist completed.")




def one_round_scans(plugin_scan, db_bindings, token):

    products_to_scan = select_data(db_bindings, plugin_scan)
    products_to_scan.extend(
        add_rc_components(db_bindings, plugin_scan)
                            )

    to_insert = []
    if len(products_to_scan) > 0:
        for plugin in products_to_scan:
            slug = plugin.lower().replace(' ', '-')
            try:
                match = perform_request(slug, token, plugin_scan)

                tmp_res = defaultdict( lambda : None )
                tmp_res.update({
                    'what': plugin,
                    'scrape_timestamp': datetime.now()
                })

                if match:
                    for what in match.keys():
                        tmp_res.update(match[what])

                    vulns = tmp_res.pop('vulnerabilities', None)
                    if vulns:
                        tmp_res['vulnerabilities'] = [Json(x) for x in vulns]

                to_insert.append(tmp_res)

                logger.debug('Sleeping...')
                time.sleep(10)

            except Exception as e:
                logger.warning(e)
                break

    if len(to_insert) > 0:
        logger.info(f'Persisting {len(to_insert)} records...')
        persist_data(to_insert, db_bindings)
        
        
def main(db_bindings, token):
    logger.info('Proceeding to scan plugins...')
    one_round_scans(True, db_bindings, token)

    logger.info('Proceeding to scan themes...')
    one_round_scans(False, db_bindings, token)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scanning the Wpscan DB.')
    parser.add_argument('--conf', dest='conf_fname', nargs='?', action='store',
                        const=sum, default="/path/to/conf/config.yaml",
                        help='Pipeline configuration file . (default: %(default)s)')

    args = parser.parse_args()
    config = load_config_yaml(args.conf_fname)
    postgres = config['global']['postgres']
    key = config['services']['wpscanDB']['api_key']

    main(postgres, key)
