import requests
import yaml
import argparse
import time
from datetime import datetime
import psycopg2
from collections import defaultdict
from psycopg2.extras import execute_values, Json
import sys
import pandas as pd
from sqlalchemy import create_engine


import log
logger = log.getlogger("NVD_queries", level=log.DEBUG, filename='/path/to/logs/NVD_scrape.log')


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



# https://en.wikipedia.org/wiki/Common_Platform_Enumeration
# cpe:<cpe_version>:<part>:<vendor>:<product>:<version>:<update>:<edition>:<language>:<sw_edition>:<target_sw>:<target_hw>:<other>
def handle_cpe_request(product, version, token):
    if not product or not version:
        raise ValueError

    cpeMatchString = f'cpe:2.3:a:*:{product}:{version}'
    results_per_page = 1000
    start_index = 0

    args = {
        'addOns': 'cves',
        'apiKey': token,
        'resultsPerPage': results_per_page,
        'cpeMatchString': cpeMatchString,
        'startIndex': start_index
            }

    endpoint = "https://services.nvd.nist.gov/rest/json/cpes/1.0?"
    url = endpoint + ''.join([f'{k}={v}&' for k, v in args.items()])

    json_resp_pagination = []

    response = perform_request(url)
    json_resp_pagination.append(response)

    total_res = response.get('totalResults', 0)
    if total_res > 0 and (start_index + results_per_page) < total_res:
        logger.info('Response is paginated...')
        times_request = int(total_res / results_per_page)
        times_request += total_res % results_per_page

        for _ in range(times_request):
            logger.debug('Sleeping...')
            time.sleep(10)
            start_index += results_per_page

            args = {
                'addOns': 'cve',
                'apiKey': token,
                'resultsPerPage': results_per_page,
                'cpeMatchString': cpeMatchString,
                'startIndex': start_index
            }

            url = endpoint + ''.join([f'{k}={v}&' for k, v in args.items()])

            response = perform_request(url)
            json_resp_pagination.append(response)

    logger.debug('Sleeping...')
    time.sleep(10)

    return parse_cpe_response(json_resp_pagination, product, version)

def handle_cve_request(cve_id, token):
    results_per_page = 1000
    start_index = 0
    args = {
        'apiKey': token,
        'resultsPerPage': results_per_page
            }

    endpoint = f"https://services.nvd.nist.gov/rest/json/cve/1.0/{cve_id}?"
    url = endpoint + ''.join([f'{k}={v}&' for k, v in args.items()])

    json_resp_pagination = []

    response = perform_request(url)
    json_resp_pagination.append(response)

    total_res = response.get('totalResults', 0)
    if total_res > 0 and (start_index + results_per_page) < total_res:
        logger.info('Response is paginated...')
        times_request = int(total_res / results_per_page)
        times_request += total_res % results_per_page

        for _ in range(times_request):
            logger.debug('Sleeping...')
            time.sleep(10)
            start_index += results_per_page

            args = {
                'apiKey': token,
                'resultsPerPage': results_per_page,
                'startIndex': start_index
            }

            url = endpoint + ''.join([f'{k}={v}&' for k, v in args.items()])

            response = perform_request(url)
            json_resp_pagination.append(response)
    logger.debug('Sleeping...')
    time.sleep(10)

    return parse_cve_response([response], cve_id)


def perform_request(url):
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    elif response.status_code == 404:
        return {
            'error_msg' : response.content
        }
    else:
        logger.warning(f"Response code {response.status_code}")
        logger.info(url)
        raise Exception(response.content)

def parse_cpe_response(response, product, version):
    result = []

    for resp_json in response:
        total_res = resp_json.get('totalResults', 0)

        if total_res == 0:

            tmp_res = defaultdict( lambda : None )
            tmp_res['what'] = product
            tmp_res['version'] = version
            tmp_res['analysis_timestamp'] = datetime.now()
            tmp_res['error_msg'] = resp_json.get('error_msg', None)
            tmp_res['cves'] = [] # need to state this bc None cannot be cast to text[]
            result.append(tmp_res)

        else:
            json_result = resp_json.get('result', {})

            cpes = json_result.get('cpes', [])

            if len(cpes) == 0:

                tmp_res = defaultdict(lambda: None)
                tmp_res['what'] = product
                tmp_res['version'] = version
                tmp_res['analysis_timestamp'] = datetime.now()
                tmp_res['error_msg'] = resp_json.get('error_msg', None)
                tmp_res['cves'] = []  # need to state this bc None cannot be cast to text[]
                result.append(tmp_res)

            for one_cpe in cpes:

                tmp_res = defaultdict(lambda: None)
                tmp_res['what'] = product
                tmp_res['version'] = version
                tmp_res['analysis_timestamp'] = datetime.now()
                tmp_res['error_msg'] = resp_json.get('error_msg', None)

                tmp_res['cpe'] = one_cpe.get('cpe23Uri')
                tmp_res['last_modified_date'] = one_cpe.get('lastModifiedDate')

                titles = one_cpe.get('titles', [])
                for t in titles:
                    title = t.get('title', None)
                    if title:
                        tmp_res['title'] = title
                        break

                refs = one_cpe.get('refs', [])
                for r in refs:
                    if r.get('type', '') == 'Vendor':
                        tmp_res['vendor'] = r.get('ref', None)
                        break

                tmp_res['cves'] = one_cpe.get('vulnerabilities', [])

                result.append(tmp_res)

    return result

def parse_cve_response(response, cve_id):
    result = []

    for resp_json in response:
        total_res = resp_json.get('totalResults', 0)

        if total_res == 0:
            tmp_res = defaultdict( lambda : None )
            tmp_res['cve_id'] = cve_id
            tmp_res['analysis_timestamp'] = datetime.now()
            tmp_res['error_msg'] = resp_json.get('error_msg', None)
            tmp_res['cwes'] = []
            tmp_res['description'] = []
            tmp_res['reference_data'] = []
            result.append(tmp_res)

        else:

            results = resp_json.get('result', {})
            cve_json = results.get('CVE_Items', [])

            for cve_dict in cve_json:
                cve_item = cve_dict.get('cve', {})

                tmp_res = defaultdict(lambda: None)
                tmp_res['cve_id'] = cve_id
                tmp_res['analysis_timestamp'] = datetime.now()
                tmp_res['error_msg'] = resp_json.get('error_msg', None)

                tmp_res['data_type'] = cve_item.get('data_type', None)
                tmp_res['data_format'] = cve_item.get('data_format', None)
                tmp_res['CVE_data_meta'] = Json(cve_item.get('CVE_data_meta', None))

                description = cve_item.get('description', {})
                if 'description_data' in description.keys():
                    tmp_res['description'] = []
                    for description_data in description.get('description_data', []):
                        lang = description_data.get('lang', '')
                        if lang == 'en':
                            tmp_res['description'].append(description_data.get('value', ''))

                references = cve_item.get('references', {})
                if 'reference_data' in references.keys():
                    tmp_res['reference_data'] = [Json(x) for x in references.get('reference_data', '')]

                problemtype = cve_item.get('problemtype', {})
                if 'problemtype_data' in problemtype.keys():
                    tmp_res['cwes'] = []
                    for item in problemtype.get('problemtype_data', []):
                        description = item.get('description', [])
                        for desc in description:
                            lang = desc.get('lang', '')
                            if lang == 'en':
                                tmp_res['cwes'].append(desc.get('value', ''))

                result.append(tmp_res)

    return result




def persist_data_cpe(data, db_bindings):
    db_wrapper = DbWrapper(db_bindings, 'pipeline', 'postgres', True)
    cur = db_wrapper.get_cursor()

    insert_stmt = f"""
        INSERT INTO nvd_scrape ( what, version, cpe, last_modified_date, title, vendor,  cves, analysis_timestamp, error_msg )
        VALUES %s
        ON CONFLICT (cpe) DO NOTHING;
    """

    # df = pd.DataFrame.from_dict(data)
    # df.drop_duplicates(['what', 'version', 'cpe'], inplace=True)

    try:
        execute_values(cur, insert_stmt, data,
                       template="( %(what)s, %(version)s, %(cpe)s, %(last_modified_date)s, %(title)s, %(vendor)s,  %(cves)s::text[], %(analysis_timestamp)s, %(error_msg)s )",
                       page_size=500)
        logger.debug("Insert successful")
    except psycopg2.errors.UniqueViolation as e2:
        logger.exception(e2)
    except psycopg2.errors.OperationalError as op_err:
        logger.exception(op_err)
        logger.info('Retrying insert...')

        db_wrapper.release_cursor()
        cur = db_wrapper.get_cursor()

        try:
            execute_values(cur, insert_stmt, df.to_dict(orient='records'),
                           template="( %(what)s, %(version)s, %(cpe)s, %(last_modified_date)s, %(title)s, %(vendor)s,  %(cves)s::text[], %(analysis_timestamp)s, %(error_msg)s )",
                           page_size=500)
            logger.debug("Insert successful")
        except psycopg2.errors.Error as e:
            logger.exception(e)

    except psycopg2.errors.Error as e:
        logger.exception(e)

    db_wrapper.release_cursor()
    logger.info("Persist completed.")

def persist_data_cve(data, db_bindings):
    db_wrapper = DbWrapper(db_bindings, 'pipeline', 'postgres', True)
    cur = db_wrapper.get_cursor()

    insert_stmt = f"""
        INSERT INTO cve_dump ( cve_id, analysis_timestamp, error_msg, data_type, data_format, CVE_data_meta,  description, reference_data, cwes )
        VALUES %s;
    """
    try:
        execute_values(cur, insert_stmt, data,
                       template="( %(cve_id)s, %(analysis_timestamp)s, %(error_msg)s, %(data_type)s, %(data_format)s, %(CVE_data_meta)s, %(description)s::text[], %(reference_data)s::json[], %(cwes)s::text[] )",
                       page_size=500)
        logger.debug("Insert successful")
    except psycopg2.errors.UniqueViolation as e2:
        logger.exception(e2)
    except psycopg2.errors.OperationalError as op_err:
        logger.exception(op_err)
        logger.info('Retrying insert...')

        db_wrapper.release_cursor()
        cur = db_wrapper.get_cursor()

        try:
            execute_values(cur, insert_stmt, data,
                           template="(%(cve_id)s, %(analysis_timestamp)s, %(error_msg)s, %(data_type)s, %(data_format)s, %(CVE_data_meta)s, %(description)s::text[], %(reference_data)s::json[], %(cwes)s::text[] )",
                           page_size=500)
            logger.debug("Insert successful")
        except psycopg2.errors.Error as e:
            logger.exception(e)

    except psycopg2.errors.Error as e:
        logger.exception(e)

    db_wrapper.release_cursor()
    logger.info("Persist completed.")



def cves_from_random_crawling(db_bindings, key):
    query_sw_vers = f"""select distinct what, version
                from support_cfan_random_components 
                where version is not null and (label='CMS' OR label='Ecommerce' OR label='Hosting panels' OR 
                label='Web servers' OR what = 'PHP');"""
    query_known_comp = """select distinct what, version 
                          from nvd_scrape;"""

    with psycopg2.connect(dbname=db_bindings['databases']['pipeline'], user=db_bindings['users']['user'],
                          password=db_bindings['passwords']['password'],
                          host=db_bindings['host'], port=db_bindings['port']) as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(query_known_comp)  # or support_cfan_components.label='WordPress plugins' or support_cfan_components.label='WordPress themes'
                known_components = cur.fetchall()
            except psycopg2.errors.Error as e:
                logger.exception(e)
                sys.exit(-1)
    df_known_components = pd.DataFrame(known_components, columns=['what', 'version'])

    with psycopg2.connect(dbname=db_bindings['databases']['wptagent'], user=db_bindings['users']['user'],
                          password=db_bindings['passwords']['password'],
                          host=db_bindings['host'], port=db_bindings['port']) as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(query_sw_vers)  # or support_cfan_components.label='WordPress plugins' or support_cfan_components.label='WordPress themes'
                all_components = cur.fetchall()
            except psycopg2.errors.Error as e:
                logger.exception(e)
                sys.exit(-1)
    df_all_rc_components = pd.DataFrame(all_components, columns=['what', 'version'])
    merged = df_all_rc_components.merge(df_known_components, how='left', on='what', suffixes=['_rc', '_nvd'])
    merged.drop(merged[merged.version_nvd.notna()].index, inplace=True)

    if merged.empty:
        logger.debug("No component selected from random crawling.")
    else:
        logger.debug(f"Selected {len(merged.shape[0])} records from random crawling.")
        logger.warning('Raising exception to be able to test this!')
        raise Exception('Finally :)')

    # logger.info('Analysis started.')
    #
    # results = merged.apply(lambda x: handle_cpe_request(x.product, x.version, key), axis=1)


def cve_per_product(db_bindings, key):
    engine = create_engine('postgresql://{user}:{password}@{host}:{port}/{dbname}'
                           .format(dbname=db_bindings['databases']['pipeline'], user=db_bindings['users']['user'], password=db_bindings['passwords']['password'],
                                   host=db_bindings['host'], port=db_bindings['port']), pool_pre_ping=True)
    conn = engine.connect().execution_options(
        stream_results=True)

    query = """select distinct support_cfan_components.what as "product", support_cfan_components.version  
                from support_cfan_components 
                left join nvd_scrape ON support_cfan_components.what = nvd_scrape.what AND support_cfan_components.version = nvd_scrape.version
                where nvd_scrape.what is null AND  support_cfan_components.version is not null AND
                (nvd_scrape.what is null OR nvd_scrape.version is null) AND
                (support_cfan_components.label='CMS' OR support_cfan_components.label='Ecommerce' 
                OR support_cfan_components.label='Hosting panels' OR support_cfan_components.label='Web servers'
                OR support_cfan_components.what = 'PHP');"""

    logger.info('Analysis started.')
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=50):

        to_insert = []

        for row in chunk_dataframe.itertuples(index=False):
            res = handle_cpe_request(row.product, row.version, key)
            to_insert.extend(res)

        logger.info(f'Collected {len(to_insert)} records. Proceeding to insert...')
        persist_data_cpe(to_insert, db_bindings)

    engine.dispose()

def fetch_cves(db_bindings, key):
    engine = create_engine('postgresql://{user}:{password}@{host}:{port}/{dbname}'
                           .format(dbname=db_bindings['databases']['pipeline'], user=db_bindings['users']['user'], password=db_bindings['passwords']['password'],
                                   host=db_bindings['host'], port=db_bindings['port']), pool_pre_ping=True)
    conn = engine.connect().execution_options(
        stream_results=True)

    query = """
            with all_CVEs as (
                select array_remove(cves, '') as "cves"
                from nvd_scrape 
                where cardinality(cves) > 0
            ), unique_cves as (
                select distinct unnest(cves) as "dist_cves"
                from all_CVEs where cardinality(cves)>0
            ) select dist_cves
            from unique_cves
            left join cve_dump ON unique_cves.dist_cves = cve_dump.cve_id
            where cve_dump.cve_id is null;
            """

    for chunk_dataframe in pd.read_sql(query, conn, chunksize=50):

        to_insert = []

        for row in chunk_dataframe.itertuples(index=False):
            res = handle_cve_request(row.dist_cves, key)
            to_insert.extend(res)

        logger.info(f'Collected {len(to_insert)} records. Proceeding to insert...')
        persist_data_cve(to_insert, db_bindings)

    engine.dispose()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetching data from NVD DB.')
    parser.add_argument('--conf', dest='conf_fname', nargs='?', action='store',
                        const=sum, default="/path/to/conf/config.yaml",
                        help='Pipeline configuration file . (default: %(default)s)')
    parser.add_argument("--scan_type", dest="scan",      nargs=1,    action='store',
                        required="True", choices=['cpe', 'cve'],
                        help="Scan URLs or PDFs.")

    args = parser.parse_args()
    config = load_config_yaml(args.conf_fname)
    postgres = config['global']['postgres']
    key = config['services']['NVD']['api_key']

    if args.scan[0] == 'cpe':
        cve_per_product(postgres, key)
        cves_from_random_crawling(postgres, key)
    else:
        fetch_cves(postgres, key)
