import yaml
import argparse
from zipfile import ZipFile
import os
import sys
import psycopg2
from psycopg2.extras import execute_values
import time
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urlunparse
from random import sample, shuffle
import hashlib

import log

WPTAGENT_DATA_FOLDER = 'wptagent-data'
__NESTED_LEVELS__ = 4
N_LINKS_INSPECTION = 20

# https://developer.mozilla.org/en-US/docs/Web/HTML/Element/a#attr-href
# href links can start w/
# http(s), domain, /, ?, #

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



def load_config_yaml(conf_fname):
    with open(conf_fname) as s:
        config = yaml.safe_load(s)
    return config

def rel_path(fname, nested_levels=__NESTED_LEVELS__):
    """
    E.g., if fname = "abcdefghilmnopq", this function returns "ab/cd/ef/gh/"
    """
    if len(fname) >= 2 * nested_levels:
        return "/".join([fname[2 * i:2 * (i + 1)] for i in range(0, nested_levels)]) + "/"
    raise Exception(
        "Relative file storage path cannot be determined because len({})={} < {} chars"
        .format(fname, len(fname), 2 * nested_levels))


def get_hash(domain):
    m = hashlib.sha256()
    m.update(domain.encode('utf-8'))
    return m.hexdigest()


# add checks on png, jpg, svg, json, txt
def points_to_static_file(href):
    file_extensions = ['.png', '.jpg', '.jpeg', '.svg', '.json', '.js', '.txt', '.xml']
    if any([ext for ext in file_extensions if href.endswith(ext)]):
        return True
    return False

def within_scope_and_correct(domain, href):
    reference_scheme, reference_netloc, _, _, _, _ = urlparse(domain)

    href_scheme, href_netloc, href_path, href_params, href_query, href_fragment = urlparse(href)

    if href_scheme or href_netloc:
        if href_scheme == reference_scheme and href_netloc == reference_netloc: # if href startswith domain
            return href

        elif ( (not href.startswith('http')) or (not href_scheme == reference_scheme) )\
                and ( href.startswith(reference_netloc) or href_netloc == reference_netloc ):
            return urlunparse((reference_scheme, href_netloc, href_path, href_params, href_query, href_fragment))

    else:
        if href.startswith('?'): #assume it's the query
            return urlunparse((reference_scheme, reference_netloc, '', '', href[1:], ''))
        elif href.startswith('#'): #assume it's the fragment
            return urlunparse((reference_scheme, reference_netloc, '', '', '',  href[1:]))
        else: #assume it's the path
            return urlunparse((reference_scheme, reference_netloc, href, '', '', ''))

def sample_w_check(population): # avoid value errors when len population < k
    uniques_in_pop = list(set(population))
    len_population = len(uniques_in_pop)

    if len_population <= N_LINKS_INSPECTION:
        shuffle(uniques_in_pop)
        return uniques_in_pop
    else:
        return sample(uniques_in_pop, k=N_LINKS_INSPECTION)



def find_parse_files(filesys_entry_point, domain, domain_hash, logger):
    if not os.path.isdir(
        os.path.join(filesys_entry_point, rel_path(domain_hash))
    ):
        logger.warning(f'No directory found for {domain_hash}')
        return None

# TODO: what if there are 2 ?
    if os.path.exists(os.path.join(filesys_entry_point, rel_path(domain_hash), domain_hash + '.zip')):
        fname = os.path.join(filesys_entry_point, rel_path(domain_hash), domain_hash + '.zip')
    elif os.path.exists(os.path.join(filesys_entry_point, rel_path(domain_hash), domain_hash + '_rescan.zip')):
        fname = os.path.join(filesys_entry_point, rel_path(domain_hash), domain_hash + '_rescan.zip')
    else:
        logger.warning(f'No compatible file found in {domain_hash} folder!')
        sys.exit(-1)

    with ZipFile(fname, 'r') as archive:
        if 'body.txt' in archive.namelist():
            with archive.open('body.txt') as body:
                soup = BeautifulSoup(body.read(), 'lxml')
        else:
            logger.warning(f'No body found in archive {fname}, was {domain} scanned alright?')

            if os.path.exists(os.path.join(filesys_entry_point, rel_path(domain_hash), domain_hash + '_rescan.zip')):

                with ZipFile(
                        os.path.join(filesys_entry_point, rel_path(domain_hash), domain_hash + '_rescan.zip'), 'r'
                ) as archive:
                    if 'body.txt' in archive.namelist():
                        with archive.open('body.txt') as body:
                            soup = BeautifulSoup(body.read(), 'lxml')
                    else:
                        logger.warning(f"No body found in archive {os.path.join(filesys_entry_point, rel_path(domain_hash), domain_hash + '_rescan.zip')}, was {domain} scanned alright?")
                        return []
            else:
                return []


    if not soup:
        logger.error('Could not read body.txt')
        sys.exit(-1)

    links = []
    for link_element in soup.find_all('a'):
        href = link_element.get('href', None)

        if not href:
            continue

        corrected_url = within_scope_and_correct(domain, href)  # relative domains
        if corrected_url:
            if not points_to_static_file(corrected_url): # remove different domains, remove other protocols
                links.append(corrected_url)
    return links



def select_data(db_bindings, logger):
    db_wrapper_pip = DbWrapper(db_bindings, 'pipeline', 'user', True)
    cur = db_wrapper_pip.get_cursor()
    try:
        cur.execute("""
            with online_offline_netlocs as (
               select distinct all_urls.netloc
               from all_urls
               join url_headers using(uri)
               where url_headers.datetime_first_offline is null or (url_headers.datetime_first_offline is not null and url_headers.offline_count < 3)
            )
            select distinct cp_scanning.domain
            from cp_scanning
            join all_urls on cp_scanning.domain = all_urls.netloc
            join samplecluster on all_urls.filehash = samplecluster.filehash
            join imported_samples on all_urls.filehash = imported_samples.filehash
            left join hosting_services on all_urls.domain_remapped = hosting_services.provider
            where (hosting_services.hosting_type is null
                OR hosting_services.hosting_type = 'same_domain_free_webhosting')
                and samplecluster.is_seo=true
                and imported_samples.upload_date = current_date - interval '1 day'
                and all_urls.guessed_cp is null
                and all_urls.netloc in (
                   select distinct online_offline_netlocs.netloc
                   from online_offline_netlocs
                )
                AND all_urls.has_http_scheme = True
                AND all_urls.is_email = FALSE
                AND all_urls.is_empty = FALSE
                AND all_urls.is_relative_url = FALSE
                AND all_urls.is_local = FALSE
                AND all_urls.has_invalid_char = FALSE
                AND all_urls.has_valid_tld = True
                
                AND cp_scanning.endpoints IS NULL
            group by cp_scanning.domain
            having count(endpoints)=0;
            """)
        no_detection_domains = set([x[0] for x in cur.fetchall()])

        logger.debug(f'Selected {len(no_detection_domains)} from cp_scanning for today.')
    except psycopg2.errors.Error as e:
        logger.exception(e)
        sys.exit(-1)
    db_wrapper_pip.release_cursor()

    db_wrapper_wpta = DbWrapper(db_bindings, 'wptagent', 'user', True)
    cur = db_wrapper_wpta.get_cursor()
    try:
        cur.execute("""
                select cf_analyses.domain, cf_analyses.domain_hash
                from cf_analyses;
                """)
        scanned_doms_dhashes = cur.fetchall()

        cur.execute("""
                select distinct domain
                from urls_random_crawling;
                """)
        already_randomly_scanned_doms = set([x[0] for x in cur.fetchall()])
    except psycopg2.errors.Error as e:
        logger.exception(e)
        sys.exit(-1)
    db_wrapper_wpta.release_cursor()

    df = pd.DataFrame(scanned_doms_dhashes, columns=['domain', 'domain_hash'])
    to_inspect = df[df.domain.isin(no_detection_domains -  already_randomly_scanned_doms)]

    return to_inspect

def persist(db_bindings, data, logger):
    db_wrapper = DbWrapper(db_bindings, 'wptagent', 'user', True)
    cur = db_wrapper.get_cursor()

    query = """
    INSERT INTO urls_random_crawling ( domain, domain_hash, extracted_urls, sampled_url, agent )
    VALUES %s
    ON CONFLICT (sampled_url) DO NOTHING;
    """

    try:
        execute_values(cur, sql=query, argslist=data,
                       template="( %(domain)s, %(domain_hash)s, %(extracted_urls)s::text[], %(sampled_url)s, %(agent)s )",
                       page_size=1000)
    except psycopg2.errors.OperationalError as e1:
        logger.exception(e1)
        logger.info('Retrying insert...')
        db_wrapper.release_cursor()
        cur = db_wrapper.get_cursor()

        try:
            execute_values(cur, sql=query, argslist=data,
                           template="( %(domain)s, %(domain_hash)s, %(extracted_urls)s::text[], %(sampled_url)s, %(agent)s )",
                           page_size=1000)
        except Exception as e:
            logger.exception(e)
            db_wrapper.release_cursor()
            sys.exit(-1)

    except psycopg2.errors.UniqueViolation as e2:
        logger.exception(e2)

    except psycopg2.errors.Error as e:
        logger.exception(e)
        sys.exit(-1)  # save quota
    db_wrapper.release_cursor()
    logger.debug('Persist completed successfully.')


def update_domain_hash(db_bindings):
    db_wrapper = DbWrapper(db_bindings, 'wptagent', 'user', True)
    cur = db_wrapper.get_cursor()

    try:
        cur.execute("""
            select random_crawling.uri, random_crawling.domain_hash
            from random_crawling
            full join cf_analyses using (domain_hash)
            where cf_analyses.domain_hash is null;
        """)
        to_update = cur.fetchall()
    except psycopg2.errors.OperationalError as e1:
        logger.exception(e1)

        db_wrapper.release_cursor()
        cur = db_wrapper.get_cursor()

        try:
            cur.execute("""
                select random_crawling.uri, random_crawling.domain_hash
                from random_crawling
                full join cf_analyses using (domain_hash)
                where cf_analyses.domain_hash is null;
            """)
            to_update = cur.fetchall()
        except Exception as e:
            logger.exception(e)
            db_wrapper.release_cursor()
            sys.exit(-1)

    except psycopg2.errors.Error as e:
        logger.exception(e)
        sys.exit(-1)
    db_wrapper.release_cursor()

    df = pd.DataFrame(to_update, columns=['uri', 'old_domain_hash'])
    df['domain'] = df.uri.apply(lambda x: urlparse(x).scheme + '://' + urlparse(x).netloc)
    df['domain_hash'] = df.domain.apply(get_hash)

    logger.info('Starting update...')

    insert_query = """
    UPDATE random_crawling
    SET domain_hash=%s
    WHERE uri=%s AND domain_hash=%s;
    """
    db_wrapper = DbWrapper(db_bindings, 'wptagent', 'user', True)
    cursor = db_wrapper.get_cursor()

    for record in df.to_dict(orient='records'):
        try:
            cursor.execute(insert_query,
                           (record['domain_hash'],
                            record['uri'], record['old_domain_hash']))
        except psycopg2.errors.OperationalError as e1:
            logger.exception(e1)
            logger.info('Retrying insert...')
            db_wrapper.release_cursor()
            cursor = db_wrapper.get_cursor()

            try:
                cursor.execute(insert_query,
                               (record['domain_hash'],
                                record['uri'], record['old_domain_hash']))
            except Exception as e:
                logger.exception(e)
                db_wrapper.release_cursor()
                raise
        except psycopg2.errors.Error as e:
            logger.exception(e)
            db_wrapper.release_cursor()
            raise

    db_wrapper.release_cursor()

    logger.info("Update completed!")





def do(config, n_workers):
    client_logs_path = os.path.join(config['global']['file_storage'], 'logs', 'wptagents', 'extracting_urls_from_body.log')
    global logger
    logger = log.getlogger(component='parse_body', level=log.DEBUG, filename=client_logs_path)

    data = select_data(config['global']['postgres'], logger)

    if data.empty:
        logger.info('No domains to scan today!')
        return

    data['extracted_urls'] = data.apply(lambda x: find_parse_files(
        os.path.join(config['global']['file_storage'], WPTAGENT_DATA_FOLDER),
        x['domain'], x['domain_hash'], logger), axis=1)
    data['sampled_url'] = data.extracted_urls.apply(sample_w_check)

    exploded = data.explode('sampled_url')
    agents = [f'wpta{x}' for x in range(0, n_workers, 1)] * (int(exploded.shape[0] / n_workers) + 1)
    exploded['agent'] = agents[:exploded.shape[0]]

    persist(config['global']['postgres'], exploded.to_dict(orient='records'), logger)
    logger.info('Task completed.')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split daily URLs for wptagent workers.')
    parser.add_argument('--conf', dest='conf_fname', nargs='?', action='store',
                        const=sum, default="/path/to/conf/config.yaml",
                        help='Pipeline configuration file . (default: %(default)s)')
    parser.add_argument("--n_workers", dest='n_workers', help="Number of workers. Each worker must have an own DB table.", type=int,  default=10)


    args = parser.parse_args()
    config = load_config_yaml(args.conf_fname)

    do(config, args.n_workers)
    update_domain_hash(config['global']['postgres'])


