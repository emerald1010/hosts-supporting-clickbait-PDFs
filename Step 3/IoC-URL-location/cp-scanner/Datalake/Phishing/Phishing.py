import logging
from os.path import join
import datetime as dt
import psycopg2
from psycopg2.extras import execute_values, Json
from Datalake.DataLake import DataLake

EXCEPTIONS_FILE_PATH = '/path/to/domains_excluded_from_study.txt'


class PhishingDataLake(DataLake):

    def __init__(self, root_folder, db_bindings, user, autocommit_db=True):
        """
        Args:
            root_folder: folder where the data are located
            db_bindings: credentials to connect to the db
            screenshots_files_endpoint: folder where the screenshots of the samples are located
        """
        self.db_bindings = db_bindings

        self.root_folder = root_folder

        self.autocommit_db = autocommit_db

        self._user = user

        self.db_connection = None
        self.db_cursor = None

        self.regenerate_connection()

    def regenerate_connection(self):
        self.db_connection = self._connection

    @property
    def _connection(self):
        """
        :return: returns a connection object
        """
        # Establish a connection
        conn = psycopg2.connect(host=self.db_bindings["host"], dbname=self.db_bindings['databases']['pipeline'],
                                user=self.db_bindings['users'][self._user],
                                password=self.db_bindings['passwords'][self._user],
                                keepalives=1, keepalives_idle=30, keepalives_interval=10, keepalives_count=5)

        conn.autocommit = self.autocommit_db
        return conn

    @property
    def _db_cursor(self):
        """
        :returns: returns a cursor that can execute operations on the db
        """
        return self.db_connection.cursor()

    def commit(self):
        """
        :return: commit the changes to the db
        """
        return self.db_connection.commit()

    def rollback(self):
        """
        :return: commit the changes to the db
        """
        return self.db_connection.rollback()

    def get_urls_to_process(self, use_proxy=False):
        """
        Retrieve the list of urls to process
        :return:
        """
        if not use_proxy:
            return self._get_urls_to_process()
        else:
            return self._get_urls_rescan()

    def get_urls_to_process_guessed_cp(self, use_proxy=False):
        """
        Retrieve the list of urls to process
        :return:
        """
        if not use_proxy:
            return self._get_urls_analysis_guessed_cps()
        else:
            return self._get_urls_rescan_guessed_cps()

    def _get_urls_to_process(self):

        exceptions = []
        with open(EXCEPTIONS_FILE_PATH, 'r') as exceptions_file:
            for line in exceptions_file.read().split('\n'):
                domain = line.strip()
                if len(domain) > 0:
                    exceptions.append(domain)

        query = """
            with offline_netlocs as (
               select distinct all_urls.netloc as netloc
               from all_urls
               join url_headers using(uri)
               where url_headers.datetime_first_offline is null or (
                    url_headers.datetime_first_offline is not null and url_headers.offline_count < 3)
            ), distinct_todays_records as (
              select distinct all_urls.path[1:array_upper(all_urls.path, 1)-1] as path, netloc as netloc
              from all_urls
              join samplecluster using(filehash)
              join imported_samples using(filehash)
              left join hosting_services on all_urls.domain_remapped = hosting_services.provider
              left join cp_scanning on all_urls.netloc = cp_scanning.domain
              where (hosting_services.hosting_type is null
                    OR hosting_services.hosting_type = 'same_domain_free_webhosting')
                    and samplecluster.is_seo=true
                    and imported_samples.provider <> 'FromUrl'
                    and imported_samples.upload_date = current_date - interval '1 day'
                    and all_urls.guessed_cp is null
                    and cp_scanning.domain is null
                    and all_urls.netloc in (
                       select distinct offline_netlocs.netloc
                       from offline_netlocs
                    )
                    AND all_urls.has_http_scheme = True
                    AND all_urls.is_email = FALSE
                    AND all_urls.is_empty = FALSE
                    AND all_urls.is_relative_url = FALSE
                    AND all_urls.is_local = FALSE
                    AND all_urls.has_invalid_char = FALSE
                    AND all_urls.has_valid_tld = True
            )
            select distinct netloc
            from (
              select netloc, path, row_number() over (partition by path order by random() ) as rn
              from distinct_todays_records
            ) t
            where rn <= 10;
        """

        cur = self._db_cursor
        cur.execute(query)
        urls_to_process = [
            x[0] for x in cur.fetchall() if not any([e for e in exceptions if e in x[0]])
        ]

        return urls_to_process

    def _get_urls_rescan(self):

        exceptions = []
        with open(EXCEPTIONS_FILE_PATH, 'r') as exceptions_file:
            for line in exceptions_file.read().split('\n'):
                domain = line.strip()
                if len(domain) > 0:
                    exceptions.append(domain)

        query = """
            with offline_netlocs as (
               select distinct all_urls.netloc as netloc
               from all_urls
               join url_headers using(uri)
               where url_headers.datetime_first_offline is null or (
                    url_headers.datetime_first_offline is not null and url_headers.offline_count < 3)
            ), distinct_todays_records as (
              select distinct all_urls.path[1:array_upper(all_urls.path, 1)-1] as path, netloc as netloc
              from all_urls
              join samplecluster using(filehash)
              join imported_samples using(filehash)
              left join hosting_services on all_urls.domain_remapped = hosting_services.provider
              join cp_scanning on all_urls.netloc = cp_scanning.domain
              where (hosting_services.hosting_type is null
                    OR hosting_services.hosting_type = 'same_domain_free_webhosting')
                    and samplecluster.is_seo=true
                    and imported_samples.provider <> 'FromUrl'
                    and imported_samples.upload_date = current_date - interval '1 day'
                    and all_urls.guessed_cp is null
                    and all_urls.netloc in (
                       select distinct offline_netlocs.netloc
                       from offline_netlocs
                    )
                    AND all_urls.has_http_scheme = True
                    AND all_urls.is_email = FALSE
                    AND all_urls.is_empty = FALSE
                    AND all_urls.is_relative_url = FALSE
                    AND all_urls.is_local = FALSE
                    AND all_urls.has_invalid_char = FALSE
                    AND all_urls.has_valid_tld = True

                    AND cp_scanning.proxy = FALSE
                    AND cp_scanning.endpoints IS NULL
            )
            select distinct netloc
            from (
              select netloc, path, row_number() over (partition by path order by random() ) as rn
              from distinct_todays_records
            ) t
            where rn <= 10;
        """

        cur = self._db_cursor
        cur.execute(query)

        urls_to_process = [
            x[0] for x in cur.fetchall() if not any([e for e in exceptions if e in x[0]])
        ]

        return urls_to_process

    def insert(self, data):

        """
        Add a sample into the imported_sample table
        :return:
        """

        query = """
        INSERT INTO cp_scanning ( domain, timestamp, plugin, endpoints, iocs, proxy, path )
        VALUES %s
        ON CONFLICT DO NOTHING
        """

        with self._db_cursor as cur:
            execute_values(cur, sql=query, argslist=data,
                           template="( %(domain)s, %(timestamp)s, %(plugin)s, %(endpoints)s, %(iocs)s::text[], %(proxy)s, %(path)s::text[] )",
                           page_size=1000)
        return True

    def update(self, data):
        query = """
        UPDATE cp_scanning
        SET timestamp=data.timestamp, endpoints=data.endpoints, iocs=data.iocs::text[], proxy=data.proxy, path=data.path::text[]
        FROM ( VALUES %s ) AS data ( domain, timestamp, plugin, endpoints, iocs, proxy, path )
        WHERE cp_scanning.domain=data.domain AND cp_scanning.plugin = data.plugin AND cp_scanning.endpoints IS NULL
        """

        with self._db_cursor as cur:
            execute_values(cur, sql=query,
                           argslist=[(d['domain'], d['timestamp'], d['plugin'], d['endpoints'], d['iocs'], d['proxy'],
                                      d['path'])
                                     for d in data],
                           page_size=1000)
        return True

    def insert_guessed_cps(self, data):

        """
        Add a sample into the imported_sample table
        :return:
        """

        query = """
        INSERT INTO cp_scanning_guessed_cp ( domain, timestamp, plugin, endpoints, iocs, proxy, path )
        VALUES %s
        ON CONFLICT DO NOTHING
        """

        with self._db_cursor as cur:
            execute_values(cur, sql=query, argslist=data,
                           template="( %(domain)s, %(timestamp)s, %(plugin)s, %(endpoints)s, %(iocs)s::text[], %(proxy)s, %(path)s::text[] )",
                           page_size=1000)
        return True

    def update_guessed_cps(self, data):
        query = """
        UPDATE cp_scanning_guessed_cp
        SET timestamp=data.timestamp, endpoints=data.endpoints, iocs=data.iocs::text[], proxy=data.proxy, path=data.path::text[]
        FROM ( VALUES %s ) AS data ( domain, timestamp, plugin, endpoints, iocs, proxy, path )
        WHERE cp_scanning_guessed_cp.domain=data.domain AND cp_scanning_guessed_cp.plugin = data.plugin AND cp_scanning_guessed_cp.endpoints IS NULL
        """

        with self._db_cursor as cur:
            execute_values(cur, sql=query,
                           argslist=[(d['domain'], d['timestamp'], d['plugin'], d['endpoints'], d['iocs'], d['proxy'],
                                      d['path'])
                                     for d in data],
                           page_size=1000)
        return True

    def _get_urls_analysis_guessed_cps(self):

        exceptions = []
        with open(EXCEPTIONS_FILE_PATH, 'r') as exceptions_file:
            for line in exceptions_file.read().split('\n'):
                domain = line.strip()
                if len(domain) > 0:
                    exceptions.append(domain)

        query = """
            with offline_netlocs as (
               select distinct all_urls.netloc as netloc
               from all_urls
               join url_headers using(uri)
               where url_headers.datetime_first_offline is null or (
                    url_headers.datetime_first_offline is not null and url_headers.offline_count < 3)
            ), distinct_todays_records as (
              select distinct all_urls.path[1:array_upper(all_urls.path, 1)-1] as path, netloc as netloc
              from all_urls
              join samplecluster using(filehash)
              join imported_samples using(filehash)
              left join hosting_services on all_urls.domain_remapped = hosting_services.provider
              left join cp_scanning_guessed_cp on all_urls.netloc = cp_scanning_guessed_cp.domain
              where (hosting_services.hosting_type is null
                    OR hosting_services.hosting_type = 'same_domain_free_webhosting')
                    and samplecluster.is_seo=true
                    and imported_samples.provider <> 'FromUrl'
                    and imported_samples.upload_date = current_date - interval '1 day'
                    and all_urls.guessed_cp is not null
                    and cp_scanning_guessed_cp.domain is null
                    and all_urls.netloc in (
                       select distinct offline_netlocs.netloc
                       from offline_netlocs
                    )
                    AND all_urls.has_http_scheme = True
                    AND all_urls.is_email = FALSE
                    AND all_urls.is_empty = FALSE
                    AND all_urls.is_relative_url = FALSE
                    AND all_urls.is_local = FALSE
                    AND all_urls.has_invalid_char = FALSE
                    AND all_urls.has_valid_tld = True
            )
            select distinct netloc
            from (
              select netloc, path, row_number() over (partition by path order by random() ) as rn
              from distinct_todays_records
            ) t
            where rn <= 10;
        """

        cur = self._db_cursor
        cur.execute(query)
        urls_to_process = [
            x[0] for x in cur.fetchall() if not any([e for e in exceptions if e in x[0]])
        ]

        return urls_to_process

    def _get_urls_rescan_guessed_cps(self):

        exceptions = []
        with open(EXCEPTIONS_FILE_PATH, 'r') as exceptions_file:
            for line in exceptions_file.read().split('\n'):
                domain = line.strip()
                if len(domain) > 0:
                    exceptions.append(domain)

        query = """
            with offline_netlocs as (
               select distinct all_urls.netloc as netloc
               from all_urls
               join url_headers using(uri)
               where url_headers.datetime_first_offline is null or (
                    url_headers.datetime_first_offline is not null and url_headers.offline_count < 3)
            ), distinct_todays_records as (
              select distinct all_urls.path[1:array_upper(all_urls.path, 1)-1] as path, netloc as netloc
              from all_urls
              join samplecluster using(filehash)
              join imported_samples using(filehash)
              left join hosting_services on all_urls.domain_remapped = hosting_services.provider
              join cp_scanning_guessed_cp on all_urls.netloc = cp_scanning_guessed_cp.domain
              where (hosting_services.hosting_type is null
                    OR hosting_services.hosting_type = 'same_domain_free_webhosting')
                    and samplecluster.is_seo=true
                    and imported_samples.provider <> 'FromUrl'
                    and imported_samples.upload_date = current_date - interval '1 day'
                    and all_urls.guessed_cp is not null
                    and all_urls.netloc in (
                       select distinct offline_netlocs.netloc
                       from offline_netlocs
                    )
                    AND all_urls.has_http_scheme = True
                    AND all_urls.is_email = FALSE
                    AND all_urls.is_empty = FALSE
                    AND all_urls.is_relative_url = FALSE
                    AND all_urls.is_local = FALSE
                    AND all_urls.has_invalid_char = FALSE
                    AND all_urls.has_valid_tld = True

                    AND cp_scanning_guessed_cp.proxy = FALSE
                    AND cp_scanning_guessed_cp.endpoints IS NULL
            )
            select distinct netloc
            from (
              select netloc, path, row_number() over (partition by path order by random() ) as rn
              from distinct_todays_records
            ) t
            where rn <= 10;
        """

        cur = self._db_cursor
        cur.execute(query)

        urls_to_process = [
            x[0] for x in cur.fetchall() if not any([e for e in exceptions if e in x[0]])
        ]

        return urls_to_process

    @staticmethod
    def prepare_datalake(config, user, autocommit=True):
        """
        Use this function to prepare the Phishing dataset class given a configuration file
        Args:
            conf: dictionary containing the content of the configuration file

        Returns: instance of the PhishingDataLake class

        """

        # get from the config the path to the files of the phishing dataset
        phishing_entrypoint = config['global']['file_storage']

        # instantiate the PhishingDataLake class
        datalake = PhishingDataLake(phishing_entrypoint, config['global']['postgres'], user, autocommit_db=autocommit)
        return datalake


def file_hash_decode(hash, nested_levels=4):
    """
    Given the hash of a file in the phishing dataset, decode its path from the hash relative to the root
    directory:

        Args:
            hash: hash to process
            nested_levels: how many nested levels are encoded in the name

    E.g., if hash = "abcdefghilmnopq", this function returns "ab/cd/ef/gh/"
    """
    if len(hash) >= 2 * nested_levels:
        return "/".join([hash[2 * i:2 * (i + 1)] for i in range(0, nested_levels)]) + "/"
    raise Exception(
        "Relative file storage path cannot be determined because len({})={} < {} chars".format(hash, len(hash),
                                                                                               2 * nested_levels))
