import logging
from os.path import join
import datetime as dt
import psycopg2
from psycopg2.extras import Json
from Datalake.DataLake import DataLake
from urllib.parse import urlparse


class PhishingDataLake(DataLake):

    def __init__(self, root_folder, db_bindings, excluded_domains, screenshots_files_endpoint=None,thumbnail_screenshot_entrypoint=None,autocommit_db=True):
        """
        Args:
            root_folder: folder where the data are located
            db_bindings: credentials to connect to the db
            screenshots_files_endpoint: folder where the screenshots of the samples are located
        """
        self.db_bindings = db_bindings

        self.excluded_domains = excluded_domains

        self.screenshots_files_endpoint = screenshots_files_endpoint

        self.thumbnail_screenshot_entrypoint = thumbnail_screenshot_entrypoint

        self.root_folder = root_folder

        self.autocommit_db = autocommit_db

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
        conn = psycopg2.connect(host=self.db_bindings["host"], dbname=self.db_bindings['database'],
                                user=self.db_bindings['user'],
                                password=self.db_bindings['password'],keepalives=1, keepalives_idle=30, keepalives_interval=10,
                                keepalives_count=5)

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

    def get_urls_to_process(self):
        """
        Retrieve the list of urls to process
        :return:
        """
        # Create query to get all the samples to process. We want:
        # 1) Only valid and unique uri
        # 2) Only uri that from less than 6 months ago not yet marked as offline for more than 3 times
        query = """SELECT DISTINCT all_urls.uri, (uh.filehash_pdf is Null) as first_visit
                    from all_urls
                    inner join imported_samples i on all_urls.filehash = i.filehash
                    inner join samplecluster s on i.filehash = s.filehash
                    LEFT JOIN url_headers uh on all_urls.uri = uh.uri
                    where
                        i.provider <> 'FromUrl'
                        AND i.upload_date > '2022/06/19'::date
                        AND all_urls.is_pdf = True
                        AND all_urls.has_http_scheme = True
                        AND all_urls.is_email = FALSE
                        AND all_urls.is_empty = FALSE
                        AND all_urls.is_relative_url = FALSE
                        AND all_urls.is_local = FALSE
                        AND all_urls.has_invalid_char = FALSE
                        AND all_urls.has_valid_tld = True
                        AND (offline_count is NULL or offline_count < 3)
                        AND s.is_seo = True
                        AND (uh.last_updated is NULL or DATE(uh.last_updated) < DATE(NOW()))
                        """

        cur = self._db_cursor
        cur.execute(query)
        urls_to_process = [x for x in cur.fetchall() if not any([e for e in self.excluded_domains if e in urlparse(x[0]).netloc]) ]

        return urls_to_process

    def get_only_urls_to_process(self):
        """
        Retrieve the list of urls to process
        :return:
        """
        # Create query to get all the samples to process. We want:
        # 1) Only valid and unique uri
        # 2) Only uri that from less than 6 months ago not yet marked as offline for more than 3 times
        query = """SELECT DISTINCT all_urls.uri
                    from all_urls
                    inner join imported_samples i on all_urls.filehash = i.filehash
                    inner join samplecluster s on i.filehash = s.filehash
                    LEFT JOIN url_headers uh on all_urls.uri = uh.uri
                    where
                        i.provider <> 'FromUrl'
                        AND i.upload_date > '2022/06/15'::date
                        AND all_urls.is_pdf = True
                        AND all_urls.has_http_scheme = True
                        AND all_urls.is_email = FALSE
                        AND all_urls.is_empty = FALSE
                        AND all_urls.is_relative_url = FALSE
                        AND all_urls.is_local = FALSE
                        AND all_urls.has_invalid_char = FALSE
                        AND all_urls.has_valid_tld = True
                        AND (offline_count is NULL or offline_count < 3)
                        AND s.is_seo = True
                        AND (uh.last_updated is NULL or DATE(uh.last_updated) < DATE(NOW()))
                        """

        cur = self._db_cursor
        cur.execute(query)
        urls_to_process = cur.fetchall()

        return urls_to_process

    def add_imported_sample(self, filehash, mimetype, timestamp, provider, upload_date, original_filename):
        """
        Add a sample into the imported_sample table
        :return:
        """

        query = """
        INSERT INTO imported_samples(filehash,mimetype,timestamp,provider,upload_date,original_filename)
        SELECT %s,%s,%s,%s,%s,%s
        WHERE NOT EXISTS (SELECT 1 
                FROM imported_samples 
                WHERE filehash = %s
                  AND provider = 'FromUrl')
        """

        with self._db_cursor as cur:
            cur.execute(query, (filehash, mimetype, timestamp, provider, upload_date, original_filename,filehash))

        return True

    def save_online_url_results(self, uri, header_first_online, datetime_first_online, filehash_pdf, status_code_online,
                                content_type):
        query = """
            INSERT INTO url_headers(uri,header_first_online,datetime_first_online,filehash_pdf,status_code_online,
            content_type_online,last_updated)
            VALUES(%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (uri) DO UPDATE 
            SET
            header_first_online = COALESCE(url_headers.header_first_online,excluded.header_first_online),
            filehash_pdf = COALESCE(url_headers.filehash_pdf,excluded.filehash_pdf),
            status_code_online = COALESCE(url_headers.status_code_online,excluded.status_code_online),
            content_type_online = COALESCE(url_headers.content_type_online,excluded.content_type_online),
            datetime_first_online = COALESCE(url_headers.datetime_first_online,excluded.datetime_first_online),
            offline_count = 0,
            header_first_offline = NULL, 
            datetime_first_offline = NULL,
            status_code_offline = NULL,
            error_msg = NULL,
            content_type_offline = NULL,
            amz = NULL,
            last_updated = excluded.last_updated
        """

        with self._db_cursor as cur:
            cur.execute(query, (
            uri, Json(header_first_online), datetime_first_online, filehash_pdf, status_code_online, content_type,datetime_first_online))

        return True

    def save_offline_url_results(self, uri, header_first_offline, datetime_first_offline, status_code_offline,content_type_offline,
                                 error_msg,amz):
        query = """
            INSERT INTO url_headers(uri,offline_count,header_first_offline,datetime_first_offline,status_code_offline,content_type_offline,error_msg,amz,last_updated)
            VALUES(%s,1,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (uri) DO UPDATE 
            SET offline_count = url_headers.offline_count + 1,
            header_first_offline = coalesce(url_headers.header_first_offline,excluded.header_first_offline), 
            status_code_offline = coalesce(url_headers.status_code_offline,excluded.status_code_offline),
            content_type_offline = coalesce(url_headers.content_type_offline,excluded.content_type_offline),
            datetime_first_offline = coalesce(url_headers.datetime_first_offline,excluded.datetime_first_offline),
            amz = coalesce(url_headers.amz,excluded.amz),
            error_msg = coalesce(url_headers.error_msg,excluded.error_msg),
            last_updated = excluded.last_updated
        """

        #if header_first_offline:
        #    header_first_offline = Json(header_first_offline)

        with self._db_cursor as cur:
            cur.execute(query, (uri, Json(header_first_offline), datetime_first_offline, status_code_offline, content_type_offline, error_msg,amz,datetime_first_offline))

        return True

    @staticmethod
    def prepare_datalake(config,autocommit=True):
        """
        Use this function to prepare the Phishing dataset class given a configuration file
        Args:
            conf: dictionary containing the content of the configuration file

        Returns: instance of the ContagioDataset class

        """

        # get from the config the path to the files of the phishing dataset
        phishing_entrypoint = config['global']['providers']['phishing']['file_storage']

        # get from the config the path to the screenshots of the phishing dataset
        phishing_screenshot_entrypoint = config['global']['providers']['phishing']['screenshot_storage']

        # get from the config the path to the screenshots of the phishing dataset
        thumbnail_screenshot_entrypoint = config['global']['providers']['phishing']['thumbnail_storage']

        # load excluded domains from external file
        exceptions = []
        with open(config['global']['providers']['phishing']['excluded_domains'], 'r') as exceptions_file:
            for line in exceptions_file.read().split('\n'):
                domain = line.strip()
                if len(domain) > 0:
                    exceptions.append(domain)

        # instantiate the ContagioDataset class
        datalake = PhishingDataLake(phishing_entrypoint, config['global']['providers']['phishing']['postgres'], exceptions,
                                    phishing_screenshot_entrypoint,thumbnail_screenshot_entrypoint,autocommit_db=autocommit)

        return datalake

    def get_urls_blocklisted(self):
        """
        Return the list of blocklisted urls
        :return: list of urls blocklisted
        """
        select_query = """SELECT DISTINCT url FROM url_blocklisted"""

        with self._db_cursor as cur:

            cur.execute(select_query,)
            return cur.fetchall()
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
