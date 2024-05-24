import logging
from os.path import join
import datetime as dt
import psycopg2
from psycopg2.extras import execute_values
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

    def get_urls_to_process(self):

        exceptions = []
        with open(EXCEPTIONS_FILE_PATH, 'r') as exceptions_file:
            for line in exceptions_file.read().split('\n'):
                domain = line.strip()
                if len(domain) > 0:
                    exceptions.append(domain)

        query = """
            select distinct netloc 
            from all_urls 
            join imported_samples using (filehash)
            where  ('repository'=any(path) or 'gudangsoal'=any(path) )  and all_urls.is_pdf = True AND 
                all_urls.has_http_scheme = True AND all_urls.is_email = FALSE AND all_urls.is_empty = FALSE AND 
                all_urls.is_relative_url = FALSE AND all_urls.is_local = FALSE AND all_urls.has_invalid_char = FALSE AND 
                all_urls.has_valid_tld = True and all_urls.guessed_cp is null AND
                imported_samples.provider <> 'FromUrl' AND imported_samples.upload_date = current_date - interval '1 day';
        """

        cur = self._db_cursor
        cur.execute(query)
        urls_to_process = [
            x[0] for x in cur.fetchall() if not any([e for e in exceptions if e in x[0]])
        ]

        return urls_to_process

    def update(self, data):
        query = """
        UPDATE all_urls
        SET guessed_cp = %s
        WHERE ('repository'=any(path) or 'gudangsoal'=any(path) )  and all_urls.is_pdf = True AND all_urls.has_http_scheme = True 
                AND all_urls.is_email = FALSE AND all_urls.is_empty = FALSE AND all_urls.is_relative_url = FALSE
                AND all_urls.is_local = FALSE AND all_urls.has_invalid_char = FALSE AND all_urls.has_valid_tld = True
                and all_urls.guessed_cp is null AND all_urls.netloc = %s;
        """

        with self._db_cursor as cur:
            cur.execute(query, (data[1], data[0]))
        return True


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
