import logging
from os.path import join
import datetime as dt
import psycopg2
from Data.Datalake.DataLake import DataLake

class PhishingDataLake(DataLake):

    def __init__(self, root_folder, db_bindings,screenshots_files_endpoint = None):
        """
        Args:
            root_folder: folder where the data are located
            db_bindings: credentials to connect to the db
            screenshots_files_endpoint: folder where the screenshots of the samples are located
        """
        super().__init__(root_folder)

        self.db_bindings = db_bindings

        self.screenshots_files_endpoint = screenshots_files_endpoint

    def get_benign_samples_list(self, random_sampling: bool = False, count: int = False, white_list_categories=None,force_different_phashes = False):
        """
        Get list of paths to benign samples

        Args:
            random_sampling: (bool,default False) sample random elements from the dataset
            count: (int,default False) maximum amount of elements to return
            white_list_categories: specify the campaign we want our samples to be from
            force_different_phashes: do not select multiple elements with the same p_hash.
        Returns: list of str each one of whom is a path to a benign sample
        """
        res = []

        # if not count has been specified, use ALL to retrieve all the rows
        if not count:
            count = "ALL"

        # if no category has been specified, sample data without this field
        # if a white list for categories has been specified then draw samples from just that categories
        categories = ""
        if white_list_categories:
            categories = " AND category IN ('" + "','".join(white_list_categories) + "')"

        # if the shuffle is requested, sample random records from the db
        order_by = ""
        if random_sampling:
            order_by = "ORDER BY Random()"

        query = None

        if force_different_phashes:

            query = f"""select * from (
                    SELECT distinct on(screens_phash) filehash,category FROM categories INNER JOIN codphish using(filehash)
                    WHERE malicious='No' AND category != 'AS PDF / File #12 SecureOpen' {categories}) p 
                     {order_by} LIMIT {count};
                    """
        else:
            query = f"""SELECT filehash,category FROM categories INNER JOIN codphish using(filehash)
                    WHERE malicious='No' AND category != 'AS PDF / File #12 SecureOpen' {categories}
                     {order_by} LIMIT {count};
                    """

        # get the hash of the benign pdfs from the db
        with self._db_cursor as cur:
            try:
                cur.execute(query, ())

                res = cur.fetchall()
            except psycopg2.errors.Error as e:
                res = []
                logging.warning(logging.ERROR, e)

        pdfs_list = [(join(self.root_folder, file_hash_decode(r[0]), r[0]), r[1]) for
                     r in res if r[0]]

        return pdfs_list

    def get_malicious_samples_list(self, random_sampling=False, count: int = False, white_list_categories=None,force_different_phashes = False):
        """
        Get list of paths to malicious samples

        Args:
            random_sampling: (bool,default False) sample random elements from the dataset
            count: (int,default False) maximum amount of elements to return
            white_list_categories: specify the campaign we want our samples to be from
            force_different_phashes: do not select multiple elements with the same p_hash.
        Returns: list of str each one of whom is a path to a malicious sample
        """
        res = []

        # if not count has been specified, use ALL to retrieve all the rows
        if not count:
            count = "ALL"

        # if no category has been specified, sample data without this field
        # if a white list for categories has been specified then draw samples from just that categories
        categories = ""
        if white_list_categories:
            categories = " AND category IN ('" + "','".join(white_list_categories) + "')"

        # if the shuffle is requested, sample random records from the db
        order_by = ""
        if random_sampling:
            order_by = "ORDER BY Random()"

        query = None

        if force_different_phashes:

            query = f"""SELECT * from (
                    SELECT distinct on(screens_phash) filehash,category FROM categories INNER JOIN codphish  using(filehash)
                    WHERE (malicious='yes' or malicious='--') {categories}) p
                    {order_by} LIMIT {count};
                    """
        else:
            query = f"""SELECT filehash,category FROM categories INNER JOIN codphish using(filehash)
                    WHERE (malicious='yes' or malicious='--') {categories}
                    {order_by} LIMIT {count};
                    """

        # get the hash of the benign pdfs from the db
        with self._db_cursor as cur:
            try:
                cur.execute(query, (count,))
                res = cur.fetchall()
            except psycopg2.errors.Error as e:
                res = []
                logging.warning(logging.ERROR, e)
        pdfs_list = [(join(self.root_folder, file_hash_decode(r[0]), r[0]), r[1]) for
                     r in res if r[0]]

        return pdfs_list

    def get_generic_samples_list(self, random_sampling=False, count: int = False,
                                 uploaded_from: dt.date = dt.date(2021, 6, 23), uploaded_to: dt.date = None,force_different_phashes = False):
        """
        Get list of paths to unlabelled samples

        Args:
            random_sampling: (bool,default False) sample random elements from the dataset
            count: (int,default False) maximum amount of elements to return
            uploaded_from: initial date from which we want to sample the files
            uploaded_to: final date from which we want to sample the files
            force_different_phashes: do not select multiple elements with the same p_hash.
        Returns: list of str each one of whom is a path to a malicious sample
        """
        res = []

        # if not count has been specified, use ALL to retrieve all the rows
        if not count:
            count = "ALL"

        # if the shuffle is requested, sample random records from the db
        order_by = ""
        if random_sampling:
            order_by = "ORDER BY Random()"

        if uploaded_from is not None:
            uploaded_from = f"AND upload_date>='{uploaded_from}'::date"

        if uploaded_to is not None:
            uploaded_to = f"AND upload_date<'{uploaded_to}'::date"

        query = None
        if force_different_phashes:
            query = f"""select * from (
                    SELECT distinct on(screens_phash) filehash,'unlabelled' FROM codphish left join categories using(filehash) WHERE codph_doc_type ='pdf'{uploaded_from} {uploaded_to} 
                    ) p {order_by} LIMIT {count};
                    """
        else:
            query = f"""SELECT filehash,'unlabelled' FROM codphish left join categories using(filehash) WHERE codph_doc_type ='pdf'{uploaded_from} {uploaded_to} 
                    {order_by} LIMIT {count};
                    """

        # get the hash of the benign pdfs from the db
        with self._db_cursor as cur:
            try:
                cur.execute(query, (count,))
                res = cur.fetchall()
            except psycopg2.errors.Error as e:
                res = []
                logging.warning(logging.ERROR, e)

        pdfs_list = [(join(self.root_folder, file_hash_decode(r[0]), r[0]), r[1]) for
                     r in res if r[0]]


        return pdfs_list

    def get_malicious_categories_list(self, min_count=0, preseleced=True):
        """
        Get the list of malicious categories names we can use to filter the files

        Args:
            :param min_count: minimum amount of elements a category must have to be returned
            :param preseleced: if preselected is true use the preselected_malicious_categories to
                return only categories that have been land labelled as  malicious
        Returns: list of strings
        """

        with self._db_cursor as cur:
            try:
                if preseleced:
                    cur.execute(f"""SELECT distinct category,count(*) FROM categories 
                    WHERE (malicious='yes' or malicious='--') AND category IN %s GROUP BY category having count(*)>%s 
                    ORDER BY Count(*) DESC;
                        """, (self.preselected_malicious_categories, min_count))
                    res = cur.fetchall()
                else:
                    cur.execute(f"""SELECT distinct category,count(*) FROM categories 
                    WHERE malicious='yes' or malicious='--' GROUP BY category having count(*)>%s 
                    ORDER BY Count(*) DESC;
                        """, (min_count,))
                    res = cur.fetchall()

            except psycopg2.errors.Error as e:
                res = []
                logging.log(logging.ERROR, e)

            return res

    def get_list_of_seo_categories(self):
        with self._db_cursor as cur:
            try:
                    cur.execute(f"""select id,name from campaigns c2  
                                    where id >=0 
                                    and name is not null and name <>''""")
                    res = cur.fetchall()
            except psycopg2.errors.Error as e:
                res = []
                logging.log(logging.ERROR, e)

            return res

    @property
    def preselected_malicious_categories(self):
        """
        Return the list of categories wqe have selected to be malicious by hand labelling
        :return:
        """
        l = ['>>> CLICK HERE <<<', 'ACCESS ONLINE GENERATOR', 'Adobe Click', 'Al Jaber Trading Qatar', 'Amazon scam',
             'American Express', 'Apple receipts', 'AS PDF / File #1', 'AS PDF / File #10',
             'AS PDF / File #11 Adobe pdfElement', 'AS PDF / File #13', 'AS PDF / File #2', 'AS PDF / File #3',
             'AS PDF / File #4', 'AS PDF / File #6', 'AS PDF / File #7 View Document Now',
             'AS PDF / File #8 Start Download', 'AS PDF / File #9', 'AS PDF / File Excel', 'Click Here TShirt',
             'Download Btn + image', 'Download File', 'Download PDF', 'Download PDF Blurred',
             'Download Torrent + GDrive (Russian)', 'Ebooks', 'Elon Musk BTC', 'Email Apple',
             'GENERATOR Button "Click Here"', 'Get Your Files WeTransfer', 'Google search-like', 'Link farm',
             'Lottery form', 'Lukoil', 'Netflix scam', 'NSFW "Click the picture"', 'NSFW "Find me by nick"',
             'NSFW "Play Button"', 'NSFW "Purpose of dating"', 'Play Video', 'QR code', 'reCAPTCHA', 'reCAPTCHA Drive',
             'ROBLOX GENERATOR (picture)', 'ROBLOX GENERATOR Text Only', 'Russian Forum', 'Russian Lottery 25th Ann',
             'Russian Lotto Girl', 'SharePoint Online', 'Sigue Leyendo', 'Try Your Luck Press (Russian)',
             'Web Notification', 'Weird nonsense crawler trap']

        return tuple(l)

    def __len__(self):
        all_files = self.get_benign_samples_list() + self.get_malicious_samples_list()
        return len(all_files)

    @property
    def _db_cursor(self):
        """
        Returns: returns a cursor that can execute operations on the db
        """
        # Establish a connection
        conn = psycopg2.connect(host=self.db_bindings["host"], dbname=self.db_bindings['database'],
                                user=self.db_bindings['user'],
                                password=self.db_bindings['password'])

        conn.autocommit = True
        return conn.cursor()


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


def prepare_phishing_datalake(config):
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

    # instantiate the ContagioDataset class
    datalake = PhishingDataLake(phishing_entrypoint, config['global']['providers']['phishing']['postgres'],phishing_screenshot_entrypoint)

    return datalake
