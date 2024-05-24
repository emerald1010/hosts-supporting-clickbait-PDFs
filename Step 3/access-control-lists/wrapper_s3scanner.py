import argparse
import os
import yaml
import psycopg2
from psycopg2.extras import Json, execute_values
from datetime import datetime, timedelta
import pandas as pd
from urllib.parse import urlparse
import re
from sys import exit
import time
import requests
import xml.etree.ElementTree as ET

from S3Scanner.S3Bucket import S3Bucket, BucketExists, Permission
from S3Scanner.S3Service import S3Service
from concurrent.futures import ThreadPoolExecutor, as_completed
from S3Scanner.exceptions import InvalidEndpointException, BucketMightNotExistException
from botocore.exceptions import ClientError

import log


CURRENT_VERSION = '2.0.2'
AWS_ENDPOINT = 'https://s3.amazonaws.com'


### ADAPTED CODE FROM https://github.com/sa7mon/S3Scanner/



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
                time.sleep(5)

                try:
                    self.conn = psycopg2.connect(dbname=self.database, user=self.user, password=self.password,
                                                 host=self.host, port=self.port,
                                                 keepalives=1, keepalives_idle=30, keepalives_interval=10,
                                                 keepalives_count=5)
                    self.conn.autocommit = self.autocommit
                except Exception as e:
                    __LOGGER__.exception(e)
                    exit(-1)
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


# We want to use both formatter classes, so a custom class it is
class CustomFormatter(argparse.RawTextHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def load_config_yaml(conf_fname):
    with open(conf_fname) as s:
        config = yaml.safe_load(s)
    return config


def get_s3_region(url):
    _, netloc, _, _, _, _ = urlparse(url)
    if netloc == 's3.amazonaws.com':
        region = 'us-east-1'
    else:
        s3_index = netloc.find('.s3')
        aws_domain_index = netloc.find('.amazonaws')
        if netloc[s3_index + 4 : aws_domain_index] == '':
            region = 'us-east-1'
        else:
            region = netloc[s3_index + 4 : aws_domain_index]

    return region

def get_s3_bucket_name(url):
        """
        Checks to make sure bucket names input are valid according to S3 naming conventions
        :param str name: Name of bucket to check
        :return: dict: ['valid'] - bool: whether or not the name is valid, ['name'] - str: extracted bucket name
        """

        _, netloc, path, _, _, _ = urlparse(url)

        bucket_name = ""
        # Check if bucket name is valid and determine the format
        if ".amazonaws.com" in netloc and '.s3' in netloc:    # We were given a full s3 url
            bucket_name = netloc[:netloc.rfind(".s3")]
        elif ":" in netloc:               # We were given a bucket in 'bucket:region' format
            bucket_name = netloc.split(":")[0]
        else:                               # We were given a regular bucket name
            bucket_name = path.split('/')[1]

        # Bucket names can be 3-63 (inclusively) characters long.
        # Bucket names may only contain lowercase letters, numbers, periods, and hyphens
        pattern = r'(?=^.{3,63}$)(?!^(\d+\.)+\d+$)(^(([a-z0-9]|[a-z0-9][a-z0-9\-]*[a-z0-9])\.)*([a-z0-9]|[a-z0-9][a-z0-9\-]*[a-z0-9])$)'
        if re.match(pattern, bucket_name):
            return bucket_name
        else:
            return None

def get_do_bucket_name(url):
    # The Spaces URL naming pattern is spacename.region.digitaloceanspaces.com and region.digitaloceanspaces.com/spacename, where spacename is the name of your Space and region is the region your Space is in.
    # https://docs.digitalocean.com/products/spaces/how-to/create/

    _, netloc, path, _, _, _ = urlparse(url)

    splits = netloc.split('.')
    if len(splits) == 4:
        bucket_name = splits[0]
    else:
        bucket_name = path.split('/')[1]

    return bucket_name

def get_do_region(url):
    # The Spaces URL naming pattern is spacename.region.digitaloceanspaces.com and region.digitaloceanspaces.com/spacename, where spacename is the name of your Space and region is the region your Space is in.
    # https://docs.digitalocean.com/products/spaces/how-to/create/

    _, netloc, _, _, _, _ = urlparse(url)
    return netloc.split('.')[-3].lower()

def get_alibaba_b_name_region(url):
    _, netloc, path, _, _, _ = urlparse(url)

    splits = netloc.split('.')

    if len(splits) == 4 and splits[2] == 'aliyuncs' and splits[3] == 'com':
        region = splits[1]
        name = splits[0]
        return name, region
    else:
        return None, None

def xml_to_json(tree):
    objects = {}

    for node in tree:

        subtree = list(node)
        if len(subtree) > 0:
            node_value = xml_to_json(subtree)
        else:
            node_value = node.text or ''

        if node.tag in objects.keys():
            if not isinstance(objects[node.tag], list):
                previous_values = objects.pop(node.tag)
                objects[node.tag] = [previous_values]
            else:
                objects[node.tag].append(node_value)
        else:
            objects[node.tag] = node_value

    return objects


def scan_single_bucket(s3service, anons3service, bucket_name, checkPermWrite, checkPermWriteAcl=False):
    """
    Scans a single bucket for permission issues. Exists on its own so we can do multi-threading
    :param S3Service s3service: S3Service with credentials to use for scanning
    :param S3Service anonS3Service: S3Service without credentials to use for scanning
    :param bool do_dangerous: Whether or not to do dangerous checks
    :param str bucket_name: Name of bucket to check
    :return: None
    """
    try:
        b = S3Bucket(bucket_name)
    except ValueError as ve:
        if str(ve) == "Invalid bucket name":
            print(f"{bucket_name} | bucket_invalid_name")
        else:
            print(f"{bucket_name} | {str(ve)}")
        raise


    # Check if bucket exists first
    # Use credentials if configured and if we're hitting AWS. Otherwise, check anonymously
    if s3service.endpoint_url == AWS_ENDPOINT:
        s3service.check_bucket_exists(b)
    else:
        anons3service.check_bucket_exists(b)

    if b.exists == BucketExists.NO:
        return b.name, datetime.now(), None, None, None, False
    checkAllUsersPerms = True
    checkAuthUsersPerms = True


    try:
        # 1. Check for ReadACP
        anons3service.check_perm_read_acl(b)  # Check for AllUsers
        if s3service.aws_creds_configured and s3service.endpoint_url == AWS_ENDPOINT:
            s3service.check_perm_read_acl(b)  # Check for AuthUsers

        # If FullControl is allowed for either AllUsers or AnonUsers, skip the remainder of those tests
        if b.AuthUsersFullControl == Permission.ALLOWED:
            checkAuthUsersPerms = False
        if b.AllUsersFullControl == Permission.ALLOWED:
            checkAllUsersPerms = False

        # 2. Check for Read
        if checkAllUsersPerms:
            anons3service.check_perm_read(b)
        if s3service.aws_creds_configured and checkAuthUsersPerms and s3service.endpoint_url == AWS_ENDPOINT:
            s3service.check_perm_read(b)

        # Do dangerous/destructive checks
        if checkPermWrite:
            # 3. Check for Write
            if checkAllUsersPerms:
                anons3service.check_perm_write(b)
            if s3service.aws_creds_configured and checkAuthUsersPerms:
                s3service.check_perm_write(b)

        if checkPermWriteAcl:
            # 4. Check for WriteACP
            if checkAllUsersPerms:
                anons3service.check_perm_write_acl(b)
            if s3service.aws_creds_configured and checkAuthUsersPerms:
                s3service.check_perm_write_acl(b)
    except (ClientError, BucketMightNotExistException) as e:
        __LOGGER__.exception(e)
        __LOGGER__.info(b.name)
        return b.name, datetime.now(), b.get_db_writable_permissions(), b.foundACL, b.objects_listed, False

    return b.name, datetime.now(), b.get_db_writable_permissions(), b.foundACL, b.objects_listed, True

def scan_s3_buckets(bucketsIn, checkPermWrite, n_threads):

    s3service = None
    anons3service = None
    try:
        s3service = S3Service() # rely on defaults
        anons3service = S3Service(forceNoCreds=True)
    except InvalidEndpointException as e:
        __LOGGER__.exception(e)
        exit(1)

    if s3service.aws_creds_configured is False:
        print("Warning: AWS credentials not configured - functionality will be limited. Run:"
              " `aws configure` to fix this.\n")

    res = []
    for bucket_name in bucketsIn:
        try:
            res.append(
                scan_single_bucket(s3service, anons3service, bucket_name, checkPermWrite)
            )
        except Exception as e:
            __LOGGER__.exception(e)
            __LOGGER__.info(bucket_name)
            continue

    return res

def scan_digitalocean_buckets(to_inspect, checkPermWrite, n_threads):
    res = []
    for bucket_name, region in to_inspect.itertuples(index=False):
        anons3service = None
        try:
            endpoint_url = f'https://{region}.digitaloceanspaces.com'
            anons3service = S3Service(endpoint_url=endpoint_url, forceNoCreds=True)
        except InvalidEndpointException as e:
            __LOGGER__.exception(e)
            exit(1)

        try:
            res.append(
                # workaround relying on the checks on `aws_creds_configured`
                scan_single_bucket(anons3service, anons3service, bucket_name, checkPermWrite)
            )
        except Exception as e:
            __LOGGER__.exception(e)
            __LOGGER__.info(bucket_name)
            continue

    return res

def try_listdir(pdf_link_inspect, pdf_link_se):

    if pdf_link_inspect and not pd.isna(pdf_link_inspect):
        pdf_link = pdf_link_inspect
    elif pdf_link_se and not pd.isna(pdf_link_se):
        pdf_link = pdf_link_se
    else:
        raise ValueError('Arguments are None!')

    scheme, netloc, path, _, _, _ = urlparse(pdf_link)
    if get_do_bucket_name(pdf_link) in netloc:
        endpoint = scheme + '://' + netloc
    else:
        endpoint = scheme + '://' + netloc + '/' + get_do_bucket_name(pdf_link)

    try:
        resp = requests.get(endpoint)

        if resp.status_code != 200:
            return None

        return xml_to_json(
            ET.fromstring(resp.text)
        )

    except Exception as e:
        __LOGGER__.info(e)
        __LOGGER__.info(endpoint)
        return None



def mark_se_buckets_in_vt(db_bindings, aws_urls, do_urls):
    db_wrapper = DbWrapper(db_bindings, 'pipeline', 'user', True)
    cur = db_wrapper.get_cursor()
    try:
        cur.execute("""
        SELECT bucket_name, bucket_region, service
        FROM s3scanner
        WHERE from_seo=true AND first_seen_vt_feed IS NULL;
        """)
        se_only_buckets = cur.fetchall()
    except psycopg2.errors.OperationalError as e:
        __LOGGER__.exception(e)
        __LOGGER__.info('Retrying select...')
        db_wrapper.release_cursor()
        cur = db_wrapper.get_cursor()

        try:
            cur.execute("""
            SELECT bucket_name, bucket_region, service
            FROM s3scanner
            WHERE from_seo=true AND first_seen_vt_feed IS NULL;
            """)
            se_only_buckets = cur.fetchall()
        except psycopg2.errors.Error as e:
            __LOGGER__.exception(e)
            se_only_buckets = []
    except psycopg2.errors.Error as e:
        __LOGGER__.exception(e)
        se_only_buckets = []
    db_wrapper.release_cursor()

    scanned_se_buckets = pd.DataFrame(se_only_buckets, columns=['bucket_name', 'bucket_region', 'service'])
    urls_in_vt_feed_aws = scanned_se_buckets[scanned_se_buckets.service=='S3'].merge(
        aws_urls, how='inner', on=['bucket_name', 'bucket_region']
    )
    urls_in_vt_feed_do = scanned_se_buckets[scanned_se_buckets.service=='DigitalOcean'].merge(
        do_urls, how='inner', on=['bucket_name', 'bucket_region']
    )

    to_update = pd.concat([urls_in_vt_feed_aws, urls_in_vt_feed_do], ignore_index=True)
    if not to_update.empty:
        to_update.drop_duplicates(['bucket_name', 'bucket_region', 'service'], inplace=True)
        to_update['first_seen_vt_feed'] = [datetime.today()] * to_update.shape[0]

        __LOGGER__.info(f'{to_update.shape[0]} SE buckets are in VTs feed today. Updating...')

        update_stmt = """
            UPDATE s3scanner
            SET first_seen_vt_feed = data.first_seen_vt_feed
            FROM (VALUES %s) AS data (bucket_name, bucket_region, service, first_seen_vt_feed)
            WHERE s3scanner.bucket_name = data.bucket_name AND s3scanner.bucket_region = data.bucket_region AND s3scanner.service = data.service ;
        """

        db_wrapper = DbWrapper(db_bindings, 'pipeline', 'user', True)
        cur = db_wrapper.get_cursor()

        try:
            execute_values(cur, update_stmt, to_update.to_dict(orient='records'),
                           template="( %(bucket_name)s, %(bucket_region)s, %(service)s, %(first_seen_vt_feed)s )",
                           page_size=500)
        except psycopg2.errors.OperationalError as e:
            __LOGGER__.exception(e)
            __LOGGER__.info('Retrying insert...')
            db_wrapper.release_cursor()
            cur = db_wrapper.get_cursor()

            try:
                execute_values(cur, update_stmt, to_update.to_dict(orient='records'),
                               template="(  %(bucket_name)s, %(bucket_region)s, %(service)s, %(first_seen_vt_feed)s )",
                               page_size=500)
            except psycopg2.errors.Error as e:
                __LOGGER__.exception(e)
                db_wrapper.release_cursor()
                exit(-1)

        except psycopg2.errors.Error as e:
            __LOGGER__.exception(e)
            db_wrapper.release_cursor()
            exit(-1)
        else:
            __LOGGER__.debug("Insert completed successfully.")

        db_wrapper.release_cursor()
    else:
        __LOGGER__.info('No previously seen SE buckets found in VT feed today.')

    return





def select_urls(db_bindings, when):
    db_wrapper = DbWrapper(db_bindings, 'pipeline', 'user', True)
    cur = db_wrapper.get_cursor()
    try:

        cur.execute("""
            SELECT DISTINCT all_urls.uri
            FROM all_urls
            JOIN imported_samples i USING(filehash)
            JOIN samplecluster using(filehash)
            WHERE
                i.provider <> 'FromUrl'
                AND samplecluster.is_seo = True
                AND i.upload_date = %s
                AND all_urls.is_pdf = True
                AND all_urls.has_http_scheme = True
                AND all_urls.is_email = FALSE
                AND all_urls.is_empty = FALSE
                AND all_urls.is_relative_url = FALSE
                AND all_urls.is_local = FALSE
                AND all_urls.has_invalid_char = FALSE
                AND all_urls.has_valid_tld = True;
        """, (when,))
        all_urls = cur.fetchall()

        cur.execute("""
        SELECT DISTINCT url
        FROM extended_pdf_serp_bing
        WHERE timestamp > %s;
        """, (when, ))
        urls_from_se = cur.fetchall()
        cur.execute("""
        SELECT DISTINCT url
        FROM extended_pdf_serp_google
        WHERE timestamp > %s;
        """, (when, ))
        urls_from_se.extend(cur.fetchall())



        cur.execute("""
            SELECT bucket_name, bucket_region, service
            FROM s3scanner;
        """)
        scanned_buckets = cur.fetchall()

    except psycopg2.errors.OperationalError as e:
        __LOGGER__.exception(e)
        __LOGGER__.info('Retrying select...')
        db_wrapper.release_cursor()
        cur = db_wrapper.get_cursor()

        try:

            cur.execute("""
                SELECT DISTINCT all_urls.uri
                FROM all_urls
                JOIN imported_samples i USING(filehash)
                join samplecluster using(filehash)
                WHERE
                    i.provider <> 'FromUrl'
                    AND samplecluster.is_seo=True
                    AND i.upload_date = %s
                    AND all_urls.is_pdf = True
                    AND all_urls.has_http_scheme = True
                    AND all_urls.is_email = FALSE
                    AND all_urls.is_empty = FALSE
                    AND all_urls.is_relative_url = FALSE
                    AND all_urls.is_local = FALSE
                    AND all_urls.has_invalid_char = FALSE
                    AND all_urls.has_valid_tld = True;
            """, (when,))
            all_urls = cur.fetchall()

            cur.execute("""
            SELECT DISTINCT url
            FROM extended_pdf_serp_bing
            WHERE timestamp > %s;
            """, (when, ))
            urls_from_se = cur.fetchall()
            cur.execute("""
            SELECT DISTINCT url
            FROM extended_pdf_serp_google
            WHERE timestamp > %s;
            """, (when,))
            urls_from_se.extend(cur.fetchall())

            cur.execute("""
                SELECT bucket_name, bucket_region, service
                FROM s3scanner;
            """)
            scanned_buckets = cur.fetchall()
        except psycopg2.errors.Error as e:
            __LOGGER__.exception(e)
            all_urls = []
            scanned_buckets = []
            urls_from_se = []

    except psycopg2.errors.Error as e:
        __LOGGER__.exception(e)
        all_urls = []
        scanned_buckets = []
        urls_from_se = []
    db_wrapper.release_cursor()
    
    return all_urls, urls_from_se, scanned_buckets

def insert(db_bindings, data):
    insert_stmt = """
        INSERT INTO s3scanner ( bucket_name, timestamp, acl, raw_acl, listobject_resp, exists, bucket_region, service, listdir_content, from_seo )
        VALUES %s;
    """

    db_wrapper = DbWrapper(db_bindings, 'pipeline', 'user', True)
    cur = db_wrapper.get_cursor()

    try:
        execute_values(cur, insert_stmt, data.to_dict(orient='records'),
                       template="( %(bucket_name)s, %(timestamp)s, %(acl)s, %(raw_acl)s, %(listobject_resp)s, %(exists)s, %(bucket_region)s, %(service)s, %(listdir_content)s, %(from_seo)s )",
                       page_size=500)
    except psycopg2.errors.OperationalError as e:
        __LOGGER__.exception(e)
        __LOGGER__.info('Retrying insert...')
        db_wrapper.release_cursor()
        cur = db_wrapper.get_cursor()

        try:
            execute_values(cur, insert_stmt, data.to_dict(orient='records'),
                           template="( %(bucket_name)s, %(timestamp)s, %(acl)s, %(raw_acl)s, %(listobject_resp)s, %(exists)s, %(bucket_region)s, %(service)s, %(listdir_content)s, %(from_seo)s )",
                           page_size=500)
        except psycopg2.errors.Error as e:
            __LOGGER__.exception(e)
            db_wrapper.release_cursor()
            exit(-1)

    except psycopg2.errors.Error as e:
        __LOGGER__.exception(e)
        db_wrapper.release_cursor()
        exit(-1)
    else:
        __LOGGER__.debug("Insert completed successfully.")

    db_wrapper.release_cursor()




def main(db_bindings, workers):
    yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    all_urls, urls_from_se, scanned_buckets = select_urls(db_bindings, yesterday)

    df_scanned_buckets = pd.DataFrame(scanned_buckets, columns=['bucket_name', 'bucket_region', 'service'])

    aliyun_urls = pd.DataFrame(
        [x for x in all_urls if 'aliyuncs.com' in x[0]]
        , columns=['url'])
    bucket_name_region = aliyun_urls.apply(lambda x: get_alibaba_b_name_region(x['url']), axis=1, result_type='expand')
    try:
        aliyun_urls[['bucket_name', 'bucket_region']] = bucket_name_region
    except ValueError:
        if bucket_name_region.empty and aliyun_urls.empty:
            aliyun_urls['bucket_name'] = pd.Series(dtype='object')
            aliyun_urls['bucket_region'] = pd.Series(dtype='object')
        else:
            raise
    aliyun_urls.drop_duplicates('bucket_name', keep='first', inplace=True)

    aliyun_se_urls = pd.DataFrame([x for x in urls_from_se if 'aliyuncs.com' in x[0]]
        , columns=['url'])
    aliyun_se_urls['bucket_name'] = aliyun_se_urls.url.apply(lambda x: get_s3_bucket_name(x))
    aliyun_se_urls.drop_duplicates('bucket_name', keep='first', inplace=True)
    aliyun_se_urls['bucket_region'] = aliyun_se_urls.url.apply(lambda x: get_s3_region(x))
    aliyun_se_urls['from_seo'] = [True] * aliyun_se_urls.shape[0]


    aliyun_to_inspect = aliyun_urls\
        .merge(aliyun_se_urls[['bucket_name', 'bucket_region', 'from_seo', 'url']], how='outer', on=['bucket_name', 'bucket_region'], suffixes=['_inspect', '_se'], indicator='i')\
        .query('i == "left_only" or i == "right_only"')\
        .merge(df_scanned_buckets[df_scanned_buckets.service=='Alibaba'][['bucket_name', 'bucket_region']], how='left', on=['bucket_name', 'bucket_region'], indicator='i2')\
        .query('i2 == "left_only"')\
        .copy()

    if aliyun_to_inspect.empty:
        __LOGGER__.debug('No new Alibaba bucket to scan today...')
        aliyun_df = pd.DataFrame()
    else:
        __LOGGER__.info(f'{aliyun_to_inspect[aliyun_to_inspect.from_seo == True].shape[0]} SEO buckets observed:\n{aliyun_to_inspect[aliyun_to_inspect.from_seo == True][["url_inspect", "bucket_name", "bucket_region", "url_se", "from_seo"]]}')

        aliyun_to_inspect['service'] = ['Alibaba'] * aliyun_to_inspect.shape[0]
        aliyun_to_inspect['from_seo'] = aliyun_to_inspect.apply(lambda x: False if pd.isna(x['from_seo']) else True, axis=1)

        __LOGGER__.info(f'Monitoring quota: going to scan {aliyun_to_inspect.shape[0]} buckets!')
        aliyun_scan_results = scan_s3_buckets(aliyun_to_inspect.bucket_name.to_list(), True, workers)
        try:
            listdir_content = aliyun_to_inspect.apply(lambda x: try_listdir(x['url_inspect'], x['url_se']), axis=1)
            aliyun_to_inspect['listdir_content'] = listdir_content.apply(lambda x: Json(x))
        except ValueError as e:
            __LOGGER__.exception(e)
            aliyun_to_inspect['listdir_content'] = aliyun_to_inspect.shape[0] * [Json({})]

        aliyun_df = pd.DataFrame(aliyun_scan_results, columns=['bucket_name', 'timestamp', 'tmp_acl', 'tmp_raw_acl', 'tmp_listobject_resp', 'exists'])\
        .merge(aliyun_to_inspect[['bucket_name', 'bucket_region', 'service', 'listdir_content', 'from_seo']], how='left', on='bucket_name')


    google_urls = pd.DataFrame(
        [x for x in all_urls if 'storage.googleapis.com' in x[0]]
        , columns=['url'])
    google_urls['bucket_name'] = google_urls.url.apply(lambda x: get_s3_bucket_name(x))
    google_urls.drop_duplicates('bucket_name', keep='first', inplace=True)
    google_urls['bucket_region'] = google_urls.url.apply(lambda x: get_s3_region(x))

    google_se_urls = pd.DataFrame([x for x in urls_from_se if 'storage.googleapis.com' in x[0]]
        , columns=['url'])
    google_se_urls['bucket_name'] = google_se_urls.url.apply(lambda x: get_s3_bucket_name(x))
    google_se_urls.drop_duplicates('bucket_name', keep='first', inplace=True)
    google_se_urls['bucket_region'] = google_se_urls.url.apply(lambda x: get_s3_region(x))
    google_se_urls['from_seo'] = [True] * google_se_urls.shape[0]


    google_to_inspect = google_urls\
        .merge(google_se_urls[['bucket_name', 'bucket_region', 'from_seo', 'url']], how='outer', on=['bucket_name', 'bucket_region'], suffixes=['_inspect', '_se'], indicator='i')\
        .query('i == "left_only" or i == "right_only"')\
        .merge(df_scanned_buckets[df_scanned_buckets.service=='Google'][['bucket_name', 'bucket_region']], how='left', on=['bucket_name', 'bucket_region'], indicator='i2')\
        .query('i2 == "left_only"')\
        .copy()

    if google_to_inspect.empty:
        __LOGGER__.debug('No new Google bucket to scan today...')
        google_df = pd.DataFrame()
    else:
        __LOGGER__.info(f'{google_to_inspect[google_to_inspect.from_seo == True].shape[0]} SEO buckets observed:\n{google_to_inspect[google_to_inspect.from_seo == True][["url_inspect", "bucket_name", "bucket_region", "url_se", "from_seo"]]}')

        google_to_inspect['service'] = ['Google'] * google_to_inspect.shape[0]
        google_to_inspect['from_seo'] = google_to_inspect.apply(lambda x: False if pd.isna(x['from_seo']) else True, axis=1)

        __LOGGER__.info(f'Monitoring quota: going to scan {google_to_inspect.shape[0]} buckets!')
        google_scan_results = scan_s3_buckets(google_to_inspect.bucket_name.to_list(), True, workers)
        try:
            listdir_content = google_to_inspect.apply(lambda x: try_listdir(x['url_inspect'], x['url_se']), axis=1)
            google_to_inspect['listdir_content'] = listdir_content.apply(lambda x: Json(x))
        except ValueError as e:
            __LOGGER__.exception(e)
            google_to_inspect['listdir_content'] = google_to_inspect.shape[0] * [Json({})]

        google_df = pd.DataFrame(google_scan_results, columns=['bucket_name', 'timestamp', 'tmp_acl', 'tmp_raw_acl', 'tmp_listobject_resp', 'exists'])\
        .merge(google_to_inspect[['bucket_name', 'bucket_region', 'service', 'listdir_content', 'from_seo']], how='left', on='bucket_name')




    aws_urls = pd.DataFrame(
        [x for x in all_urls if 'amazonaws' in x[0]]
        , columns=['url'])
    aws_urls['bucket_name'] = aws_urls.url.apply(lambda x: get_s3_bucket_name(x))
    aws_urls.drop_duplicates('bucket_name', keep='first', inplace=True)
    aws_urls['bucket_region'] = aws_urls.url.apply(lambda x: get_s3_region(x))

    aws_se_urls = pd.DataFrame([x for x in urls_from_se if 'amazonaws' in x[0]]
        , columns=['url'])
    aws_se_urls['bucket_name'] = aws_se_urls.url.apply(lambda x: get_s3_bucket_name(x))
    aws_se_urls.drop_duplicates('bucket_name', keep='first', inplace=True)
    aws_se_urls['bucket_region'] = aws_se_urls.url.apply(lambda x: get_s3_region(x))
    aws_se_urls['from_seo'] = [True] * aws_se_urls.shape[0]

    aws_to_inspect = aws_urls\
        .merge(aws_se_urls[['bucket_name', 'bucket_region', 'from_seo', 'url']], how='outer', on=['bucket_name', 'bucket_region'], suffixes=['_inspect', '_se'], indicator='i')\
        .query('i == "left_only" or i == "right_only"')\
        .merge(df_scanned_buckets[df_scanned_buckets.service=='S3'][['bucket_name', 'bucket_region']], how='left', on=['bucket_name', 'bucket_region'], indicator='i2')\
        .query('i2 == "left_only"')\
        .copy()

    if aws_to_inspect.empty:
        __LOGGER__.debug('No new S3 bucket to scan today...')
        aws_df = pd.DataFrame()
    else:
        __LOGGER__.info(f'{aws_to_inspect[aws_to_inspect.from_seo == True].shape[0]} SEO buckets observed:\n{aws_to_inspect[aws_to_inspect.from_seo == True][["url_inspect", "bucket_name", "bucket_region", "url_se", "from_seo"]]}')

        aws_to_inspect['service'] = ['S3'] * aws_to_inspect.shape[0]
        aws_to_inspect['from_seo'] = aws_to_inspect.apply(lambda x: False if pd.isna(x['from_seo']) else True, axis=1)

        __LOGGER__.info(f'Monitoring quota: going to scan {aws_to_inspect.shape[0]} buckets!')
        aws_scan_results = scan_s3_buckets(aws_to_inspect.bucket_name.to_list(), True, workers)
        try:
            listdir_content = aws_to_inspect.apply(lambda x: try_listdir(x['url_inspect'], x['url_se']), axis=1)
            aws_to_inspect['listdir_content'] = listdir_content.apply(lambda x: Json(x))
        except ValueError as e:
            __LOGGER__.exception(e)
            aws_to_inspect['listdir_content'] = aws_to_inspect.shape[0] * [Json({})]

        aws_df = pd.DataFrame(aws_scan_results, columns=['bucket_name', 'timestamp', 'tmp_acl', 'tmp_raw_acl', 'tmp_listobject_resp', 'exists'])\
        .merge(aws_to_inspect[['bucket_name', 'bucket_region', 'service', 'listdir_content', 'from_seo']], how='left', on='bucket_name')



    do_urls = pd.DataFrame(
        [x for x in all_urls if 'digitaloceanspaces.com' in x[0]]
        , columns=['url'])
    do_urls['bucket_name'] = do_urls.url.apply(lambda x: get_do_bucket_name(x))
    do_urls.drop_duplicates('bucket_name', keep='first', inplace=True)
    do_urls['bucket_region'] = do_urls.url.apply(lambda x: get_do_region(x))

    do_se_urls = pd.DataFrame([x for x in urls_from_se if 'digitaloceanspaces.com' in x[0]]
        , columns=['url'])
    do_se_urls['bucket_name'] = do_se_urls.url.apply(lambda x: get_do_bucket_name(x))
    do_se_urls.drop_duplicates('bucket_name', keep='first', inplace=True)
    do_se_urls['bucket_region'] = do_se_urls.url.apply(lambda x: get_do_region(x))
    do_se_urls['from_seo'] = [True] * do_se_urls.shape[0]
    
    do_to_inspect = do_urls \
        .merge(do_se_urls[['bucket_name', 'bucket_region', 'from_seo', 'url']], how='outer', on=['bucket_name', 'bucket_region'], suffixes=['_inspect', '_se'], indicator='i') \
        .query('i == "left_only" or i == "right_only"')\
        .merge(df_scanned_buckets[df_scanned_buckets.service=='DigitalOcean'][['bucket_name', 'bucket_region']], how='left', on=['bucket_name', 'bucket_region'], indicator='i2')\
        .query('i2 == "left_only"')\
        .copy()

    if do_to_inspect.empty:
        __LOGGER__.debug('No new DO bucket to scan today...')
        do_df = pd.DataFrame()
    else:
        __LOGGER__.info(f'{do_to_inspect[do_to_inspect.from_seo == True].shape[0]} SEO buckets observed:\n{do_to_inspect[do_to_inspect.from_seo == True][["url_inspect", "bucket_name", "bucket_region", "url_se", "from_seo"]]}')

        do_to_inspect['service'] = ['DigitalOcean'] * do_to_inspect.shape[0]
        do_to_inspect['from_seo'] = do_to_inspect.apply(lambda x: False if pd.isna(x['from_seo']) else True, axis=1)

        __LOGGER__.info(f'Going to scan {do_to_inspect.shape[0]} DigitalOcean buckets!')
        do_scan_results = scan_digitalocean_buckets(do_to_inspect[['bucket_name', 'bucket_region']], True, workers)
        try:
            listdir_content = do_to_inspect.apply(lambda x: try_listdir(x['url_inspect'], x['url_se']), axis=1)
            do_to_inspect['listdir_content'] = listdir_content.apply(lambda x: Json(x))
        except ValueError as e:
            __LOGGER__.exception(e)
            do_to_inspect['listdir_content'] = do_to_inspect.shape[0] * [Json({})]
        do_df = pd.DataFrame(do_scan_results, columns=['bucket_name', 'timestamp', 'tmp_acl', 'tmp_raw_acl', 'tmp_listobject_resp', 'exists'])\
        .merge(do_to_inspect[['bucket_name', 'bucket_region', 'service', 'listdir_content', 'from_seo']], how='left', on='bucket_name')


    to_insert = pd.concat([aws_df, do_df, aliyun_df, google_df], ignore_index=True)
    if not to_insert.empty:
        to_insert['acl'] = to_insert.tmp_acl.apply(lambda x: Json(x))
        to_insert.drop('tmp_acl', inplace=True, axis=1)
        to_insert['raw_acl'] = to_insert.tmp_raw_acl.apply(lambda x: Json(x))
        to_insert.drop('tmp_raw_acl', inplace=True, axis=1)
        to_insert['listobject_resp'] = to_insert.tmp_listobject_resp.apply(lambda x: Json(x))
        to_insert.drop('tmp_listobject_resp', inplace=True, axis=1)

        insert(db_bindings, to_insert)

        mark_se_buckets_in_vt(db_bindings, aws_urls, do_urls)

    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run scanner inspecting S3 buckets.')
    parser.add_argument("--n_workers", dest='n_workers', help="Number of workers. Each worker must have an own DB table.", type=int,  default=5)
    parser.add_argument('--conf', dest='conf_fname', nargs='?', action='store',
                        const=sum, default="/path/to/conf/config.yaml",
                        help='Pipeline configuration file . (default: %(default)s)')

    args = parser.parse_args()
    config = load_config_yaml(args.conf_fname)

    logs_path = os.path.join(config['global']['file_storage'], 'logs', 's3scanner.log')
    global __LOGGER__; __LOGGER__ = log.getlogger("s3scanner", level=log.DEBUG, filename=logs_path)
    main(config['global']['postgres'], args.n_workers)
