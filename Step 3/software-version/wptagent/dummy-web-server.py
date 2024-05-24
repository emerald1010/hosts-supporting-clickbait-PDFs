#!/usr/bin/env python
"""
Very simple HTTP server in python (Updated for Python 3.7)

Usage:

    ./dummy-web-server.py -h
    ./dummy-web-server.py -l localhost -p 8000

Send a GET request:

    curl http://localhost:8000

Send a HEAD request:

    curl -I http://localhost:8000

Send a POST request:

    curl -d "foo=bar&bin=baz" http://localhost:8000

This code is available for use under the MIT license.

----

Copyright 2021 Brad Montgomery

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and 
associated documentation files (the "Software"), to deal in the Software without restriction, 
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, 
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial 
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT 
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE 
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    

"""
from datetime import datetime
import time
import os
import sys
import yaml

import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler

# from socketserver import ThreadingMixIn
# import threading

# https://stackoverflow.com/a/13330449/7052103
from cgi import parse_header, parse_multipart
from urllib.parse import parse_qs, urlparse

from io import BytesIO
from zipfile import ZipFile, BadZipFile
import gzip
import json
import struct

import hashlib
import psycopg2
from psycopg2.extras import Json, execute_values
import pandas as pd
from numpy import nan as np_nan

import log

TARGET_FILES = {
    'job': 'job.json',
    'page_data': '1_page_data.json',
    'netlog_requests': '1_netlog_requests.json'
}
WPTAGENT_DATA_FOLDER = 'wptagent-data'
__NESTED_LEVELS__ = 4


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
        "Relative file storage path cannot be determined because len({})={} < {} chars".format(fname, len(fname),
                                                                                               2 * nested_levels))
def to_json(series_element):
    res = {}
    if not series_element or series_element is np_nan:
        return res
    for e in series_element:
        splits = [x.strip() for x in e.split(':') if len(x.strip()) > 0]
        if len(splits) == 1:
            if ':' not in e:
                res['custom_codphish'] = splits[0]
                splits_whitespace = splits[0].split(' ')
                if len(splits_whitespace) >= 2:
                    res['status'] = splits_whitespace[1]  # custom patch
            else:
                res[splits[0]] = ''
        else:
            res[splits[0]] = ' '.join(splits[1:])
    return res

def get_path(x):
    path = urlparse(x).path
    splits=[x.strip() for x in path.split('/') if len(x.strip()) > 0]
    return '/'.join(splits[:-1])


def clean_status_code(sc):
    if not sc: # avoid None cases
        return sc

    try:
        return int(sc)
    except Exception:
        sc_number = sc.strip()[:3]

        try:
            return int(sc_number)
        except Exception:
            return None



def parse_folder_content(folder_contents, path):
    job_data = folder_contents.get(TARGET_FILES['job'], {})
    __LOGGER__.debug(job_data)
    job_url = job_data.get('url')
    m = hashlib.sha256()
    m.update(job_url.encode('utf-8'))
    url_hash = m.hexdigest()

    error_msg = None
    start_query_index = path.index('?')
    query = path[start_query_index + 1:]
    if query.find('&error=') > -1:
        error_msg = query[query.find('&error=') + 7:]

    to_insert_job = {
        'url': job_url,
        'domain_path': urlparse(job_url).scheme + '://' + urlparse(job_url).netloc + '/' + get_path(job_url),
        'url_hash': url_hash,
        'timestamp': datetime.now(),
        'cmdline': job_data.get('addCmdLine'),
        'error_msg': error_msg
    }
    __LOGGER__.debug(to_insert_job)

    pagedata_data = folder_contents.get(TARGET_FILES['page_data'], {})
    #pagedata_url = pagedata_data.get('URL'), might be NULL if the request failed, to avoid NotNullViolation we use job_url
    to_insert_pagedata = {
        'url': job_url,
        'url_hash': url_hash,
        'result': pagedata_data.get('result'),
        'detected': Json(pagedata_data.get('detected')),
        'detected_apps': Json(pagedata_data.get('detected_apps'))
    }
    __LOGGER__.debug(to_insert_pagedata)

    netlog_data = folder_contents.get(TARGET_FILES['netlog_requests'], {})
    if not netlog_data:
        df_netlog_json = pd.DataFrame()
    else:
        df_netlog_json = pd.json_normalize(netlog_data)
    if not df_netlog_json.empty:
        if "tls_version" not in df_netlog_json:
            df_netlog_json['tls_version'] = pd.Series()
        if "server_address" not in df_netlog_json:
            df_netlog_json['server_address'] = pd.Series()
        if 'response_headers' not in df_netlog_json:
            df_netlog_json['response_headers'] = pd.Series([None] * df_netlog_json.shape[0])

        df_netlog = df_netlog_json[
            ['url', 'request_headers', 'response_headers', 'method', 'server_address', 'tls_version']].copy()

        df_netlog['url_hash'] = pd.Series([url_hash] * df_netlog.shape[0])
        new_req_head_series = df_netlog['request_headers'].apply(lambda x: to_json(x))
        new_resp_head_series = df_netlog['response_headers'].apply(lambda x: to_json(x))
        df_netlog['status_code'] = new_resp_head_series.apply(lambda x: clean_status_code(x.get('status', None)))
        df_netlog.drop(columns=['response_headers', 'request_headers'])
        df_netlog.rename(columns={
            'new_req_head': 'request_headers',
            'new_resp_head': 'response_headers'
        }, inplace=True)
        df_netlog['request_headers'] = new_req_head_series.apply(lambda x: Json(x))
        df_netlog['response_headers'] = new_resp_head_series.apply(lambda x: Json(x))
        __LOGGER__.debug(df_netlog)
    else:
        df_netlog = df_netlog_json #return empty DF

    return to_insert_job, to_insert_pagedata, df_netlog


def cdn_db_persist(to_insert_job, to_insert_pagedata, df_netlog):

    ######
    ## URL / DATA COULD HAVE BACKSLASHES !!!       CLEAN UP!
    ######

    __LOGGER__.debug(f"Persisting content for request {to_insert_job['url_hash']}")

    insert_stmt_job = """
        INSERT INTO job ( url, url_hash, timestamp, cmdline, domain_path )
        VALUES ( %(url)s, %(url_hash)s, %(timestamp)s, %(cmdline)s, %(domain_path)s );
    """
    db_wrapper = DbWrapper(db_bindings, 'wptagent', 'user', True)
    cur = db_wrapper.get_cursor()
    try:
        cur.execute(insert_stmt_job, to_insert_job)
    except psycopg2.errors.UniqueViolation as e2:
        __LOGGER__.exception(e2)
        pass
    except psycopg2.errors.OperationalError as e:
        __LOGGER__.exception(e)
        __LOGGER__.info('Retrying insert...')
        db_wrapper.release_cursor()
        cur = db_wrapper.get_cursor()

        try:
            cur.execute(insert_stmt_job, to_insert_job)
        except psycopg2.errors.Error as e:
            __LOGGER__.exception(e)

    except psycopg2.errors.Error as e:
        __LOGGER__.exception(e)
    else:
        __LOGGER__.debug("Insert into `job` completed successfully.")



    insert_stmt_pagedata = """
        INSERT INTO page_data ( url, url_hash, result, detected, detected_apps )
        VALUES ( %(url)s, %(url_hash)s, %(result)s, %(detected)s, %(detected_apps)s );
    """
    cur = db_wrapper.get_cursor()
    try:
        cur.execute(insert_stmt_pagedata, to_insert_pagedata)
    except psycopg2.errors.OperationalError as e:
        __LOGGER__.exception(e)
        __LOGGER__.info('Retrying insert...')
        db_wrapper.release_cursor()
        cur = db_wrapper.get_cursor()

        try:
            cur.execute(insert_stmt_pagedata, to_insert_pagedata)
        except psycopg2.errors.Error as e:
            __LOGGER__.exception(e)

    except psycopg2.errors.Error as e:
        __LOGGER__.exception(e)
    else:
        __LOGGER__.debug("Insert into `page_data` completed successfully.")


    insert_stmt_netlog = """
        INSERT INTO netlog_requests ( url, url_hash, request_headers, response_headers, method, server_address, tls_version, status_code )
        VALUES %s;
    """
    cur = db_wrapper.get_cursor()
    #df_netlog.where(pd.notnull(df_netlog), None) -> status_code is None when response is null
    # none itself is not a problem and can be inserted in the DB normally, but if mixed with int types becomes a float!
    # pandas converts it to a proper number r.t. to NULL and returns psycopg2.errors.NumericValueOutOfRange: integer out of range
    # we need to force pandas to cast everything to `object` and then psycopg2 will handle the rest
    try:
        execute_values(cur, insert_stmt_netlog, df_netlog.where(pd.notnull(df_netlog), None).to_dict(orient='records'),
                       template="( %(url)s, %(url_hash)s, %(request_headers)s, %(response_headers)s, %(method)s, %(server_address)s, %(tls_version)s, %(status_code)s )",
                       page_size=500)
    except psycopg2.errors.OperationalError as e:
        __LOGGER__.exception(e)
        __LOGGER__.info('Retrying insert...')
        db_wrapper.release_cursor()
        cur = db_wrapper.get_cursor()

        try:
            execute_values(cur, insert_stmt_netlog, df_netlog.where(pd.notnull(df_netlog), None).to_dict(orient='records'),
                           template="( %(url)s, %(url_hash)s, %(request_headers)s, %(response_headers)s, %(method)s, %(server_address)s, %(tls_version)s, %(status_code)s )",
                           page_size=500)
        except psycopg2.errors.Error as e:
            __LOGGER__.exception(e)

    except psycopg2.errors.Error as e:
        __LOGGER__.exception(e)
    else:
        __LOGGER__.debug("Insert into `netlog_requests` completed successfully.")

    db_wrapper.release_cursor()
    return

def cf_db_persist(query, data):
    __LOGGER__.debug(f"Persisting content for request {data[2]}")

    db_wrapper = DbWrapper(db_bindings, 'wptagent', 'user', True)
    cur = db_wrapper.get_cursor()
    try:
        cur.execute(query, data)
    except psycopg2.errors.CheckViolation as e:
        __LOGGER__.info(e)
        pass
    except psycopg2.errors.OperationalError as e2:
        __LOGGER__.exception(e2)
        __LOGGER__.info('Retrying insert...')
        db_wrapper.release_cursor()
        cur = db_wrapper.get_cursor()

        try:
            cur.execute(query, data)
        except psycopg2.errors.Error as e:
            __LOGGER__.exception(e)

    except psycopg2.errors.Error as e:
        __LOGGER__.exception(e)
    else:
        __LOGGER__.debug("Insert into `cf_analyses` completed successfully.")

    db_wrapper.release_cursor()

def rc_db_persist(query, data):
    __LOGGER__.debug(f"Persisting content for request {data[2]}")

    db_wrapper = DbWrapper(db_bindings, 'wptagent', 'user', True)
    cur = db_wrapper.get_cursor()
    try:
        cur.execute(query, data)
    except psycopg2.errors.OperationalError as e2:
        __LOGGER__.exception(e2)
        __LOGGER__.info('Retrying insert...')
        db_wrapper.release_cursor()
        cur = db_wrapper.get_cursor()

        try:
            cur.execute(query, data)
        except psycopg2.errors.Error as e:
            __LOGGER__.exception(e)

    except psycopg2.errors.Error as e:
        __LOGGER__.exception(e)
    else:
        __LOGGER__.debug("Insert into `random_crawling` completed successfully.")

    db_wrapper.release_cursor()



def filesystem_persist(folder_contents, url_hash, rescan=False, persist_html=False):
    dest_folder = os.path.join(filestorage_path, WPTAGENT_DATA_FOLDER, rel_path(url_hash))
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    if rescan:
        path = os.path.join(dest_folder, url_hash + '_rescan.zip')
    else:
        path = os.path.join(dest_folder, url_hash + '.zip')

    with ZipFile(path, 'w') as zfolder:
        for filename, filecontent in folder_contents.items():
            try:
                zfolder.writestr(filename, data=json.dumps(filecontent))
            except TypeError as e:
                if filename == 'body.txt' and persist_html: # dirty workaround
                    zfolder.writestr(filename, data=filecontent)
                else:
                    __LOGGER__.exception(e)
    return


# https://docs.python.org/3.5/library/socketserver.html#asynchronous-mixins
# https://stackoverflow.com/a/14089457/7052103
# class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
#     """Handle requests in a separate thread."""


class Handler(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

    def _html(self, message):
        """This just generates an HTML document that includes `message`
        in the body. Override, or re-write this do do more interesting stuff.

        """
        content = f"<html><body><h1>{message}</h1></body></html>"
        return content.encode("utf8")  # NOTE: must return a bytes object!

    def do_GET(self):
        self._set_headers()
        self.wfile.write(self._html("hi!"))

    def do_HEAD(self):
        self._set_headers()

    def do_POST(self):
        __LOGGER__.info(f"{self.headers['Host']}/{self.path}")
        
        folder_content = self.parse_POST()
        job, pagedata, netlogdata = parse_folder_content(folder_content, self.path)
        
        if 'cdn_analyses' in self.headers['Host']:
            self.persist_cdn(folder_content, job, pagedata, netlogdata)
        elif 'compromised_infrastructure' in self.headers['Host']:
            self.persist_compr_infr(folder_content, job, pagedata, netlogdata)
        elif 'random_crawling' in self.headers['Host']:
            self.persist_rand_crawl(folder_content, job, pagedata, netlogdata)

        self._set_headers()
        self.send_response(200)
        self.end_headers()

    def persist_cdn(self, folder_content, job, pagedata, netlogdata):

        if (job['error_msg'] and not 'ERR_CONNECTION_TIMED_OUT' in job['error_msg'] and not 'ERR_NAME_NOT_RESOLVED' in job['error_msg'])\
                or not job['error_msg']:
           cdn_db_persist(job, pagedata, netlogdata)
        filesystem_persist(folder_content, job['url_hash'])

        return

    def persist_compr_infr(self, folder_content, job, pagedata, netlogdata):
        if netlogdata.empty:
            response_headers = None
        else:
            netlogdata['url'] = netlogdata.url.apply(lambda x: x[:-1] if x.endswith('/') else x)
            response_headers = netlogdata[netlogdata.url==job['url']]['response_headers'].to_list()

        if 'rescan' not in self.headers['Host']:

            filesystem_persist(folder_content, job['url_hash'], rescan=False, persist_html=True)

            query = """
            INSERT INTO cf_analyses ( domain, domain_hash, page_data_onl, timestamp_onl, error_msg, headers_onl )
            VALUES ( %s, %s, %s, %s, %s, %s::json[] )
            ON CONFLICT (domain) DO UPDATE
            SET page_data_onl = %s, timestamp_onl = %s, error_msg = %s, headers_onl = %s::json[],  times_scanned_onl = cf_analyses.times_scanned_onl + 1
            WHERE cf_analyses.domain = %s AND cf_analyses.domain_hash = %s;
            """

            to_insert = (
                job['url'], job['url_hash'], pagedata['detected'], job['timestamp'], job['error_msg'], response_headers,
                pagedata['detected'], job['timestamp'], job['error_msg'], response_headers,
                job['url'], job['url_hash']
            )

        else:

            filesystem_persist(folder_content, job['url_hash'], rescan=True, persist_html=True)

                        # query = """
            # UPDATE cf_analyses
            # SET page_data_off = %s, timestamp_off = %s, headers_off = %s::json[], error_msg = %s, times_scanned_off = times_scanned_off + 1
            # WHERE domain = %s AND domain_hash = %s;
            # """
            # to_insert = (
            #     pagedata['detected'], job['timestamp'], netlogdata[netlogdata.url==job['url']]['response_headers'].to_list(), job['error_msg'],
            #     job['url'], job['url_hash']
            # )
            query = """
            INSERT INTO cf_analyses ( domain, domain_hash, page_data_off, timestamp_off, error_msg, headers_off )
            VALUES ( %s, %s, %s, %s, %s, %s::json[] )
            ON CONFLICT (domain) DO UPDATE
            SET page_data_off = %s, timestamp_off = %s, error_msg = %s, headers_off = %s::json[],  times_scanned_off = cf_analyses.times_scanned_off + 1
            WHERE cf_analyses.domain = %s AND cf_analyses.domain_hash = %s;
            """
            to_insert = (
                job['url'], job['url_hash'], pagedata['detected'], job['timestamp'], job['error_msg'], response_headers,
                pagedata['detected'], job['timestamp'], job['error_msg'], response_headers,
                job['url'], job['url_hash']
            )

        cf_db_persist(query, to_insert)

        return


    def persist_rand_crawl(self, folder_content, job, pagedata, netlogdata):
        if netlogdata.empty:
            response_headers = None
        else:
            netlogdata['url'] = netlogdata.url.apply(lambda x: x[:-1] if x.endswith('/') else x)
            response_headers = netlogdata[netlogdata.url==job['url']]['response_headers'].to_list()

        if 'rescan' not in self.headers['Host']:

            filesystem_persist(folder_content, job['url_hash'])

            query = """
            INSERT INTO random_crawling ( uri, domain_hash, page_data, timestamp, error_msg, headers )
            VALUES ( %s, %s, %s, %s, %s, %s::json[] )
            ON CONFLICT (uri) DO NOTHING;
            """

            to_insert = (
                job['url'], job['url_hash'], pagedata['detected'], job['timestamp'], job['error_msg'], response_headers
            )

            rc_db_persist(query, to_insert)

        return


    def parse_POST(self):
        ctype, pdict = parse_header(self.headers.get('Content-Type'))
        pdict['boundary'] = bytes(pdict['boundary'], "utf-8")
        pdict['CONTENT-LENGTH'] = int(self.headers.get('Content-length'))
        if ctype == 'multipart/form-data':
            postvars = parse_multipart(self.rfile, pdict)
            f = postvars.get('file')

            dir_content = {}

            # https://techoverflow.net/2018/01/16/downloading-reading-a-zip-file-in-memory-using-python/
            with ZipFile(BytesIO(f[0])) as thezip:
                for zipinfo in thezip.infolist():
                    with thezip.open(zipinfo) as thefile:
                        __LOGGER__.debug(f'Processing {thefile}')
                        if zipinfo.filename.endswith('.json.gz'):
                            filecontent = gzip.decompress(thefile.read()).decode('utf-8')
                            try:
                                dir_content[zipinfo.filename.replace('.gz', '')] = json.loads(filecontent)
                            except json.decoder.JSONDecodeError as e:
                                __LOGGER__.exception(e)
                                __LOGGER__.info(thefile)

                        elif 'bodies.zip' in zipinfo.filename:
                            body_filedata = BytesIO(thefile.read())
                            try:
                                with ZipFile(body_filedata) as nested_zip:
                                    body_file_name = [x for x in nested_zip.namelist() if x.endswith('body.txt')][0]
                                    body_text = nested_zip.open(body_file_name).read()
                                    dir_content['body.txt'] = body_text
                            except IndexError:
                                continue
                            except BadZipFile as e:
                                __LOGGER__.info(e)
                                continue

                        elif zipinfo.filename.endswith('json') or zipinfo.filename.endswith('txt'):
                            filecontent = thefile.read()
                            try:
                                dir_content[zipinfo.filename] = json.loads(filecontent)
                            except json.decoder.JSONDecodeError as e:
                                __LOGGER__.exception(e)
                                __LOGGER__.info(thefile)
                        else:
                            __LOGGER__.debug(f'Ignored {thefile}')
                            continue
                        # with open(os.path.join(dest_dir, zipinfo.filename.replace('.gz', '')), 'w+') as f_out:
                        #     f_out.write(filecontent.decode('utf-8'))
            return dir_content
        # elif ctype == 'application/x-www-form-urlencoded':
        #     length = int(self.headers['content-length'])
        #     postvars = parse_qs(
        #             self.rfile.read(length),
        #             keep_blank_values=1)
        # else:
        #     postvars = {}
        # return postvars
        else:
            return {}


def run(server_class=HTTPServer, handler_class=Handler, addr="localhost", port=8000):
    server_address = (addr, port)
    httpd = server_class(server_address, handler_class)

    print(f"Starting httpd server on {addr}:{port}")
    httpd.serve_forever()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run a simple HTTP server")
    parser.add_argument(
        "-l",
        "--listen",
        default="localhost",
        help="Specify the IP address on which the server listens",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8089,
        help="Specify the port on which the server listens",
    )
    parser.add_argument('--conf', dest='conf_fname', nargs='?', action='store',
                        const=sum, default="/path/to/conf/config.yaml",
                        help='Pipeline configuration file . (default: %(default)s)')

    args = parser.parse_args()

    config = load_config_yaml(args.conf_fname)
    global db_bindings
    db_bindings = config['global']['postgres']
    global filestorage_path
    filestorage_path = config['global']['file_storage']
    global __LOGGER__
    __LOGGER__ = log.getlogger("dummy-webserver", level=log.INFO, filename=os.path.join(filestorage_path, 'logs', 'wptagents', f'{os.uname().nodename}.log'))

    run(addr=args.listen, port=args.port)
