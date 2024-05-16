import argparse
import re
import os
import time
import sys
import yaml
import requests
import psycopg2
from sqlalchemy import create_engine
from collections import defaultdict
import pandas as pd
from datetime import datetime, timedelta
from urllib.parse import urlparse
import tldextract

import json
from pikepdf import Object, PdfError

from func_timeout import func_timeout, FunctionTimedOut
import errno
import signal
import functools
# https://stackoverflow.com/a/2282656/7052103

from get_pdf_tree import pdf_to_graph, create_augmented_tree, identify_page_tree, find_catalog_id, attempt_find_path, \
                        find_mediabox_for_uri, flush_temporary_files, Annotation, \
                        ANNOTATION_RE, ANNOTAION_WOUT_URI, URI_ONLY_RE, \
                        PDF_TREE_PATH, GETPDFTREE_LOGS_PATH


import log

__NESTED_LEVELS__ = 4
__LOGGER__ = log.getlogger('all_urls',  level=log.INFO, filename=os.path.join(GETPDFTREE_LOGS_PATH, datetime.today().strftime("%Y-%m-%d") + ".log"))

SECONDS_TIMEOUT = 60 * 10


class DbWrapper:
    def __init__(self, db_bindings, database, user, autocommit):
        self.database = db_bindings['databases'][database]
        self.user = db_bindings['users'][user]
        self.password = db_bindings['passwords'][user]
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


class HostingServices:
    def __init__(self, db_bindings):
        self._known_services = self._fetch_known_services(db_bindings)

    def _fetch_known_services(self, db_bindings):
        with psycopg2.connect(dbname=db_bindings['databases']['pipeline'], user=db_bindings['users']['ginopino'],
                              password=db_bindings['passwords']['ginopino'], host=db_bindings['host']) as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute("""
                        select provider, hosting_type
                        from hosting_services;""")
                    services = cur.fetchall()
                except psycopg2.errors.Error as e:
                    __LOGGER__.exception(e)
                    services = []
        res = {}
        for p, ht in services:
            res[p] = ht

        return res

    @property
    def known_services(self):
        return self._known_services


class TimeoutError(Exception):
    pass

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator



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


######################################################
### START CODE FROM analysis/extract_clean_urls.py ###
######################################################

WILDCARD_TLD_RE = re.compile(r'^\*\.\S+$', re.S)
PUNYCODE_RE = re.compile(r'^// (xn--\S+) .*$', re.A)
EMPTY_URL_RE = re.compile(r'^http(s)?(://)?$', re.S)
NUMERIC_IP_RE = re.compile(r'\d+\.\d+\.\d+\.\d+')
# https://www.ietf.org/rfc/rfc3986.txt
VALID_URL_CHARS_RE = re.compile('[^(*?:*?@)+(\w\-\.\~ )(:\d)+]')
SIMPLE_EMAIL_RE = re.compile(r'[^@]+@[^@]+\.[^@]+') # source https://stackoverflow.com/a/8022584

def fetch_tlds():
    resp = requests.get('https://raw.githubusercontent.com/publicsuffix/list/master/public_suffix_list.dat')
    resp_text = resp.text

    valid_tlds = set()
    wildcard_tlds = set()
    punycode_tld = set()
    for line in resp_text.split('\n'):
        line = line.strip()
        if len(line) == 0:
            continue
        elif line.startswith('// '):
            if line[3:7] == 'xn--':
                try:
                    international_tld = PUNYCODE_RE.match(line).group(1)
                    punycode_tld.add(international_tld)
                except AttributeError as e:
                    __LOGGER__.warning(e)
                    __LOGGER__.info(line)
                    continue
        elif re.fullmatch(WILDCARD_TLD_RE, line):
            wildcard_tlds.add(line.replace('*.', ''))
        else:
            valid_tlds.add(line)
    return valid_tlds, wildcard_tlds, punycode_tld

global VALID_TLDS
global WILDCARD_TLDS
global PUNYCODE_TLDS
VALID_TLDS, WILDCARD_TLDS, PUNYCODE_TLDS = fetch_tlds()
    

def area_not_null (rect_coord_string):
    if len(rect_coord_string) == 0:
        return False
    try:
        rect_coords = [float(x) for x in rect_coord_string[1:-1].split(',') if len(x) > 0]
    except Exception as e:
        __LOGGER__.exception(e)
        return False
    if rect_coords[0] == rect_coords[2] or rect_coords[1] == rect_coords[3]:
        return False
    else:
        return True


####################################################
### END CODE FROM analysis/extract_clean_urls.py ###
####################################################

def spot_compromised_plugin(path, param, query, fragment, tld):
    to_inspect = path + param + query + fragment
    to_inspect = to_inspect.lower()

    if 'ckfinder' in to_inspect or 'ckimage' in to_inspect:
        return 'ckfinder'
    if 'kcfinder' in to_inspect:
        return 'kcfinder'
    if 'fckeditor' in to_inspect:
        return 'fckeditor'
    if 'ckeditor' in to_inspect:
        return 'ckeditor'
    if 'formcraft' in to_inspect:
        return 'formcraft'
    if 'webform' in to_inspect:
        return 'webform'
    if 'super-forms' in to_inspect:
        return 'webform'
    if 'formidable' in to_inspect:
        return 'formidable'
    # if verify_slims(tld, path):
    #     return 'slims'

    return None

def verify_slims(tld, path):

    splits = [x for x in path.split('/') if len(x.strip()) > 0]
    if tld.endswith('ac.id') or tld.endswith('sch.id'):
        if len(splits) == 2 and splits[0] == 'repository':
            return True
        if len(splits) >=1 and splits[0] == 'slims7':
            return True

    # __statics,gudangsoal,files
    if len(splits) == 4 and splits[0] == '__statics' and splits[1] == 'gudangsoal' and splits[2] == 'files':
        return True

    return False

def remap_netlocs(netloc):
    for hosting_provider in known_service_providers.keys():
        if hosting_provider in netloc:
            return hosting_provider

    return netloc



@timeout(SECONDS_TIMEOUT)
def parse_binary(pdf_filepath):
    with open(pdf_filepath, 'rb') as f_in:
        fcontent = f_in.read()

    filehash = os.path.basename(pdf_filepath)

    urls = re.findall(URI_ONLY_RE, fcontent)
    clean = []
    for _, url in urls:

        uri = url.replace(b'/URI', b'').replace(b'(', b'').replace(b')', b'').strip()
        try:
            uri_decoded = uri.decode('utf-8')
        except UnicodeDecodeError:
            uri_decoded = 'UnicodeDecodeError'

        clean.append({
            'uri_binary': uri,
            'filehash': filehash,
            'uri': uri_decoded
        })

    return clean


def parse_pdf(pdf_path, screenshot_width, screenshot_height):
    try:
        graph = pdf_to_graph(pdf_path)
    except Exception as e:
        __LOGGER__.exception(e)
        return []

    filehash = os.path.basename(pdf_path)

    augmented_tree = create_augmented_tree(graph, os.path.join(PDF_TREE_PATH, os.path.basename(pdf_path)))
    __LOGGER__.debug('Finished extracting PDF features.')
    flush_temporary_files(os.path.join(PDF_TREE_PATH, filehash))
    to_remove_json = [x for x in os.listdir(PDF_TREE_PATH) if x.endswith('.json') and filehash in x]
    for fname in to_remove_json:
        os.remove(os.path.join(PDF_TREE_PATH, fname))
    __LOGGER__.debug('Cleaned up files.')

    catalog_id = find_catalog_id(augmented_tree)
    uri_elements = [k for k, v in augmented_tree.items() if re.search(URI_ONLY_RE, v['object'])]

    try:

        uri_data = func_timeout(SECONDS_TIMEOUT, tree_properties, args=(graph, augmented_tree, catalog_id, uri_elements, screenshot_width, screenshot_height))

    except FunctionTimedOut:
        __LOGGER__.warning(f"tree_properties for filehash {filehash} could not complete within 10 minutes and was terminated.\n")
        uri_data = []

    except Exception as e:
        __LOGGER__.exception(e)
        uri_data = []


    final_res = []
    for d in uri_data:
        d['filehash'] = filehash
        final_res.append(d)
    return final_res

def tree_properties(graph, augmented_tree, catalog_id, uri_elements, screenshot_width, screenshot_height):
    uri_rects = []

    for uri_element in uri_elements:
        res = defaultdict(lambda : None)

        try:
            path_catalog_uri = attempt_find_path(graph, augmented_tree, catalog_id, uri_element)
            res['is_in_catalog_path'] = True
        except Exception:
            __LOGGER__.info("No path to Catalog for URI element " + uri_element)
            path_catalog_uri = []
            res['is_in_catalog_path'] = False

        try:
            _, mediabox_coord = find_mediabox_for_uri(path_catalog_uri, augmented_tree)
            res['has_mediabox'] = True
        except Exception:
            res['has_mediabox'] = False
            mediabox_coord = None

        uri_object = augmented_tree[uri_element]['object']
        match = re.match(ANNOTATION_RE, uri_object)
        rect_coordinates = None

        if match:
            # URI and Annotation are defined in the same PDF object
            extracted_uri = match.group(2)
            res['has_annot'] = True
            res['are_bytes_in_tree'] = True
            rect_coordinates = match.group(1)
        else:
            uri_match = re.match(URI_ONLY_RE, uri_object)
            if not uri_match:
                __LOGGER__.warning("Cannot extract URI from element.")
                if not (uri_object.startswith(b'[') and  uri_object.endswith(b']')):
                    res['are_bytes_in_tree'] = False
                    extracted_uri = None
                else: # object is an array. may include multiple URLs... not good

                    __LOGGER__.info('Processing object as an Array...')
                    try:
                        json_array = json.loads(
                            Object.parse(uri_object).to_json()
                        )
                        uri_df = pd.json_normalize(json_array)
                        attributes_urls_in_array = uri_df.apply(
                            lambda x: extract_info_df_array(x, screenshot_width, screenshot_height, mediabox_coord, res['is_in_catalog_path']),
                            axis=1).to_list()
                        uri_rects.extend(attributes_urls_in_array)
                    except (PdfError, json.JSONDecodeError):
                        __LOGGER__.warning('Processing failed. Moving on.')
                        pass
                    continue

            else:
                extracted_uri = uri_match.group(1)
                res['are_bytes_in_tree'] = True

            # find the parent object having Rect
            for parent in reversed(path_catalog_uri):
                match = re.match(ANNOTAION_WOUT_URI, augmented_tree[parent]['object'])
                if match:
                    res['has_annot'] = True
                    rect_coordinates = match.group(1)
                    break
            if not rect_coordinates:
                __LOGGER__.warning("/URI is there but we could not find any /Rect for it !?")
                res['has_annot'] = False

        try:
            res['uri'] = extracted_uri.decode('utf-8') \
                .replace("\x00", "").replace('(', '').replace(')', '').replace('/URI', '').strip()
            res['is_uri_unicode'] = True
        except UnicodeDecodeError:
            res['uri'] = 'UnicodeDecodeError'
            res['is_uri_unicode'] = False
            __LOGGER__.warning('UnicodeDecodeError when getting URI.')
        except AttributeError:
            __LOGGER__.info('My best guess: extracted_uri is None. We\'ll just ignore this case.')
            continue

        if screenshot_height and screenshot_width and mediabox_coord and rect_coordinates:
            try:
                ann = Annotation(res['uri'], rect_coordinates, screenshot_width, screenshot_height, mediabox_coord[2],
                                 mediabox_coord[3])
                clickable_area = ann.width * ann.height
                res['clickable_area_gt_zero'] = True if clickable_area > 0 else False
            except Exception:
                pass
        else:
            res['clickable_area_gt_zero'] = False

        uri_rects.append(res)

    return uri_rects


def extract_info_df_array(row, screenshot_height, screenshot_width, mediabox_coord=None, is_in_catalog_path=False):
    res = defaultdict(lambda : None)

    res['has_mediabox'] = True if mediabox_coord else False
    res['is_in_catalog_path'] = is_in_catalog_path
    res['are_bytes_in_tree'] = True

    # we need to use this ridiculous way of accessing columns because we have no clue how they are going to be
    # named for each PDF element, especially due to the flattening of nested elements.
    # but we *assume* each list (`uri_col`, `has_annot`, `has_rect`) is either empty or of length 1, bc we process one ROW
    # at a time (see `apply`)
    uri_col = [x for x in row.index if 'URI' in x]
    if not uri_col:
        return res

    res['uri'] = row.loc[uri_col[0]].replace("\x00", "").replace('(', '').replace(')', '').replace('/URI', '').strip()
    res['is_uri_unicode'] = True

    has_annot = [x for x in row.index if 'Annot' in x]
    res['has_annot'] = True if len(has_annot) > 0 else False

    has_rect = [x for x in row.index if 'Rect' in x]
    if has_rect and screenshot_height and screenshot_width and mediabox_coord:
        prepared_coords = str(row.loc[has_rect[0]]).replace('[', '').replace(']', '').replace(',', '').encode('utf-8')
        try:
            ann = Annotation(row.loc[uri_col[0]], prepared_coords, screenshot_width, screenshot_height,
                             mediabox_coord[2], mediabox_coord[3])
            clickable_area = ann.width * ann.height
        except ValueError as e:
            __LOGGER__.warning(e)
            __LOGGER__.info(row)
            clickable_area = 0
        res['clickable_area_gt_zero'] = True if clickable_area > 0 else False
    else:
        res['clickable_area_gt_zero'] = False

    return res


def url_properties(uri):
    res = defaultdict(lambda : None)
    if not pd.notnull(uri):
        return res

    # remove relative URLs
    if uri.startswith('/') or uri.startswith('#'):
        res['is_relative_url'] = True
    else:
        res['is_relative_url'] = False

    # remove empty url
    if EMPTY_URL_RE.fullmatch(uri):
        res['is_empty'] = True
    else:
        res['is_empty'] = False

    parsed_url = urlparse(uri)
    # remove email addresses
    if not parsed_url.scheme and not parsed_url.netloc and re.match(SIMPLE_EMAIL_RE, uri):
        res['is_email'] = True
    else:
        res['is_email'] = False

    # remove filepaths and other protocols
    if len(parsed_url.scheme) > 0 and parsed_url.scheme.startswith('http'):
        res['has_http_scheme'] = True
        res['netloc'] = parsed_url.scheme + '://' + parsed_url.netloc
    else:
        res['has_http_scheme'] = False
        res['netloc'] = parsed_url.netloc

    # remove localhost and 197.168.56.X
    if '127.0.0' in parsed_url.netloc or '192.168.' in parsed_url.netloc or parsed_url.netloc == 'localhost':
        res['is_local'] = True
    else:
        res['is_local'] = False

    # remove URLs with invalid chars:
    if VALID_URL_CHARS_RE.search(parsed_url.netloc):
        res['has_invalid_char'] = True
    else:
        res['has_invalid_char'] = False

    # remove URLs with invalid tlds
    tld = tldextract.extract(uri).suffix
    res['has_valid_tld'] = True
    if not tld or (tld not in VALID_TLDS and not tld in PUNYCODE_TLDS and not any([x for x in WILDCARD_TLDS if tld.endswith(x)])):
        if not NUMERIC_IP_RE.match(parsed_url.netloc):
            res['has_valid_tld'] = False

    if '.pdf' in uri[len(parsed_url.scheme) + len('://') + len(parsed_url.netloc):]:
        res['is_pdf'] = True
    else:
        res['is_pdf'] = False

    if len(tldextract.extract(uri).suffix) > 0:
        tld = tldextract.extract(uri).domain + '.' + tldextract.extract(uri).suffix
    else:
        tld = tldextract.extract(uri).domain
    res['tld'] = tld
    res['guessed_cp'] = spot_compromised_plugin(parsed_url.path, parsed_url.params, parsed_url.query, parsed_url.fragment, tld)
    res['domain_remapped'] = remap_netlocs(parsed_url.netloc)
    res['path'] = [x.strip() for x in parsed_url.path.split('/') if len(x.strip()) > 0]

    return res

    # dates = pd.date_range(start=datetime.strptime(args_start_date,'%Y-%m-%d'), end=datetime.strptime(args_end_date,'%Y-%m-%d'), freq='W').to_list()
    # for i in range(len(dates[:-1])):
    #     start_date = dates[i].strftime('%Y-%m-%d')
    #     end_date = dates[i + 1].strftime('%Y-%m-%d')
    #     __LOGGER__.info("{} {}".format(start_date, end_date))




def do(config):
    db_bindings = config['global']['postgres']
    hs = HostingServices(db_bindings)
    global known_service_providers
    known_service_providers = hs.known_services

    yesterday = (datetime.today() - timedelta(days=1)).date().strftime("%Y-%m-%d")
    with psycopg2.connect(dbname=db_bindings['databases']['pipeline'], user=db_bindings['users']['ginopino'], password=db_bindings['passwords']['ginopino'], host=db_bindings['host']) as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                SELECT DISTINCT ON (imported_samples.filehash) imported_samples.filehash, codphish.screenshot_width, codphish.screenshot_height
                FROM imported_samples
                LEFT JOIN codphish USING (filehash)
                LEFT JOIN all_urls USING (filehash)
                WHERE imported_samples.mimetype = 'application/pdf' AND imported_samples.upload_date = %s AND all_urls.filehash IS NULL  AND ( imported_samples.provider = 'cisco' OR imported_samples.provider = 'inquest' )
                ORDER BY imported_samples.filehash;
                """, (yesterday, ))
                filehashes_w_h = cur.fetchall()

                __LOGGER__.debug("SELECT completed.")
            except psycopg2.errors.Error as e:
                __LOGGER__.exception(e)
                filehashes_w_h = []


    # insert_stmt = """
    # INSERT INTO all_urls ( uri_binary, filehash, uri, is_in_catalog_path, has_annot, are_bytes_in_tree, is_uri_unicode, has_mediabox, clickable_area_gt_zero, is_pdf, occurrences_bin, occurrences_pdf, is_relative_url, is_empty, is_email, has_http_scheme, is_local, has_invalid_char, has_valid_tld )
    # VALUES %s;
    # """
    #
    # column_keys = ['uri_binary', 'filehash', 'uri', 'is_in_catalog_path', 'has_annot', 'are_bytes_in_tree', 'is_uri_unicode', 'has_mediabox',
    #                'clickable_area_gt_zero', 'is_pdf', 'occurrences_bin', 'occurrences_pdf', 'is_relative_url', 'is_empty', 'is_email', 'has_http_scheme',
    #                'is_local', 'has_invalid_char', 'has_valid_tld']

    for filehash, w, h in filehashes_w_h:
        __LOGGER__.info(filehash)

        pdf_filepath = os.path.join(config['global']['file_storage'], 'samples', rel_path(filehash), filehash)

        try:

            urls_from_bytes = parse_binary(pdf_filepath)

        except TimeoutError:
            __LOGGER__.warning(
                f"parse_binary for filehash {filehash} could not complete within 10 minutes and was terminated.\n")
            urls_from_bytes = []

        except Exception as e:
            __LOGGER__.exception(e)
            urls_from_bytes = []

        urls_from_parsed_pdf = parse_pdf(pdf_filepath, w, h)

        df_binary = pd.DataFrame(urls_from_bytes, columns=['uri', 'filehash', 'uri_binary'])

        dupl_bin = df_binary.groupby('uri').size()
        if any([x for x in dupl_bin.to_list() if x > 1]):
            sizes_series = df_binary.apply(lambda x: dupl_bin.loc[x['uri']], axis=1)
            sizes_series.name = 'occurrences_bin'
            df_binary = pd.concat([df_binary, sizes_series], axis=1).drop_duplicates('uri', keep='first')

        df_pdf_ft = pd.DataFrame(urls_from_parsed_pdf, columns=['uri', 'filehash', 'is_in_catalog_path', 'has_annot',
                                                                'are_bytes_in_tree', 'is_uri_unicode', 'has_mediabox',
                                                                'clickable_area_gt_zero'])

        if not df_pdf_ft.empty:
            url_ft = df_pdf_ft.apply(lambda x: url_properties(x['uri']), axis=1, result_type='expand')
            merged = pd.concat([df_pdf_ft, url_ft], axis=1) # join inner or outer?

            dupl_pdf_ft = merged.groupby('uri').size()
            if any([x for x in dupl_pdf_ft.to_list() if x > 1]):
                sizes_series = merged.apply(lambda x: dupl_pdf_ft.loc[x['uri']], axis=1)
                sizes_series.name = 'occurrences_pdf'
                merged = pd.concat([merged, sizes_series], axis=1).drop_duplicates('uri', keep='first')
        else:
            merged = pd.DataFrame([], columns=['uri', 'filehash', 'is_in_catalog_path', 'has_annot',
                                               'are_bytes_in_tree', 'is_uri_unicode', 'has_mediabox', 'clickable_area_gt_zero',
                                               'netloc', 'tld', 'guessed_cp', 'domain_remapped', 'path'])

        if merged.empty and df_binary.empty:
            __LOGGER__.info('Skipping insert of empty DF...')
            continue

        engine = create_engine('postgresql://{user}:{password}@{host}:{port}/{dbname}'.format(dbname=db_bindings['databases']['pipeline'], user=db_bindings['users']['ginopino'], password=db_bindings['passwords']['ginopino'], host=db_bindings['host'], port=db_bindings['port']), pool_pre_ping=True)

        final = df_binary.merge(merged, how='outer', on=['uri', 'filehash'])
        try:
            final.to_sql('all_urls', con=engine, if_exists='append', index=False)
        except Exception as e:
            __LOGGER__.exception(e)
            __LOGGER__.warning("Error inserting filehash {}.\n".format(filehash))
            continue
        else:
            __LOGGER__.info('Insert completed.')
        engine.dispose()


    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract URLs from PDFs')
    parser.add_argument('--conf', dest='conf_fname', nargs='?', action='store', default="/path/to/conf/config.yaml",
                        help='Pipeline configuration file . (default: %(default)s)')
    # parser.add_argument('--start', dest='start_date', nargs='?', action='store', help='Date when to start processing.', metavar="YYYY-MM-DD")
    # parser.add_argument('--end', dest='end_date', nargs='?', action='store', help='Date when to start processing.', metavar="YYYY-MM-DD")
    args = parser.parse_args()

    config = load_config_yaml(args.conf_fname)

    do(config)
