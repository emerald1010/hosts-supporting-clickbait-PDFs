import subprocess
import time
import yaml
import psycopg2
import numpy as np
from psycopg2.extensions import register_adapter, AsIs
from psycopg2.extras import execute_values
from collections import deque, defaultdict
import os, sys
import resource
import json
import re
import networkx as nx
import pydot
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
from networkx.drawing.nx_pydot import write_dot
import pandas as pd
from datetime import datetime, timedelta
from time import perf_counter, sleep
from statistics import mean, stdev
from progress.bar import Bar
import shutil
import fitz # pymupdf
from pikepdf import Object

# from test_draw_rect import copy_and_draw
import log

__NESTED_LEVELS__ = 4


PDF_TREE_PATH = "/path/to/pdf_trees"
PEEPDF_PATH = '/path/to/PDF_graph/peepdf_interface.py'

LOGS_PATHS = [
    '/path/to/logs/'
]
GETPDFTREE_LOGS_PATH = '/path/to/logs/getpdftree_logs'

if not os.path.exists(PDF_TREE_PATH):
    os.makedirs(PDF_TREE_PATH)

__LOGGER__ = log.getlogger('building-tree', level=log.WARNING, filename=os.path.join(GETPDFTREE_LOGS_PATH, datetime.today().strftime("%Y-%m-%d") + ".log"))

DOT_LAYOUT = 'dot'
KW_CATALOG = '/Catalog'

ANNOTATION_RE = re.compile(b'<<(?=.*?/Subtype\s*?/Link)(?=.*?/Rect\s*?\[(.*?)\])(?=.*?/Type\s*?/Annot)(?=.*?/URI\s*?\((.*?)\)).*?>>$', re.S)
ANNOTAION_WOUT_URI = re.compile(b'<<(?=.*?/Subtype\s*?/Link)(?=.*?/Rect\s*?\[(.*?)\])(?=.*?[/Type\s*?/Annot|/A]).*?>>$', re.S)
URI_ONLY_RE = re.compile(b'<<.*(/URI\s*\((.*?)\)).*?>>', re.S)
MEDIABOX_RE = re.compile(b'^(?=.*?/Page)(?=.*?/MediaBox\s*?\[(.*?)\]).*?$', re.S)
KIDS_RE = re.compile(b'<<(?=.*?/Page)(?=.*?Kids\s*?\[(.*?)\])(?=.*?Count\s*?(\d+)).*?>>$', re.S)
PAGE_TREE_ROOT_RE = re.compile(b'<<(?=.*?/Page)(?=.*?Kids\s*?\[(.*?)\])(?=.*?Count\s*?(\d+))(?!.*?/Parent.*?).*?>>$', re.S)
OBJ_ATTR_RE = re.compile(r'(?P<attr>/[a-zA-Z]+) ([^\n].*?)[\n|\r|\f]', re.S)
OBJ_REFERENCES_RE = re.compile(r'([0-9]* [0-9]* R)*', re.S)


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
                self.conn = psycopg2.connect(dbname=self.database, user=self.user, password=self.password, host=self.host, port=self.port,
                                             keepalives=1, keepalives_idle=30, keepalives_interval=10, keepalives_count=5)
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


class Annotation(object):
    # https://www.adobe.com/content/dam/acom/en/devnet/pdf/pdfs/pdf_reference_archives/PDFReference.pdf section 3.8.3
    def __init__(self, extracted_url, raw_rect_coordinates, image_width, image_height, mediabox_width, mediabox_height):
        unicode_coords = raw_rect_coordinates.decode('utf-8').strip()
        # possible decode error not handled on purpose
        splits = [float(x.strip()) for x in unicode_coords.split() if len(x.strip()) > 0]
        if any([x for x in splits if x < 0]):
            __LOGGER__.warning("Negative vertex detected....")
        low_left_x, low_left_y, up_right_x, up_right_y = [round(vertex) for vertex in splits]
        tmp_low_left_x, tmp_low_left_y, tmp_up_right_x, tmp_up_right_y = [round(vertex) for vertex in splits]
        # these coordinates only make sense w r t the mediabox coordinates. to draw them on the png image, we need to rescale them.
        # https://stackoverflow.com/a/45725015
        self.scaled_llx = round(low_left_x * image_width / mediabox_width)
        self.scaled_urx = round(up_right_x * image_width / mediabox_width)
        scaled_lly = round(low_left_y * image_height / mediabox_height)
        scaled_ury = round(up_right_y * image_height / mediabox_height)
        # pdf axis origin is on the bottom left, and it is thus incompatible with cv2's one
        self.translated_lly = image_height - scaled_lly
        self.translated_ury = image_height - scaled_ury

        self.width = abs(self.scaled_urx - self.scaled_llx)
        # self.height = self.translated_ury - self.translated_lly
        # facepalm: with the new coordinates, the LEFT one is greater
        self.height = abs(self.translated_lly - self.translated_ury)

        self.url = extracted_url

    def __str__(self):
        return """
        scaled llx {}
        scaled urx {}
        translated lly {}
        translated ury {}
        width {}
        height {}
        """.format(self.scaled_llx, self.scaled_urx, self.translated_lly, self.translated_ury, self.width, self.height)

def list_to_coord(str_list):
    if len(str_list) == 0:
        return []

    rectangles = []
    for string in str_list:
        splits = [float(x.strip()) for x in string.split() if len(x.strip()) > 0]
        rectangles.append(tuple([round(vertex) for vertex in splits]))

    return rectangles


def normalize_id(id):
    return str(id)

def pdf_to_graph(pdf_path):
    basename = os.path.basename(pdf_path)
    output_path = os.path.join(PDF_TREE_PATH, basename.replace('.pdf', '') + '.json')
    cmd = ['/usr/bin/python2', PEEPDF_PATH, '--pdf', pdf_path, '--output', output_path]
    subprocess.run(cmd, timeout=120)

    tree_candidates = [os.path.join(PDF_TREE_PATH, x) for x in  os.listdir(os.path.dirname(output_path)) if basename in x and not os.path.isdir(os.path.join(PDF_TREE_PATH, x))]
    if len(tree_candidates) < 1:
        __LOGGER__.error(output_path + " does not seem to exist. Was there any error calling PEEPDF?")
        __LOGGER__.warning("If AttributeError: 'NoneType' object has no attribute 'getRawValue' in e.message:\tXref table is probably corrupted. Falling back to muPDF.")
        raise Exception("Error in reading PDF.")

    trees = {}
    for tree_path in tree_candidates:
        version = os.path.basename(tree_path).split("_v")[1].replace('.json', '')
        with open(tree_path, 'r') as f:
            pdf_tree = json.load(f)
        trees[version] = pdf_tree
    if len(trees) < 1:
        __LOGGER__.error("PDF trees were not loaded correctly.")
        raise Exception("Error in reading PDF.")

    versions_sorted = sorted(list(trees.keys()), reverse=True)
    graph = unfold_tree(trees, versions_sorted, pdf_path) # is there a better way to pass the path to this function? >.<

    return graph

# TODO WIP !!!
def parse_mupdf(pdf_path):
    basename = os.path.basename(pdf_path)
    output_path = os.path.join(PDF_TREE_PATH, basename.replace('.pdf', ''))
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    version = 'muPDF'

    doc = fitz.open(pdf_path)
    xreflen = doc.xref_length()
    for obj_indir_ref in range(1, xreflen):
        with open(os.path.join(output_path, '{}_v{}.txt'.format(obj_indir_ref, version)), 'w+') as f_out:
            f_out.write(doc.xref_object(obj_indir_ref, compressed=False))


def unfold_tree(trees, versions_sorted, pdf_path):
    graph = nx.DiGraph()
    graph.add_node('root')

    visited_nodes = set() # older nodes defined in a newer version are not explored and not added to the graph

    for version in versions_sorted:
        for root in trees[version]:
            assert (len(root) == 2)
            subtree_pdf_id = normalize_id(root[0])
            subtree = root[1]
            assert (isinstance(subtree, dict))
            graph.add_edge('root', subtree_pdf_id)

            for leaf in subtree.keys():
                graph.add_node(normalize_id(leaf), tag=subtree[leaf][0], version=version)

            queue = deque([subtree_pdf_id])
            while len(queue) > 0:
                # __LOGGER__.info("Still " + str(len(queue)) + " elements in the queue.")
                current_node = queue.popleft()
                if current_node in visited_nodes:
                    continue
                # __LOGGER__.debug("Current node is {}.\n".format(current_node))
                try:
                    neighbours = subtree[current_node][1]
                    visited_nodes.add(current_node) # i.e. this node has an entry in the tree
                except KeyError:
                    __LOGGER__.debug("Node {} has no entry in the tree. This is version {}.".format(current_node, version))
                    catalog_obj = find_catalog_obj(current_node, pdf_path)
                    if len(catalog_obj) > 0: # works only if this node is the catalog
                        visited_nodes.add(current_node)

                    neighbours = []
                    for attribute, value in catalog_obj:
                        neighbours.extend([x.strip().split(' ')[0]
                                        for x in OBJ_REFERENCES_RE.findall(value) if len(x.strip()) > 0])
                    # for x in subtree.keys():
                    #     normalized_node = normalize_id(x)
                    #     if normalized_node in visited_nodes:
                    #         continue
                    #     queue.append(normalized_node)
                    # continue
                for neighbour in neighbours:
                    norm_neighbour = normalize_id(neighbour)
                    graph.add_edge(current_node, norm_neighbour)
                    # __LOGGER__.debug("Node " + current_node + " connected to " + norm_neighbour)
                    if norm_neighbour not in visited_nodes:
                        queue.append(norm_neighbour)

            remaining_nodes = set(subtree.keys()) - visited_nodes
            if len(remaining_nodes) > 0:
                # assert (set(nx.isolates(graph)) == remaining_nodes)
                __LOGGER__.debug("Adding {} yet not visited nodes.".format(len(remaining_nodes)))

                for isolated_node in remaining_nodes:
                    subtree = trees[version][0][1]
                    assert ( isolated_node in subtree.keys() )
                    visited_nodes.add(isolated_node)
                    neighbours = subtree[isolated_node][1]
                    for neighbour in neighbours:
                        norm_neighbour = normalize_id(neighbour)
                        graph.add_edge(isolated_node, norm_neighbour)
        __LOGGER__.debug("Unfolded tree version " + str(version))

    return graph


def draw_graph(graph, layout, save=None):
    if layout != 'dot' and layout != 'twopi':
        __LOGGER__.error('Please provide either "dot" or "twopi" as layout parameter.')
        return
    if layout == 'twopi':
        pos = graphviz_layout(graph, prog="twopi")
        nx.draw(graph, pos, with_labels=True)
        plt.show()
        plt.close('all')
    if layout == 'dot' and not save:
        __LOGGER__.error('Please provide a directory path where to save the printed graph.')
        return
    else:
        if not os.path.exists(save):
            os.makedirs(save)
        dot_filepath = os.path.join(save, "graph.dot")
        write_dot(graph, dot_filepath)
        pdf_graph_filepath = os.path.join(save, "printed_graph.pdf")
        cmd = ['dot', '-o', pdf_graph_filepath, '-Tpdf', dot_filepath]
        p = subprocess.Popen(cmd)
        p.communicate()


def find_catalog_id(augmented_tree):
    for key, val in augmented_tree.items():
        if '/Catalog' in val['attribute']:
            return key
    __LOGGER__.warning('Found no /Catalog in this tree!')

def find_mediaboxes_ids(augmeented_tree):
    mediaboxes_ids = []
    for key, val in augmeented_tree.items():
        if '/MediaBox' in val['attribute']:
            mediaboxes_ids.append(key)
    return mediaboxes_ids

def find_catalog_obj(subtree_id, pdf_path):
    if subtree_id == 'None':
        __LOGGER__.error("Cannot look for None object key.")
        return []

    pdf = fitz.open(pdf_path)
    mupdf_catalog = pdf.pdf_catalog()
    if subtree_id != str(mupdf_catalog):
        __LOGGER__.info('Object {} (peepdf) does not seem to be the Catalog (mupdf: {}).'.format(subtree_id, mupdf_catalog))
        if subtree_id == '0':
            __LOGGER__.error('Trying to reference object 0 (Bad xref)!')
        return []
    else:
        catalog_obj = pdf.xref_object(mupdf_catalog)
        attributes = OBJ_ATTR_RE.findall(catalog_obj)

        for match in attributes:
            if '/Type' in match[0] and '/Catalog' in match[1]:
                __LOGGER__.debug('/Catalog found!')
                return attributes
        else:
            __LOGGER__.warning('No /Catalog found with mupdf.')
            return []


def identify_page_tree(augmented_tree):
    pages = {k: v for k, v in augmented_tree.items() if "Page" in v['attribute']}
    kids_ids = []
    page_tree_root = set()
    for k, v in pages.items():
        page_tree_root_match = re.match(PAGE_TREE_ROOT_RE, v['object'])
        if page_tree_root_match:
            reference = page_tree_root_match.group(1).decode('utf-8').strip().split('R')
            # some page roots have kids which are not Pages...
            root_kids = [x.strip().split(' ')[0] for x in reference if len(x) > 0 ]
            kids_ids.extend([x for x in root_kids if x in pages.keys()])
            page_tree_root.add(k)
            page_count_value = int(page_tree_root_match.group(2))
    if len(page_tree_root) != 1:
        __LOGGER__.warning('Cannot determine hierarchy of Page elements.')
        raise Exception()

    kids_queue = deque(kids_ids)
    while len(kids_queue) > 0:
        first_kid = kids_queue.popleft()
        match = re.match(KIDS_RE, pages[first_kid]['object'])
        if not match:
            continue
        reference = match.group(1).decode('utf-8').strip().split('R')
        for split in reference:
            if len(split) > 0:
                id = split.strip().split(' ')[0]
                kids_ids.append(id)
                kids_queue.append(id)

    # check if the number of kids found is equal to the number of declared ones
    page_id = page_tree_root.pop()
    if len(kids_ids) != int(page_count_value):
        __LOGGER__.warning("PDF Page Tree Root node declared {} (page) leaves, but {} were observed!".format(page_count_value, len(kids_ids)))

    return page_id, kids_ids

def create_augmented_tree(graph, path):
    res = {}
    id_to_tag = nx.get_node_attributes(graph, 'tag')
    id_to_version = nx.get_node_attributes(graph, 'version')

    for element_id, version in id_to_version.items():
        obj_path = os.path.join(path, "{}_v{}.txt".format(element_id, version))
        if not os.path.exists(obj_path):
            __LOGGER__.error('Missing file {} in {}!'.format("{}_v{}.txt".format(element_id, version), path))
            sys.exit(-1)
        with open(obj_path, 'rb') as obj_file:
            res[element_id] = {
                'attribute': id_to_tag[element_id],
                'object': obj_file.read(),
                'version': version
            }
    return res

def find_mediabox_for_uri(path_catalog_uri, augmented_tree):
    for element_id in reversed(path_catalog_uri):
        coordinates = re.findall(MEDIABOX_RE, augmented_tree[element_id]['object'])
        if coordinates:
            x, y, w, h = list_to_coord(coordinates)[0]
            if w <= 0 or h <= 0:
                continue
            else:
                return (element_id, (x, y, w, h))
    else:
        __LOGGER__.error("Unable to find MediaBox with area > 0 for this URI node.")
        raise Exception("Element not found.")

def attempt_find_path(graph, augmented_tree, catalog_id, uri_element):
    for path in nx.all_shortest_paths(graph, catalog_id, uri_element):
        path_wout_uri = path
        annot_found = False
        mbox_found = False
        for element_id in reversed(path_wout_uri):
            if b'Annot' in augmented_tree[element_id]['object']:
                annot_found = True
            if b'MediaBox' in augmented_tree[element_id]['object']:
                mbox_found = True
            if annot_found and mbox_found:
                return path
    else:
        for path in nx.all_simple_paths(graph, catalog_id, uri_element, cutoff=7):
            path_wout_uri = path
            annot_found = False
            mbox_found = False
            for element_id in reversed(path_wout_uri):
                if b'Annot' in augmented_tree[element_id]['object']:
                    annot_found = True
                if b'MediaBox' in augmented_tree[element_id]['object']:
                    mbox_found = True
                if annot_found and mbox_found:
                    return path
        __LOGGER__.warning("Unable to find Annot and MediaBox elements in shortest&simple paths ({} -> {}, cutoff=7)."
                           .format(uri_element, catalog_id))
        raise Exception("Element not found.")

def find_uri_clickable_area(graph, augmented_tree):
    uri_rects = []

    try:
        _, page_tree_leaves = identify_page_tree(augmented_tree)
    except Exception:
        page_tree_leaves = None

    catalog_id = find_catalog_id(augmented_tree)

    uri_elements = [k for k, v in augmented_tree.items() if re.search(URI_ONLY_RE, v['object'])]
    for uri_element in uri_elements:
        tmp = defaultdict(lambda : None)
        # path_catalog_uri = nx.shortest_path(graph, catalog_id, uri_element)[:-1]  # can already remove the last element (URI)
        try:
            path_catalog_uri = attempt_find_path(graph, augmented_tree, catalog_id, uri_element)
        except Exception:
            __LOGGER__.info("No path to Catalog for URI element " + uri_element)
            continue


        try:
            mediabox_id, mediabox_coord = find_mediabox_for_uri(path_catalog_uri, augmented_tree)
        except Exception:
            mediabox_id, mediabox_coord = None, None
        tmp['mediabox_coord'] = mediabox_coord
        tmp['mediabox_id'] = str(mediabox_id)

        if page_tree_leaves:
            try:
                page_id = (set(page_tree_leaves) & set(path_catalog_uri)).pop()
                tmp['page_id'] = page_id
                tmp['tentative_page_num'] = page_tree_leaves.index(page_id) + 1
            except KeyError:
                __LOGGER__.warning("No match between {} and {}: Page id not found.".format(page_tree_leaves, path_catalog_uri))
        else:
            tmp['page_id'] = None
            tmp['tentative_page_num'] = None


        uri_object = augmented_tree[uri_element]['object']
        match = re.match(ANNOTATION_RE, uri_object)
        if match:
            extracted_uri = match.group(2)
            rect_coordinates = match.group(1)
            rect_id = uri_element
        else:
            uri_match = re.match(URI_ONLY_RE, uri_object)
            if not uri_match:

                if uri_object.startswith(b'[') and  uri_object.endswith(b']'): # from here we'll get uri and rect for more than one uri probably
                    __LOGGER__.info('Processing object as an Array...')
                    try:
                        json_array = json.loads(
                            Object.parse(uri_object).to_json()
                        )
                        uri_df = pd.json_normalize(json_array)
                        attributes_urls_in_array = uri_df.apply(
                            lambda x: extract_info_array(x, uri_element),
                            axis=1).to_list()
                        uri_rects.extend(attributes_urls_in_array)
                    except (PdfError, json.JSONDecodeError):
                        __LOGGER__.warning('Processing failed. Moving on.')
                        pass
                else:
                    __LOGGER__.warning("Cannot extract URI from element.")
                continue

            else:
                extracted_uri = uri_match.group(1)
                # find the parent object having Rect
                for parent in reversed(path_catalog_uri):
                    match = re.match(ANNOTAION_WOUT_URI, augmented_tree[parent]['object'])
                    if match:
                        rect_coordinates = match.group(1)
                        rect_id = parent
                        break
                else:
                    __LOGGER__.warning("/URI is there but we could not find any /Rect for it !?")
                    continue

        tmp['uri_id'] = str(uri_element)
        try:
            tmp['uri'] = extracted_uri.decode('utf-8').replace("\x00", "").replace('(', '').replace(')', '').replace('/URI', '').strip()
        except UnicodeDecodeError:
            tmp['uri'] = 'UnicodeDecodeError'
            __LOGGER__.warning('UnicodeDecodeError when getting URI.')

        tmp['rect_id'] = str(rect_id)
        tmp['rect_raw'] = rect_coordinates

        uri_rects.append(tmp)
    return uri_rects


def extract_info_array(row, array_id):
    res = defaultdict(lambda : None)

    # we need to use this ridiculous way of accessing columns because we have no clue how they are going to be
    # named for each PDF element, especially due to the flattening of nested elements.
    # but we *assume* each list (`uri_col`, `has_annot`, `has_rect`) is either empty or of length 1, bc we process one ROW
    # at a time (see `apply`)
    uri_col = [x for x in row.index if 'URI' in x]
    if not uri_col:
        return res

    res['uri_id'] = array_id # the elements inside the array do not have an own element ID
    res['uri'] = row.loc[uri_col[0]].replace("\x00", "").replace('(', '').replace(')', '').replace('/URI', '').strip()

    has_rect = [x for x in row.index if 'Rect' in x]
    if has_rect:
        # row.loc[has_rect[0]] is most often a list... but not always, and this leads to unpredictable errors
        # so we stringify it and bring it back to bytes for compatibility with Annotation :\
        res['rect_raw']  = str(row.loc[has_rect[0]]).replace('[', '').replace(']', '').replace(',', '').encode('utf-8')
        res['rect_id'] = array_id  # the elements inside the array do not have an own element ID
    else:
        res['rect_raw'] = None
        res['rect_id'] = None

    return res


def aggregate_data(elements_with_uri):
    if not elements_with_uri:
        return defaultdict(lambda: None)

    res = defaultdict(int)

    df = pd.DataFrame.from_dict(elements_with_uri)
    df['clickable_area'] = df['clickable_area'].astype(int)
    df['mediabox_area'] = df['mediabox_area'].astype(int)
    areas_per_p_id = []
    for w, h in df['screenshot_w_h'].unique():
        areas_per_p_id.append(w * h)

    n_urls_per_page = []
    page_number_group = df.groupby('tentative_page_num')
    for g in page_number_group.groups:
        n_urls_per_page.append(len(page_number_group.get_group(g)['uri']))

    # res['filehash'] = set([x['filehash'] for x in elements_with_uri]).pop()
    res['tot_num_urls'] = len([x for x in elements_with_uri if x['uri'] != 'UnicodeDecodeError'])
    res['num_unique_urls'] = len(set([x['uri'] for x in elements_with_uri if x['uri'] != 'UnicodeDecodeError']))
    res['num_unparsable_urls'] = len([x for x in elements_with_uri if x['uri'] == 'UnicodeDecodeError'])
    res['n_uris_first_page'] = df[df['tentative_page_num']==1]['uri'].count()
    res['avg_uris_per_page'] = round(mean(n_urls_per_page), ndigits=2)
    if len (n_urls_per_page) >= 2 :
        res['stdev_uris_per_page'] = round(stdev(n_urls_per_page), ndigits=2)
    res['avg_click_area'] = df['clickable_area'].mean().round()
    res['avg_page_area'] = round(mean(areas_per_p_id), ndigits=0)

    res['num_urls_per_page'] = list(page_number_group['uri_id'].count().items())
    res['avg_click_area_per_page'] = list(page_number_group['clickable_area'].mean().round(decimals=0).items())
    res['perc_avg_clickable_over_avg_page_area'] = round(df['clickable_area'].mean() / res['avg_page_area'] * 100, ndigits=2)
    res['stddev_click_area_per_page'] = list(page_number_group['clickable_area'].std().round(decimals=0).items())
    return res

# need to handle case where screenshot does not exist !
def compute_clickable_areas(filehash, elements_with_uri, screenshot_width, screenshot_height):
    if not screenshot_height or not screenshot_width:
        __LOGGER__.warning("Cannot compute clickable area without screenshot width or height.")
        return []

    clickable_areas = []

    for clickable_uri_element in elements_with_uri:
        if not clickable_uri_element['mediabox_coord']:
            __LOGGER__.warning("Cannot compute clickable area without MediaBox information.")
            continue

        ann = Annotation(clickable_uri_element['uri'], clickable_uri_element['rect_raw'], screenshot_width,
                             screenshot_height, clickable_uri_element['mediabox_coord'][2], clickable_uri_element['mediabox_coord'][3])
        tmp_areas = {
            'filehash': filehash,
            'uri': ann.url,
            'uri_id': clickable_uri_element['uri_id'],
            'rect_id': clickable_uri_element['rect_id'],
            'rect_coord': (ann.scaled_llx, ann.translated_ury, ann.scaled_urx, ann.translated_lly),
            'mediabox_coord': clickable_uri_element['mediabox_coord'],
            'screenshot_w_h': (screenshot_width, screenshot_height),
            'page_id': clickable_uri_element['page_id'],
            'tentative_page_num': clickable_uri_element['tentative_page_num'],
            'clickable_area': ann.width * ann.height,
            'mediabox_area': clickable_uri_element['mediabox_coord'][2] * clickable_uri_element['mediabox_coord'][3]
        }
        clickable_areas.append(tmp_areas)
    return clickable_areas

def process_one_pdf(pdf_path, screenshot_width, screenshot_height):
    try:
        # trees = get_pdf_tree(pdf_path)
        graph = pdf_to_graph(pdf_path)
    except Exception as e:
        __LOGGER__.exception(e)
        return [], defaultdict(lambda: None)
    # graph = pdf_tree_to_graph(trees)
    # draw_graph(graph, DOT_LAYOUT, os.path.join(PDF_TREE_PATH, os.path.basename(pdf_path)))

    augmented_tree = create_augmented_tree(graph, os.path.join(PDF_TREE_PATH, os.path.basename(pdf_path)))
    elements_with_uri = find_uri_clickable_area(graph, augmented_tree)
    if len(elements_with_uri) == 0:
        __LOGGER__.warning("No clickable areas found. Returning...")
        return [], defaultdict(lambda: None)

    clickable_areas_and_uris = compute_clickable_areas(os.path.basename(pdf_path).replace('.pdf', ''), elements_with_uri, screenshot_width, screenshot_height)
    return clickable_areas_and_uris, aggregate_data(clickable_areas_and_uris)



def visualize_annots(clickable_areas_and_uris, pdf_path):
    copy_and_draw(pdf_path, [x['rect_coord'] for x in clickable_areas_and_uris if x['tentative_page_num'] == 1])

def flush_temporary_files(d):
    target_files = os.listdir(d)

    if len(target_files) < 1:
        return

    with Bar('Bar', max=len(target_files)) as bar:
        for file in target_files:
            if os.path.isfile(os.path.join(d, file)):
                os.remove(os.path.join(d, file))
            elif os.path.isdir(os.path.join(d, file)):
                shutil.rmtree(os.path.join(d, file))
            bar.next()

def evaluate_page_num():
    with psycopg2.connect(dbname='pipeline', user='postgres', password='postgres', host='127.0.0.1') as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                         SELECT filehash, uri, tentative_page_num from url_finegrained;""")
                res = cur.fetchall()
            except psycopg2.errors.Error as e:
                print(e)

    df = pd.DataFrame(res)
    no_files = df[~df[1].str.contains('file://') & ~df[1].str.contains('/C:')].to_numpy().tolist()
    random.shuffle(no_files)
    sampled = random.sample(no_files, k=100)


def main():
    config = load_config_yaml("/path/to/conf/config.yaml")

    db_bindings = config["global"]["postgres"]
    filesys_entrypoint = config['global']['file_storage']

    yesterday = (datetime.today() - timedelta(days=1)).date()

    __LOGGER__.info("Running with default setting: process PDFs from {}.\nFetching data...".format(yesterday.strftime("%Y-%m-%d")))
    with psycopg2.connect(dbname=db_bindings['databases']['pipeline'], user=db_bindings['users']['ginopino'], password=db_bindings['passwords']['ginopino'], host=db_bindings['host']) as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                SELECT DISTINCT ON (imported_samples.filehash) imported_samples.filehash, codphish.pages, codphish.screenshot_width, codphish.screenshot_height
                FROM imported_samples
                LEFT JOIN codphish USING (filehash)
                LEFT JOIN url_finegrained USING (filehash)
                WHERE imported_samples.mimetype = 'application/pdf' AND imported_samples.upload_date = %s AND url_finegrained.filehash IS NULL  AND ( imported_samples.provider = 'cisco' OR imported_samples.provider = 'inquest' )
                ORDER BY imported_samples.filehash;
                """, (yesterday, ))
                sampled_files = cur.fetchall()

            except psycopg2.errors.Error as e:
                __LOGGER__.error(e)
                sampled_files = []

    aggregate_results = []
    finegrained_results = []
    total = len(sampled_files)
    whole_analysis_timer = perf_counter()

    psycopg2.extensions.register_adapter(np.int64, psycopg2._psycopg.AsIs)
    db_wrapper = DbWrapper(db_bindings, 'pipeline', 'ginopino', True)

    for counter, (filehash, pages, screenshot_width, screenshot_height) in enumerate(sampled_files):
        pdf_path = os.path.join(filesys_entrypoint, 'samples', rel_path(filehash), filehash)
        __LOGGER__.info("{} / {}: Starting analysis for {}".format(counter, total, filehash))
        task_started = perf_counter()
        try:  # outer try catches exception thrown by resource consumption
            try:
                resource.setrlimit(resource.RLIMIT_VMEM,
                                   (31457280, 31457280))  # hopefully 30 gigs -- in kb, see man ulimit
            except AttributeError:
                # vmem may not be defined http://git.savannah.gnu.org/cgit/bash.git/tree/builtins/ulimit.def
                resource.setrlimit(resource.RLIMIT_AS,
                                   (30000000000, 30000000000))  # hopefully 30 gigs -- in bytes, see man setrlimit
            finegrained, aggregated = process_one_pdf(pdf_path, screenshot_width, screenshot_height)
        except ValueError as e:
            __LOGGER__.exception(e)
            finegrained, aggregated = [], defaultdict(lambda: None)

        __LOGGER__.info("Task took {:0.4f}".format(perf_counter() - task_started))
        aggregated['malicious'] = None #malicious
        aggregated['category'] = None #category
        aggregated['codph_pages'] = pages
        aggregated['filehash'] = filehash
        aggregate_results.append(aggregated)
        # finegrained_results.extend(finegrained)


        cursor = db_wrapper.get_cursor()

        finegrained_insert_stmt = """
            INSERT INTO url_finegrained ( filehash , uri_id, uri, rect_id, rect_coord, mediabox_coord, screenshot_w_h, page_id, tentative_page_num, clickable_area, mediabox_area )
            VALUES %s
            ON CONFLICT DO NOTHING;
            """

        try:
            cursor.execute("""
            INSERT INTO url_aggregated ( filehash, tot_num_urls, num_unique_urls, num_unparsable_urls, n_uris_first_page, avg_uris_per_page, stdev_uris_per_page, avg_click_area, avg_page_area, num_urls_per_page, avg_click_area_per_page, stddev_click_area_per_page, perc_avg_clickable_over_mbox_area, malicious, category, codph_pages )
            VALUES (%(filehash)s, %(tot_num_urls)s, %(num_unique_urls)s, %(num_unparsable_urls)s, %(n_uris_first_page)s, %(avg_uris_per_page)s, %(stdev_uris_per_page)s, %(avg_click_area)s, %(avg_page_area)s, %(num_urls_per_page)s, %(avg_click_area_per_page)s, %(stddev_click_area_per_page)s, %(perc_avg_clickable_over_mbox_area)s, %(malicious)s, %(category)s, %(codph_pages)s);
            """, aggregated)
            __LOGGER__.debug("    Stored aggregated record for {}".format(aggregated['filehash']))
            
            execute_values(cursor, finegrained_insert_stmt, finegrained,
                                  template="( %(filehash)s, %(uri_id)s, %(uri)s, %(rect_id)s, %(rect_coord)s, %(mediabox_coord)s, %(screenshot_w_h)s, %(page_id)s, %(tentative_page_num)s, %(clickable_area)s, %(mediabox_area)s )",
                                  page_size=100)
            if finegrained:
                __LOGGER__.debug("    Stored finegrained record for {}".format(finegrained[0]['filehash']))
        except psycopg2.errors.OperationalError as e1:
            __LOGGER__.exception(e1)
            __LOGGER__.info('Retrying insert...')
            db_wrapper.release_cursor()
            cursor = db_wrapper.get_cursor()

            try:
                cursor.execute("""
                INSERT INTO url_aggregated ( filehash, tot_num_urls, num_unique_urls, num_unparsable_urls, n_uris_first_page, avg_uris_per_page, stdev_uris_per_page, avg_click_area, avg_page_area, num_urls_per_page, avg_click_area_per_page, stddev_click_area_per_page, perc_avg_clickable_over_mbox_area, malicious, category, codph_pages )
                VALUES (%(filehash)s, %(tot_num_urls)s, %(num_unique_urls)s, %(num_unparsable_urls)s, %(n_uris_first_page)s, %(avg_uris_per_page)s, %(stdev_uris_per_page)s, %(avg_click_area)s, %(avg_page_area)s, %(num_urls_per_page)s, %(avg_click_area_per_page)s, %(stddev_click_area_per_page)s, %(perc_avg_clickable_over_mbox_area)s, %(malicious)s, %(category)s, %(codph_pages)s);
                """, aggregated)
                __LOGGER__.debug("    Stored aggregated record for {}".format(aggregated['filehash']))

                execute_values(cursor, finegrained_insert_stmt, finegrained,
                                      template="( %(filehash)s, %(uri_id)s, %(uri)s, %(rect_id)s, %(rect_coord)s, %(mediabox_coord)s, %(screenshot_w_h)s, %(page_id)s, %(tentative_page_num)s, %(clickable_area)s, %(mediabox_area)s )",
                                      page_size=100)
                if finegrained:
                    __LOGGER__.debug("    Stored finegrained record for {}".format(finegrained[0]['filehash']))
            except Exception as e:
                __LOGGER__.exception(e)
                db_wrapper.release_cursor()
                continue
        except psycopg2.errors.UniqueViolation as e2:
            __LOGGER__.exception(e2)
            pass
        except psycopg2.errors.Error as e:
            __LOGGER__.exception(e)
            sys.exit(-1)
    db_wrapper.release_cursor()

    __LOGGER__.info("Whole analysis took {:0.4f}".format(perf_counter() - whole_analysis_timer))

    # finegrained_df = pd.DataFrame.from_dict(finegrained_results)
    # aggregated_df = pd.DataFrame.from_dict(aggregate_results)

    # finegrained_df.to_csv(
    #     os.path.join(os.path.dirname(PDF_TREE_PATH), "{}-finegrained.csv".format(datetime.today().strftime('%Y-%m-%d'))),
    #     index=False)
    # aggregated_df.to_csv(
    #     os.path.join(os.path.dirname(PDF_TREE_PATH), "{}-aggregated.csv".format(datetime.today().strftime('%Y-%m-%d'))),
    #     index=False)

    # category_grouped_df = aggregated_df.groupby('category')
    # category_df = pd.DataFrame()
    # category_df['category'] = aggregated_df['category'].unique()
    # category_df.set_index('category', inplace=True)
    # category_df['n_inspected'] = pd.merge(category_df, category_grouped_df.size().to_frame(name='n_inspected'),
    #                                       how='outer', left_index=True, right_on='category')
    # try:
    #     category_df = pd.merge(category_df, category_grouped_df['malicious'].unique(), how='inner', on='category')
    #     category_df = pd.merge(category_df, category_grouped_df['tot_num_urls'].mean().round(decimals=2), how='inner', on='category')
    #     category_df = pd.merge(category_df, category_grouped_df['num_unique_urls'].mean().round(decimals=2), how='inner', on='category')
    #     category_df = pd.merge(category_df, category_grouped_df['n_uris_first_page'].mean().round(decimals=2), how='inner', on='category')
    #     category_df = pd.merge(category_df, category_grouped_df['avg_click_area'].mean().round(decimals=2), how='inner', on='category')
    #     category_df = pd.merge(category_df, category_grouped_df['avg_page_area'].mean().round(decimals=2), how='inner', on='category')
    #     category_df['perc_clickable_over_tot'] = round(category_df['avg_click_area'] / category_df['avg_page_area'] * 100, ndigits=2)
    # except KeyError as e:
    #     __LOGGER__.warn(e)
    #     pass
    # category_df.to_csv(os.path.join(
    #     os.path.dirname(PDF_TREE_PATH), "{}-category_aggregated.csv".format(datetime.today().strftime('%Y-%m-%d'))))
    time.sleep(5)
    __LOGGER__.info("Flushing data in " + PDF_TREE_PATH + "...!")
    flush_temporary_files(PDF_TREE_PATH)
    __LOGGER__.debug('Done.')


if __name__ == '__main__':
    main()
