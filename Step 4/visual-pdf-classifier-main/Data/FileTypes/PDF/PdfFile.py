import json
import os
import pathlib
import re
import subprocess
import sys
import time
from collections import deque, defaultdict
from os.path import join, basename

import fitz
from pathlib import Path
from PIL import Image
from networkx.drawing.nx_agraph import graphviz_layout, write_dot
from pdf2image import convert_from_path
import numpy as np

from Data.Datalake.Phishing.Phishing import file_hash_decode
from Data.FileTypes.BaseFile import BaseFile
from Utilities.Logger.Logger import Logger
import networkx as nx

ANNOTATION_RE = re.compile(
    b'<<(?=.*?/Subtype\s*?/Link)(?=.*?/Rect\s*?\[(.*?)\])(?=.*?/Type\s*?/Annot)(?=.*?/URI\s*?\((.*?)\)).*?>>$', re.S)
ANNOTAION_WOUT_URI = re.compile(b'<<(?=.*?/Subtype\s*?/Link)(?=.*?/Rect\s*?\[(.*?)\])(?=.*?[/Type\s*?/Annot|/A]).*?>>$',
                                re.S)
URI_ONLY_RE = re.compile(b'<<.*(/URI\s*\((.*?)\)).*?>>', re.S)
MEDIABOX_RE = re.compile(b'^(?=.*?/Page)(?=.*?/MediaBox\s*?\[(.*?)\]).*?$', re.S)
KIDS_RE = re.compile(b'<<(?=.*?/Page)(?=.*?Kids\s*?\[(.*?)\])(?=.*?Count\s*?(\d+)).*?>>$', re.S)
PAGE_TREE_ROOT_RE = re.compile(b'<<(?=.*?/Page)(?=.*?Kids\s*?\[(.*?)\])(?=.*?Count\s*?(\d+))(?!.*?/Parent.*?).*?>>$',
                               re.S)
OBJ_ATTR_RE = re.compile(r'(?P<attr>/[a-zA-Z]+) ([^\n].*?)[\n|\r|\f]', re.S)
OBJ_REFERENCES_RE = re.compile(r'([0-9]* [0-9]* R)*', re.S)

class PdfFile(BaseFile, Logger):

    def __init__(self, path):
        """
        :param path: path to the file we are working on
        :param tmp_folder: folder where to save necessary temoprary files
        """
        super().__init__(path)
        self._peep_pdf_path = os.path.join(str(pathlib.Path(__file__).parent.resolve()),
                                           "./PeepPDF/peepdf_interface.py")
        self.tmp_folder = os.path.join(os.environ["CACHE"],
                                       os.path.basename(self.path).split(".")[0] + "-" + str(time.time()))
        os.makedirs(self.tmp_folder)

    @property
    def hash_path(self):
        return file_hash_decode(basename(self.path))

    def _compute_graph(self):
        """
        Compute the graph of the pdf
        :return: path to the file in which the graph has been saved
        """

        basename = os.path.basename(self.path)
        output_path = os.path.join(self.tmp_folder, basename.replace('.pdf', '') + '.json')
        peepdf_path = os.path.join(os.path.abspath(os.environ["PEEP-PDF"]), 'peepdf_interface.py')
        cmd = ['/usr/bin/python2', peepdf_path, '--pdf', self.path, '--output', output_path]
        subprocess.run(cmd)

        tree_candidates = [os.path.join(self.tmp_folder, x) for x in os.listdir(os.path.dirname(output_path)) if
                           basename in x and not os.path.isdir(os.path.join(self.tmp_folder, x))]
        if len(tree_candidates) < 1:
            self.logger_module.error(output_path + " does not seem to exist. Was there any error calling PEEPDF?")
            self.logger_module.warning(
                "If AttributeError: 'NoneType' object has no attribute 'getRawValue' in e.message:\tXref table is probably corrupted. Falling back to muPDF.")
            raise Exception("Error in reading PDF.")

        trees = {}
        for tree_path in tree_candidates:
            version = os.path.basename(tree_path).split("_v")[1].replace('.json', '')
            with open(tree_path, 'r') as f:
                pdf_tree = json.load(f)
            trees[version] = pdf_tree
        if len(trees) < 1:
            self.logger_module.error("PDF trees were not loaded correctly.")
            raise Exception("Error in reading PDF.")

        versions_sorted = sorted(list(trees.keys()), reverse=True)
        graph = unfold_tree(trees, versions_sorted, self.path,self.logger_module)

        return graph

    def print_graph(self, output_path):
        """
        Print the graph of the PDF
        :param output_path: path where to print the image
        :return:
        """
        if not os.path.exists(Path(output_path).parent):
            os.makedirs(Path(output_path).parent)
        dot_filepath = output_path.split(".")[0] + ".dot"
        write_dot(self._compute_graph(), dot_filepath)

        cmd = ['dot', '-o', output_path, '-Tpdf', dot_filepath]
        p = subprocess.Popen(cmd)
        p.communicate()

    def print_interactable_mask(self, page_shape, output_path, first_page_only=True, scale_ratio=1):
        """
        Print a binary mask of the areas of the file that are interactable
        :return:
        """

        graph = self._compute_graph()
        augmented_tree = create_augmented_tree(graph, os.path.join(self.tmp_folder, os.path.basename(self.path)),
                                               self.logger_module)
        interactable_areas = find_uri_clickable_area(graph, augmented_tree, self.logger_module)

        mask = np.zeros(page_shape)

        for area in interactable_areas:

            if first_page_only and area["tentative_page_num"] is not None and area["tentative_page_num"] > 1:
                continue

            rect_coordinates = area["rect_raw"].decode().split(" ")
            rect_coordinates = [r for r in rect_coordinates if r != ""]
            coordinates = tuple(map(int, map(float, rect_coordinates))) * scale_ratio

            mask[page_shape[0] - coordinates[3]:page_shape[0] - coordinates[1],
            coordinates[0]:coordinates[2]] = 1

        Image.fromarray((mask * 255).astype(np.uint8)).save(output_path)

    def print_page(self, page_number, desired_output_path, dpi=72):
        """
        Print a page as an image
        :param page_number: number of the page to print
        :param desired_output_path: path where to save the image
        :return:
        """

        process = subprocess.Popen(
            ["pdftoppm", "-f", str(page_number), "-l", str(page_number),
             "-rx", str(dpi), "-ry", str(dpi), "-png", self.path, desired_output_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        output, error = process.communicate()
        error = error.decode('utf-8').strip()

        tool_output_path = None

        if Path(desired_output_path + f"-{page_number}.png").exists():
            tool_output_path = desired_output_path + f"-{page_number}.png"
        elif Path(desired_output_path + f"-0{page_number}.png").exists():
            tool_output_path = desired_output_path + f"-0{page_number}.png"
        elif Path(desired_output_path + f"-00{page_number}.png").exists():
            tool_output_path = desired_output_path + f"-00{page_number}.png"
        else:
            raise ValueError

        os.rename(tool_output_path, desired_output_path)
        return True

def normalize_id(id):
    return str(id)

def unfold_tree(trees, versions_sorted, pdf_path,logger):
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
                # __LOGGER__.debug("Still " + str(len(queue)) + " elements in the queue.")
                current_node = queue.popleft()
                if current_node in visited_nodes:
                    continue
                # __LOGGER__.debug("Current node is {}.\n".format(current_node))
                try:
                    neighbours = subtree[current_node][1]
                    visited_nodes.add(current_node) # i.e. this node has an entry in the tree
                except KeyError:
                    logger.debug("Node {} has no entry in the tree. This is version {}.".format(current_node, version))
                    catalog_obj = find_catalog_obj(current_node, pdf_path,logger)
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
                logger.debug("Adding {} yet not visited nodes.".format(len(remaining_nodes)))

                for isolated_node in remaining_nodes:
                    subtree = trees[version][0][1]
                    assert ( isolated_node in subtree.keys() )
                    visited_nodes.add(isolated_node)
                    neighbours = subtree[isolated_node][1]
                    for neighbour in neighbours:
                        norm_neighbour = normalize_id(neighbour)
                        graph.add_edge(isolated_node, norm_neighbour)
        logger.debug("Unfolded tree version " + str(version))

    return graph

def create_augmented_tree(graph, path, logger):
    res = {}
    id_to_tag = nx.get_node_attributes(graph, 'tag')
    id_to_version = nx.get_node_attributes(graph, 'version')

    for element_id, version in id_to_version.items():
        obj_path = os.path.join(path, "{}_v{}.txt".format(element_id, version))
        if not os.path.exists(obj_path):
            raise Exception('Missing file {} in {}!'.format("{}_v{}.txt".format(element_id, version), path))
        with open(obj_path, 'rb') as obj_file:
            res[element_id] = {
                'attribute': id_to_tag[element_id],
                'object': obj_file.read(),
                'version': version
            }
    return res


def find_catalog_id(augmented_tree, logger):
    for key, val in augmented_tree.items():
        if '/Catalog' in val['attribute']:
            return key
    logger.warning('Found no /Catalog in this tree!')


def find_uri_clickable_area(graph, augmented_tree, logging_module):
    uri_rects = []

    try:
        _, page_tree_leaves = identify_page_tree(augmented_tree, logging_module)
    except Exception:
        page_tree_leaves = None

    catalog_id = find_catalog_id(augmented_tree, logging_module)

    uri_elements = [k for k, v in augmented_tree.items() if re.search(URI_ONLY_RE, v['object'])]
    for uri_element in uri_elements:
        tmp = defaultdict(lambda: None)
        # path_catalog_uri = nx.shortest_path(graph, catalog_id, uri_element)[:-1]  # can already remove the last element (URI)
        try:
            path_catalog_uri = attempt_find_path(graph, augmented_tree, catalog_id, uri_element, logging_module)
        except Exception:
            logging_module.debug("No path to Catalog for URI element " + uri_element)
            continue
        uri_object = augmented_tree[uri_element]['object']
        match = re.match(ANNOTATION_RE, uri_object)
        if match:
            extracted_uri = match.group(2)
            rect_coordinates = match.group(1)
            rect_id = uri_element
        else:
            uri_match = re.match(URI_ONLY_RE, uri_object)
            if not uri_match:
                logging_module.warning("Cannot extract URI from element.")
                raise Exception
            extracted_uri = uri_match.group(1)
            # find the parent object having Rect
            for parent in reversed(path_catalog_uri):
                match = re.match(ANNOTAION_WOUT_URI, augmented_tree[parent]['object'])
                if match:
                    rect_coordinates = match.group(1)
                    rect_id = parent
                    break
            else:
                logging_module.warning("/URI is there but we could not find any /Rect for it !?")
                continue

        tmp['uri_id'] = str(uri_element)
        try:
            tmp['uri'] = extracted_uri.decode('utf-8').replace('(', '').replace(')', '').replace('/URI', '').strip()
        except UnicodeDecodeError:
            tmp['uri'] = 'UnicodeDecodeError'
            logging_module.warning('UnicodeDecodeError when getting URI.')

        tmp['rect_id'] = str(rect_id)
        tmp['rect_raw'] = rect_coordinates
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
                logging_module.warning(
                    "No match between {} and {}: Page id not found.".format(page_tree_leaves, path_catalog_uri))
        else:
            tmp['page_id'] = None
            tmp['tentative_page_num'] = None

        uri_rects.append(tmp)
    return uri_rects


def attempt_find_path(graph, augmented_tree, catalog_id, uri_element, logger):
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
        for path in nx.all_simple_paths(graph, catalog_id, uri_element, cutoff=15):
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
        logger.warning("Unable to find Annot and MediaBox elements in shortest&simple paths ({} -> {}, cutoff=15)."
                       .format(uri_element, catalog_id))
        raise Exception("Element not found.")


def find_mediabox_for_uri(path_catalog_uri, augmented_tree, logger):
    for element_id in reversed(path_catalog_uri):
        coordinates = re.findall(MEDIABOX_RE, augmented_tree[element_id]['object'])
        if coordinates:
            x, y, w, h = list_to_coord(coordinates)[0]
            if w <= 0 or h <= 0:
                continue
            else:
                return (element_id, (x, y, w, h))
    else:
        logger.error("Unable to find MediaBox with area > 0 for this sample.")
        raise Exception("Element not found.")


def list_to_coord(str_list):
    if len(str_list) == 0:
        return []

    rectangles = []
    for string in str_list:
        splits = [float(x.strip()) for x in string.split() if len(x.strip()) > 0]
        rectangles.append(tuple([round(vertex) for vertex in splits]))

    return rectangles


def identify_page_tree(augmented_tree, logger):
    pages = {k: v for k, v in augmented_tree.items() if "Page" in v['attribute']}
    kids_ids = []
    page_tree_root = set()
    for k, v in pages.items():
        match = re.match(PAGE_TREE_ROOT_RE, v['object'])
        if match:
            reference = match.group(1).decode('utf-8').strip().split('R')
            kids_ids.extend([x.strip().split(' ')[0] for x in reference if len(x) > 0])
            page_tree_root.add(k)
            page_count_value = int(match.group(2))
    assert (len(page_tree_root) == 1)

    kids_queue = deque(kids_ids)
    while len(kids_queue) > 0:
        first_kid = kids_queue.popleft()
        try:
            match = re.match(KIDS_RE, pages[first_kid]['object'])
        except KeyError as e:
            logger.error("Inconsistency found in the page tree.")
            raise Exception(e)
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
        logger.warning("PDF Page Tree Root node declared {} page leaves, but {} were observed!".format(page_count_value,
                                                                                                       len(kids_ids)))

    return page_id, kids_ids


def find_catalog_obj(subtree_id, pdf_path,logger):
    if subtree_id == 'None':
        logger.error("Cannot look for None object key.")
        return []

    pdf = fitz.open(pdf_path)
    mupdf_catalog = pdf.pdf_catalog()
    if subtree_id != str(mupdf_catalog):
        logger.debug('Object {} (peepdf) does not seem to be the Catalog (mupdf: {}).'.format(subtree_id, mupdf_catalog))
        if subtree_id == '0':
            logger.error('Trying to reference object 0 (Bad xref)!')
        return []
    else:
        catalog_obj = pdf.xref_object(mupdf_catalog)
        attributes = OBJ_ATTR_RE.findall(catalog_obj)

        for match in attributes:
            if '/Type' in match[0] and '/Catalog' in match[1]:
                logger.debug('/Catalog found!')
                return attributes
        else:
            logger.warning('No /Catalog found with mupdf.')
            return []
