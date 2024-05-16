#! /usr/bin/python2

import argparse
import os, sys
import json
import log
from PDFCore import PDFParser


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Python2 interface to get data from PEEPDF')
    parser.add_argument('--pdf', type=str, metavar='PATH', action='store', required='True',
                        help='Absolute path of the file to analyze.')
    parser.add_argument('--output', type=str, metavar='PATH', action='store', required='True',
                        help='Absolute path where to save the output.')

    args = parser.parse_args()
    logger = log.getlogger('peepdf-interface', level=log.INFO)
    if not os.path.exists(args.pdf):
        logger.error("The PDF path provided does not seem to exist.")
        sys.exit(-1)
    if os.path.exists(args.output):
        logger.info("Path " + args.output + " already exists. Going to overwrite that!")
    pdf_parser = PDFParser()
    logger.debug("Parsing with forceMode enabled by default.")
    status, parsed_pdf = pdf_parser.parse(args.pdf, forceMode=True)

    if status != 0:
        logger.error("There was an error with processing the file. Details of errors by parser:")
        logger.info(parsed_pdf.getErrors())
        logger.error("Exiting the process.")
        sys.exit(-1)

    versions = parsed_pdf.getNumUpdates() + 1
    logger.debug("This PDF file has {} versions.".format(versions))
    for current_version in range(versions):
        pdf_tree = parsed_pdf.getTree(current_version)
        path_curr_version = "{}_v{}.json".format(args.output.split('.json')[0], current_version)
        with open(path_curr_version, 'w+') as f:
            json.dump(pdf_tree, f)
        logger.debug("PDF tree saved successfully at " + path_curr_version + ".")

        pdf_objs_dict = pdf_tree[0][1]

        objects_path = os.path.join(os.path.dirname(args.output), os.path.basename(args.pdf))
        if not os.path.exists(objects_path):
            os.mkdir(objects_path)
        logger.debug("Now saving " + str(len(pdf_objs_dict.keys())) + " PDF objects in " + objects_path + ".")
        for object_id in pdf_objs_dict.keys():
            try:
                file_text = parsed_pdf.getObject(object_id, current_version).toFile()
            except Exception as e:
                logger.error("Error when retrieving the object through PeePDF.")
                sys.exit(-1)
            out_path = os.path.join(objects_path, str(object_id) + "_v" + str(current_version) + '.txt')
            with open(out_path, 'wb+') as f_out:
                f_out.write(file_text)
        logger.debug("PDF objects saved successfully at " + objects_path)

    sys.exit(0)
