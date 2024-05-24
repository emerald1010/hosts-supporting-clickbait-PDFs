'''
Created on 14 Sep 2010

@author: gianko
'''

from logging import *
import coloredlogs
import os, time
# from codphish.paths import path_parsed_data


# def paths(): # TODO refactor s.t. log location in FS is handled by this class and not by single modules !!
#     path_todays_analysis = os.path.join(path_parsed_data, time.strftime('%Y-%m-%d'))
#     if not os.path.exists(path_todays_analysis):
#         os.mkdir(path_todays_analysis)
#     logfile = os.path.join(path_todays_analysis, "{}_ServerSideLog.txt".format(time.strftime('%Y-%m-%d')))
#     return logfile

def getlogger(component, level, filename=None, stdout=True):
    # create logger
    logger = getLogger(component)
    logger.setLevel(level)

    # create formatter
    formatter = Formatter(fmt="[%(asctime)s] %(name)s (%(levelname)s) %(message)s", datefmt='%d%m%Y-%I:%M%p')

    # create console handler and set level to debug
    if stdout:
        ch = StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)

    if filename:
        # fh = FileHandler(paths() if filename == 'codphishrest' else filename, encoding='utf-8')
        fh = FileHandler(filename, encoding='utf-8')
        fh.setLevel(level)
        fh.setFormatter(formatter)

    # add ch to logger
    if not len(logger.handlers):
        if stdout: logger.addHandler(ch)
        if filename: logger.addHandler(fh)

    coloredlogs.install(level, logger=logger)

    return logger
