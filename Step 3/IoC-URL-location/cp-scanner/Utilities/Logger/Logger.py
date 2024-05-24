import logging
import os
import sys
from os.path import join


class Logger:
    """
    This class implements a usefull interface to add logging functionalities easily to any of its children
    """
    log_file_path = ""
    ready = False
    @ staticmethod
    def setup(log_file_path="", console_debug_level=logging.WARNING):
        """
            :param console_debug_level: minimu level of importance the messages have to be to be printed in the console
        """
        if Logger.ready:
            return

        Logger.ready = True
        Logger.log_file_path = log_file_path

        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        # Log all data into the console.log file
        file_handler = logging.FileHandler(filename=log_file_path, mode="a")
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

        # In the console print only messages having a level equal or above console_debug_level
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_debug_level)

        logger.addHandler(console_handler)

    @property
    def logger_module(self):
        name = '.'.join([__name__, self.__class__.__name__])
        return logging.getLogger(name)
