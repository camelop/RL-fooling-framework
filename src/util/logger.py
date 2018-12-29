import logging
import sys

from util.config import Config
config = Config()

class Logger(object):

    __instance = None
    __init = False

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = object.__new__(cls)
        return cls.__instance

    def __init__(self):
        if not Logger.__init:
            # init logger
            self.logger = logging.getLogger("lrDialogSys")
            self.logger.setLevel(logging.DEBUG)
            self.fh = logging.FileHandler(config.logging_loc)
            self.fh.setLevel(logging.DEBUG)
            self.ch = logging.StreamHandler(sys.stderr)
            self.ch.setLevel(logging.INFO)
            formatter_f = logging.Formatter('[%(asctime)s][%(levelname)s]: %(message)s')
            self.fh.setFormatter(formatter_f)
            formatter_c = logging.Formatter('[%(asctime)s][%(levelname)s]: %(message)s')
            self.ch.setFormatter(formatter_c)
            self.logger.addHandler(self.fh)
            self.logger.addHandler(self.ch)
            print("Logger init", file=sys.stderr)
            Logger.__init = True
    
    def debug(self, message):
        self.logger.debug(message)
    
    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def critical(self, message):
        self.logger.critical(message)