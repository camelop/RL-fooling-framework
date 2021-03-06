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
            # init default logger
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
            # answer logger
            self.answer = logging.getLogger("lrDialogSys_ans")
            self.answer.setLevel(logging.INFO)
            self.fh_ans = logging.FileHandler(config.logging_ans_loc)
            self.fh_ans.setLevel(logging.INFO)
            self.ch_ans = logging.StreamHandler(sys.stdout)
            self.ch_ans.setLevel(logging.INFO)
            formatter_ans = logging.Formatter('# [%(asctime)s][%(levelname)s]:\n%(message)s')
            self.fh_ans.setFormatter(formatter_ans)
            self.ch_ans.setFormatter(formatter_ans)
            self.answer.addHandler(self.fh_ans)
            self.answer.addHandler(self.ch_ans)
            # single instance
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
    
    def record(self, message):
        self.answer.info(message)