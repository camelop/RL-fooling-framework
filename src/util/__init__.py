import os
import datetime
from util.logger import Logger
logger = Logger()

def getTimeStr():
    t = datetime.datetime.now()
    return "{}-{}-{}-{}-{}-{}".format(str(t.year), str(t.month), str(t.day), str(t.hour), str(t.minute), str(t.second))

def check_file(loc, fail_message=None):
    if fail_message is None:
        fail_message = "File not downloaded or not generated yet, please check your code."
    if os.path.exists(loc):
        return True
    else:
        logger.error("{} not found: {}".format(loc, fail_message))
        return False

def read_dict(loc):
    if not check_file(loc):
        logger.error("read_dict cannot read {}".format(loc))
        return
    d = {}
    index = 0
    with open(loc, 'r') as f:
        for line in f.readlines():
            line = line.strip("\n")
            if len(line) == 0:
                continue
            d[line] = index
            index += 1
    return d