import sys
import datetime

def getDateStr():
    t = datetime.datetime.now()
    return "{}-{}-{}".format(str(t.year), str(t.month), str(t.day))

class Config(object):

    __instance = None
    __init = False


    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = object.__new__(cls)
        return cls.__instance

    def __init__(self, d={}):
        if not Config.__init:
            # directories
            self.logging_loc = "logs/test/{}.log".format(getDateStr())
            self.logging_ans_loc = "logs/test/{}_ans.log".format(getDateStr())
            self.mnist_train_image_loc = "dataset/original/train-images.idx3-ubyte"
            self.mnist_train_label_loc = "dataset/original/train-labels.idx1-ubyte"
            self.mnist_test_image_loc = "dataset/original/t10k-images.idx3-ubyte"
            self.mnist_test_label_loc = "dataset/original/t10k-labels.idx1-ubyte"
            self.trajectory_save_dir = "logs/trajectory"
            
            self.mnist_agent_model_checkpoint_dir = "src/agent/mnist/model/checkpoint"
            Config.__init = True
            print("Configuration init", file=sys.stderr)
        # update configurations
        for k, v in d.items():
            setattr(self, k, v)
