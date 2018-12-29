from util.config import Config
config = Config()
from util.logger import Logger
logger = Logger()

def test_logger():
    logger.info("test logger :D")

def test_mnist():
    from util.dataset.mnist import MnistDataset
    mnist = MnistDataset(config.mnist_train_image_loc, config.mnist_train_label_loc, config.mnist_test_image_loc, config.mnist_test_label_loc)
    return mnist

def test_mnist_model_random():
    from model.mnist.Random import Random
    return Random()

def test_funcname():
    import sys
    print(sys._getframe())

test_funcname()