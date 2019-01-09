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


# from model.mnist.models.lenet import LeNet
# lenet = LeNet()
def test_classifier(classifier, dataset=test_mnist()):
    import matplotlib.pyplot as plt
    import numpy as np
    while True:
        data, label = dataset.getExample(useRandom=True)
        result, probs = classifier.predict(data, prob=True)
        fig, (ax_origin_image, ax_probs) = plt.subplots(1, 2, figsize=(8, 4))
        ax_origin_image.set_title("Testing {}".format(classifier.__class__.__name__))
        ax_origin_image.imshow(data, cmap="Greys", vmax=255, vmin=0)
        ax_probs.set_title("Predict: {}/Truth: {}".format(str(result), str(label)))
        n_groups = 10
        bar_width =0.35
        index = np.arange(n_groups)
        prob_value = np.array([probs[i] for i in sorted(probs.keys())])
        ax_probs.bar(index, prob_value, bar_width, label="Probability")
        ax_probs.set_xticks(index)
        ax_probs.set_xticklabels(sorted(probs.keys()))
        fig.tight_layout()
        plt.show()

def test_mnist_replay_memory():
    from random import randint
    from memory.MnistReplayMemory import MnistReplayMemory
    mrm = MnistReplayMemory(max_size=5)
    def generate_s_a_r_s():
        return (randint(0,5), randint(0,5), randint(0,5), randint(0,5), randint(0,5))
    for i in range(10):
        nw = generate_s_a_r_s()
        print("append ", str(nw))
        mrm.append(nw)
        print(mrm)

def test_saving_net():
    from agent.mnist.model.CNNDQN import CNNDQN
    import numpy as np
    c = CNNDQN()
    c.predict(np.random.random((2, 28, 28)))
    c.save()

test_saving_net()

# test_mnist_replay_memory()