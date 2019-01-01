from util.config import Config
config = Config()
from util.logger import Logger
logger = Logger()

from util.similarity import punish_ssim
from env.EnvBase import EnvBase
from util.dataset.mnist import MnistDataset
from model.mnist.Random import Random as random_classifier
from model.mnist.models.lenet import LeNet as lenet_classifier


import random
import numpy as np


class MnistClassifierEnv(EnvBase):

    def reset(self):
        self.turn = 0
        self.finish = False
        self.success = False
        self.original_pic = None
        self.current_pic = None
        self.true_label = None
        self.current_punish = 0

    def __init__(self, model=lenet_classifier(), sim_punish_model=punish_ssim, sim_punish_weight=1, success_reward=10, max_turn=1000, turn_punish=0.01):
        super(MnistClassifierEnv, self).__init__()
        self.model = model
        self.dataset = MnistDataset(config.mnist_train_image_loc, config.mnist_train_label_loc, config.mnist_test_image_loc, config.mnist_test_label_loc)
        self.sim_punish_model = sim_punish_model
        self.sim_punish_weight = sim_punish_weight
        self.success_reward = success_reward
        self.turn_punish = turn_punish
        self.max_turn = max_turn
    
    def update(self, new_pic):
        '''return state, reward, isFinish'''
        state = (self.original_pic, self.current_pic)
        reward = 0
        if self.finish: # then do no updates
            return state, reward, self.finish
        # check shape
        assert new_pic.shape == self.original_pic.shape
        # compute reward
        # -- turn punish
        reward -= self.turn_punish
        # -- similarity punish
        sim_punish = self.sim_punish_model(self.original_pic, new_pic)
        reward -= (sim_punish - self.current_punish) * self.sim_punish_weight
        self.current_punish = sim_punish
        # -- check if fooling success and update state
        cur_label = self.model.predict(new_pic)
        if cur_label != self.true_label:
            # fooling success
            self.success = True
            self.finish = True
            reward += self.success_reward
        self.turn += 1
        if self.turn >= self.max_turn:
            # reach max_turn, failed
            self.finish = True
        self.current_pic = new_pic
        state = (self.original_pic, self.current_pic)
        return state, reward, self.finish
    
    def getState(self):
        if self.turn == 0:
            # randomly pick a picture and a goal
            img_set = self.dataset.train_image
            index = random.randint(0, len(img_set)-1)
            img = np.array(img_set[index]).reshape((self.dataset.train_row, self.dataset.train_col))
            label = str(self.dataset.train_label[index])
            self.setGoal(img, label)
        return self.original_pic, self.current_pic

    def setGoal(self, original_pic, true_label):
        assert isinstance(original_pic, np.ndarray)
        self.reset()
        self.original_pic = original_pic
        self.current_pic = original_pic
        self.true_label = true_label
