from util.config import Config
config = Config()
from util.logger import Logger
logger = Logger()

from agent.AgentBase import AgentBase
from memory.MnistReplayMemory import MnistReplayMemory
import numpy as np
from random import random, randint

class DqnAgent(AgentBase):

    def reset(self, resetMemory=False):
        if resetMemory:
            self.memory.reset()
    
    def __init__(self, model, memory_size=2000, action_step=4, gamma=0.8, eps_greed=0.5, double_greed_every=1):
        self.double_greed_every = double_greed_every
        self.double_greed_count = 0
        self.eps_greedy = eps_greed
        self.action_step = action_step
        self.memory_size = memory_size
        self.gamma = gamma
        self.memory = MnistReplayMemory(max_size=memory_size)
        self.model = model
        super(DqnAgent, self).__init__()

    def act(self, state, eps_greedy=None):
        original_image, cur_image = state
        assert isinstance(cur_image, np.ndarray)
        row, col = cur_image.shape
        if eps_greedy is None:
            eps_greedy = self.eps_greedy
        if random() < eps_greedy:
            # greed
            # model should output w * h * 2(up or down)
            Q = self.model.predict(np.vstack((original_image.reshape(1, row, col), cur_image.reshape(1, row, col))))
            _, isUp, x, y = np.unravel_index(np.argmax(Q), Q.shape) # up or down, w, h
        else:
            x = randint(0, row-1)
            y = randint(0, col-1)
            isUp = randint(0, 1)
        # build new action image by applying changes
        action = np.zeros_like(cur_image)
        action[:] = cur_image[:]
        if isUp > 0.5:
            action[x, y] = min(255, action[x, y] + self.action_step)
        else:
            action[x, y] = max(0, action[x, y] - self.action_step)
        return action
    
    def record(self, _state, action, reward, state_, isFinish):
        self.memory.append((_state, action, reward, state_, isFinish))

    def train(self, batchSize=None, epoch=10):
        if not self.memory.isFull():
            logger.info("Memory not full ({}/{}), skip training.".format(str(self.memory.size()), str(self.memory.max_size)))
            return
        if batchSize is None:
            batchSize = self.memory_size // epoch
        for e in range(epoch):
            samples = self.memory.sample(batchSize)
            row, col = samples[0][0][0].shape
            # states: batchsize * original channel & current channel * w * h 
            states = np.zeros((batchSize, 2, row, col))
            # action_masks: batchsize * up or down * w * h
            action_masks = np.zeros((batchSize, 2, row, col))
            # rewards: batchsize * up or down * w * h
            rewards = np.zeros((batchSize, 2, row, col))
            for i in range(len(samples)):
                states[i, 0, :, :] = samples[i][0][0] 
                states[i, 1, :, :] = samples[i][0][1]
                action_masks[i, 0, :, :] = (samples[i][1] - samples[i][0][0]) > 0
                action_masks[i, 1, :, :] = (samples[i][1] - samples[i][0][0]) < 0
                if samples[i][4]:
                    # finished
                    rewards[i, 0, :, :] = samples[i][2] * ((samples[i][1] - samples[i][0][0]) > 0)
                    rewards[i, 1, :, :] = samples[i][2] * ((samples[i][1] - samples[i][0][0]) < 0)
                else:
                    Q = samples[i][2] + self.gamma * self.model.predict(states[i]).max() # np.inf?
                    if Q == np.inf or Q == -np.inf or Q == np.nan:
                        logger.warning("Invalid Q value: {} encountered.".format(str(Q)))
                        Q = 0
                    rewards[i, 0, :, :] = Q * ((samples[i][1] - samples[i][0][0]) > 0) 
                    rewards[i, 1, :, :] = Q * ((samples[i][1] - samples[i][0][0]) < 0)
            self.model.train(states, action_masks, rewards)
        # update eps-greed
        self.double_greed_count += 1
        if (self.double_greed_count + 1) % self.double_greed_every == 0:
            logger.info("eps_greedy updated {} -> {}".format(str(self.eps_greedy), str(1.0 - (1.0 - self.eps_greedy) / 2)))
            self.eps_greedy = 1.0 - (1.0 - self.eps_greedy) / 2
    def __str__(self):
        return "{}({})".format(self.__class__.__name__, str(self.model))