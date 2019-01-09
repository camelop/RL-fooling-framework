from util.config import Config
config = Config()
from util.logger import Logger
logger = Logger()
from util import getTimeStr

from agent.mnist.model.MnistAgentModelBase import MnistAgentModelBase

import numpy as np
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn


class CNNDQN(MnistAgentModelBase):

    def __init__(self, learning_rate=0.01, diff_state=True, load_from=None, save_after_every=50):
        self.diff_state = diff_state
        self.learning_rate = learning_rate
        self.save_counter = 0
        self.save_after_every = save_after_every
        self.net = nn.Sequential()
        self.net.add(
            nn.BatchNorm(),
            nn.Conv2D(channels=8, kernel_size=3, padding = 1, activation='relu'),
            nn.BatchNorm(),
            nn.Conv2D(channels=4, kernel_size=7, padding = 3, activation='relu'),
            nn.BatchNorm(),
            # nn.Conv2D(channels=4, kernel_size=3, padding = 1, activation='relu'),
            nn.Conv2D(channels=2, kernel_size=1, padding = 0, activation='relu'),
            )
        def try_gpu():
            try:
                ctx = mx.gpu()
                _ = nd.zeros((1,), ctx=ctx)
            except mx.base.MXNetError:
                ctx = mx.cpu()
            return ctx
        self.ctx = try_gpu()
        self.net.initialize(force_reinit=True, ctx=self.ctx, init=init.Xavier(magnitude=0.01))
        self.loss = gloss.L2Loss()
        self.trainer = gluon.Trainer(self.net.collect_params(), 'sgd', {'learning_rate': learning_rate})
        if load_from is not None:
            self.load(load_from)

    def predict(self, state):
        channel, row, col = state.shape
        if self.diff_state:
            state[0] = np.abs(state[0] - state[1])
        state = nd.array(state.reshape(1, channel, row, col)).as_in_context(self.ctx)
        return self.net(state).asnumpy()

    def train(self, state, action_mask, reward, epoch=3):
        if self.diff_state:
            state[:, 0] = np.abs(state[:, 0] - state[:, 1])
        state = nd.array(state).as_in_context(self.ctx)
        action_mask = nd.array(action_mask).as_in_context(self.ctx)
        reward = nd.array(reward).as_in_context(self.ctx)
        losses = []
        for e in range(epoch):
            with autograd.record():
                reward_hat = self.net(state) * action_mask
                l = self.loss(reward_hat, reward).sum()
            losses.append(l.asscalar())
            l.backward()
            self.trainer.step(state.shape[0])
        logger.info("loss: {}".format(str(losses)))
        self.save_counter += 1
        if (self.save_counter+1) % self.save_after_every == 0:
            self.save()

    def save(self, loc=None):
        if loc is None:
            loc = "{}/{}[{}]-{}.mxparam".format(config.mnist_agent_model_checkpoint_dir, str(self), self.save_counter, getTimeStr())
        logger.info("Saving net to {}".format(loc))
        self.net.save_parameters(loc)
        logger.info("Saved net to {}".format(loc))

    def load(self, loc):
        self.net.load_parameters(loc)