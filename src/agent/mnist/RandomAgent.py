from agent.AgentBase import AgentBase

import numpy as np
import random

class RandomAgent(AgentBase):

    def reset(self):
        self.original_image = None
    
    def __init__(self, pixel_change_max=32):
        super(RandomAgent, self).__init__()
        self.pixel_change_max = pixel_change_max

    def act(self, state):
        original_image, cur_image = state
        assert isinstance(cur_image, np.ndarray)
        row, col = cur_image.shape
        # select random pixel
        x = random.randint(0, row-1)
        y = random.randint(0, col-1)
        # randomly add a small change
        new_image = np.zeros((row, col))
        new_image[:][:] = cur_image[:][:]
        def littleChange(pixel):
            change = random.randint(-self.pixel_change_max, self.pixel_change_max)
            return min(255, max(0, pixel+change))
        new_image[x, y] = littleChange(new_image[x, y])
        # return new image
        return new_image