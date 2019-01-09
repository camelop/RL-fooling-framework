from memory.ReplayMemory import ReplayMemory
import random

class MnistReplayMemory(ReplayMemory):

    def reset(self):
        self.index = 0
        self.memory = [] 

    def __init__(self, max_size=1000):
        self.max_size = max_size
        super(MnistReplayMemory, self).__init__()

    def append(self, sample):
        assert isinstance(sample, tuple) and len(sample) == 5 # _s, a, r, s_, isFinish
        if not self.isFull():
            self.memory.append(sample)
        else:
            self.memory[self.index] = sample # Round Robin
        # update self.index
        self.index += 1
        if self.index >= self.max_size:
            self.index = 0

    def sample(self, size):
        assert size <= self.size()
        return random.sample(self.memory, size)
    
    def size(self):
        return len(self.memory)
    
    def isFull(self):
        return self.size() == self.max_size

    def __str__(self):
        ret = "{} size={}/{} :\n".format(self.__class__.__name__, str(self.size()), str(self.max_size))
        for r in self.memory:
            ret = ret + str(r) + '\n'
        return ret