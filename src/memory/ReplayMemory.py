class ReplayMemory(object):
    
    def reset(self):
        raise NotImplementedError

    def __init__(self):
        self.reset()
    
    def append(self, sample):
        raise NotImplementedError
    
    def sample(self, size):
        raise NotImplementedError