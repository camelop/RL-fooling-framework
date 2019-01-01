class EnvBase(object):

    def reset(self):
        raise NotImplementedError

    def __init__(self):
        self.success = False # must implement
        self.turn = 0 # must implement
        self.reset()
    
    def update(self, action):
        raise NotImplementedError

    def getState(self):
        raise NotImplementedError