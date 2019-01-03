class AgentBase(object):

    def reset(self):
        raise NotImplementedError

    def __init__(self):
        self.reset()

    def act(self, state):
        '''return an action'''
        raise NotImplementedError
    
    def record(self, _state, action, reward, state_):
        pass # since this is not necessary for rule based agent
    
    def train(self):
        pass

    def __str__(self):
        return self.__class__.__name__