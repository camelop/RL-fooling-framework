class MnistAgentModelBase(object):

    def __init__(self):
        pass

    def predict(self, state):
        ''' params:
        [state] 2 (origin&current) * w * h
        output:
        [Q_value] 2 (up&down) * w * h
        '''
        raise NotImplementedError
    
    def train(self, state, action_mask, reward):
        ''' params:
        [state] bs * 2 (origin&current) * w * h
        [action_mask] bs * 2 (up&down) * w * h
        [reward] bs * 2 (up&down) * w * h
        '''
        raise NotImplementedError
    
    def __str__(self):
        return self.__class__.__name__