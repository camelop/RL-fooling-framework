import random
from .MnistModelBase import MnistModelBase

class Random(MnistModelBase):
    def predict(self, im, prob=False):
        if prob:
            _probs = [random.random() for i in range(10)]
            probs = [_probs[i] * 1.0 / sum(_probs) for i in range(10)]
            return self.label_set[random.randint(0, len(self.label_set)-1)], dict(zip([str(i) for i in range(10)], probs))
        else:
            return self.label_set[random.randint(0, len(self.label_set)-1)]
    
    def __str__(self):
        return "MnistRandomModel"