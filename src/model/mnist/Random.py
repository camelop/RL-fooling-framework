from model.mnist.MnistModelBase import MnistModelBase
import random

class Random(MnistModelBase):
    
    def predict(self, im):
        return MnistModelBase.label_set[random.randint(0, len(MnistModelBase.label_set)-1)]