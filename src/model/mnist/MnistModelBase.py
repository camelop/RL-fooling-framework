from model.ModelBase import ModelBase

class MnistModelBase(ModelBase):
    label_set = [str(i) for i in range(10)]

    def predict(self, image, prob=False):
        raise NotImplementedError