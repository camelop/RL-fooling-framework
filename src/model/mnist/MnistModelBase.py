class MnistModelBase(object):

    label_set = [str(i) for i in range(10)]

    def predict(self, im):
        raise NotImplementedError