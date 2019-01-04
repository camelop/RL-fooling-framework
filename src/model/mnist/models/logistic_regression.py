'''LogisticRegression in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
from ..MnistTorchClassifierBase import MnistTorchClassifierBase


class LogisticRegressionModule(nn.Module):
    def __init__(self):
        super(LogisticRegressionModule, self).__init__()
        self.linear = nn.Linear(28 * 28, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        out = self.sigmoid(self.linear(x))
        out = F.softmax(out, dim=1)
        return out

class LogisticRegression(MnistTorchClassifierBase):
    def __init__(self):
        print('==> Loading {} model'.format('LogisticRegression'))
        self.net = LogisticRegressionModule()
        self._loadModel('LogisticRegression')