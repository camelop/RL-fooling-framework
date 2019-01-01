'''LeNet in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..MnistModelBase import MnistModelBase

class LeNetModule(nn.Module):
    def __init__(self):
        super(LeNetModule, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

class LeNet(MnistModelBase):
    def __init__(self):
        print('==> Loading {} model'.format('LeNet'))
        self.net = LeNetModule()
        self._loadModel('LeNet')