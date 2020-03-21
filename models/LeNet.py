from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F


class LeNet1(nn.Module):
    def __init__(self):
        super(LeNet1, self).__init__()
        self.conv1 = nn.Conv2d(1,4, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(4, 12, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(588, 10)
    def forward(self, x, index=-1, metric=0):
        layer = F.relu(self.conv1(x))
        layer = F.max_pool2d(layer, 2)
        layer = F.relu(self.conv2(layer))
        layer = F.max_pool2d(layer, 2)

        layer = layer.view(-1, 588)
        layer = self.fc1(layer)
        output = F.log_softmax(layer, dim=1)
        return output

class LeNet3(nn.Module):
    def __init__(self):
        super(LeNet3, self).__init__()
        self.conv1 = nn.Conv2d(1,6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2)

        self.fc1 = nn.Linear(784, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):

        layer0 = F.relu(self.conv1(x))
        layer1 = F.max_pool2d(layer0, 2)
        layer2 = F.relu(self.conv2(layer1))
        layer3 = F.max_pool2d(layer2, 2)

        layer_ = layer3.view(-1, 784)
        layer4 = F.relu(self.fc1(layer_))
        layer5 = self.fc2(layer4)
        output = F.log_softmax(layer5, dim=1)
        return output
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1,6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(784, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):

        layer0 = F.relu(self.conv1(x))
        layer1 = F.max_pool2d(layer0, 2)
        layer2 = F.relu(self.conv2(layer1))
        layer3 = F.max_pool2d(layer2, 2)

        layer_ = layer3.view(-1, 784)
        layer4 = F.relu(self.fc1(layer_))
        layer5 = F.relu(self.fc2(layer4))
        layer6 = self.fc3(layer5)
        output = F.log_softmax(layer6, dim=1)

        return output