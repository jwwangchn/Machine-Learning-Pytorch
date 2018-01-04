import torch
import torch.nn as nn


class LeNet5_v1(nn.Module):
    def __init__(self, in_channels=3):
        """ LeNet5 architecture for CIFAR10, use convolutional replace fully connection


        :param in_channels:
            in_channels: image channels for input, 32*32*1 or 32*32*3

        """
        super(LeNet5_v1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        in_size = x.size(0)
        x = self.relu(self.maxpool(self.conv1(x)))
        x = self.relu(self.maxpool(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = x.view(in_size, -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LeNet5_v2(nn.Module):
    def __init__(self, in_channels=3):
        """ LeNet5 architecture for CIFAR10, use fully connection layer


        :param in_channels:
            in_channels: image channels for input, 32*32*1 or 32*32*3, the heigth and eidth must be 32

        """
        super(LeNet5_v2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        in_size = x.size(0)
        x = self.relu(self.maxpool(self.conv1(x)))
        x = self.relu(self.maxpool(self.conv2(x)))
        x = x.view(in_size, -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def build_lenet5_v1(in_channels = 3):
    return LeNet5_v1(in_channels)

def build_lenet5_v2(in_channels = 3):
    return LeNet5_v2(in_channels)