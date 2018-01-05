import torch
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self, in_channels=3, model_name='LeNet5_v1', nclasses=10):
        """
        :param in_channels:
            in_channels: image channels for input, 32*32*1 or 32*32*3
            model_name: LeNet5_v1 or LeNet5_v2, two versions of LeNet

        """
        super(LeNet5, self).__init__()
        if model_name == 'LeNet5_v1':
            # LeNet5 architecture for CIFAR10, use convolutional replace fully connection
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1, padding=0)
            self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
            self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1)
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.relu = nn.ReLU()
            self.fc1 = nn.Linear(in_features=120, out_features=84)
            self.fc2 = nn.Linear(in_features=84, out_features=10)
            self.model_name = 'LeNet5_v1'

        if model_name == 'LeNet5_v2':
            # LeNet5 architecture for CIFAR10, use fully connection layer
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, padding=0)  # input:32*32*3 output:28*28*6
            self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)  # input:14*14*6 output:10*10*16
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.relu = nn.ReLU()
            self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)  # input:5*5*16 output:120
            self.fc2 = nn.Linear(in_features=120, out_features=84)  # input:120 output:84
            self.fc3 = nn.Linear(in_features=84, out_features=10)  # inpuit:84 output:10
            self.model_name = 'LeNet5_v2'

    def forward(self, x):
        if self.model_name == 'LeNet5_v1':
            in_size = x.size(0)
            x = self.relu(self.maxpool(self.conv1(x)))
            x = self.relu(self.maxpool(self.conv2(x)))
            x = self.relu(self.conv3(x))
            x = x.view(in_size, -1)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)

        if self.model_name == 'LeNet5_v2':
            in_size = x.size(0)
            x = self.relu(self.maxpool(self.conv1(x)))  # input:32*32*3 output:14*14*6
            x = self.relu(self.maxpool(self.conv2(x)))  # input:14*14*6 output:5*5*16
            x = x.view(-1, 16 * 5 * 5)  # resize: batch_size * (16*5*5)
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
        return x
