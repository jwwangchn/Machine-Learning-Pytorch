import torch
import torchvision
import torchvision.transforms as transform

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from lenet import *


BATCH_SIZE = 4
LEARN_RATE = 0.001
MOMENTUM = 0.9

# Define transform
transform = transform.Compose([transform.ToTensor(), transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Set trainset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Set dataloader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Set classes name
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Define the convolution neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1,
                               padding=0)  # input:32*32*3 output:28*28*6
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1,
                               padding=0)  # input:14*14*6 output:10*10*16
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)  # input:5*5*16 output:120
        self.fc2 = nn.Linear(in_features=120, out_features=84)  # input:120 output:84
        self.fc3 = nn.Linear(in_features=84, out_features=10)  # inpuit:84 output:10

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # input:32*32*3 output:14*14*6
        x = self.pool(F.relu(self.conv2(x)))  # input:14*14*6 output:5*5*16
        x = x.view(-1, 16 * 5 * 5)  # resize: batch_size * (16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Cross Entropy loss
optimizer = optim.SGD(net.parameters(), lr=LEARN_RATE, momentum=MOMENTUM)

if __name__ == '__main__':
    # Train the network
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # use the network to generate output
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            # use loss to backward
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finish Training')
