import torch
import torchvision
import torchvision.transforms as transform

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from lenet import build_lenet5_v1, build_lenet5_v2
import argparse

import os

parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
args = parser.parse_args()



args.cuda = not args.disable_cuda and torch.cuda.is_available()

BATCH_SIZE = 8
LEARN_RATE = 0.01
MOMENTUM = 0.9

if args.cuda:
    print("Cuda is available!")
    torch.cuda.set_device(1)

else:
    print("Cuda is not available!")

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
net = build_lenet5_v2()

if args.cuda:
    net.cuda()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Cross Entropy loss
optimizer = optim.SGD(net.parameters(), lr=LEARN_RATE, momentum=MOMENTUM)

if __name__ == '__main__':
    # Train the network
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            if args.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
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
            if i % 1000 == 999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0

    print('Finish Training')

    # TODO: Save weights and load weights

    # save the model's parameter
    torch.save(net.state_dict(), './weights/LeNet5_CIFAR10_Params.pkl')

    # TODO: Move train and test parse to lenet.py file

    # evaluate the model
    correct = 0
    class_correct = list(0.0 for i in range(10))
    total = 0
    class_total = list(0.0 for i in range(10))
    for data in testloader:
        images, labels = data
        if args.cuda:
            images, labels = images.cuda(), labels.cuda()
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        if args.cuda:
            predicted = predicted.cuda()
            # Calculator total performance
        total += labels.size(0)
        correct += (predicted == labels).cpu().sum()  # 将 GPU 上的运算移动到 CPU 上

        # Calculator class performance
        c = (predicted == labels).cpu().squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

    print("Accuracy of the network on the 10000 test images: %d %%" % (100 * correct / total))

    print("Finish Evaluating!")


# import matplotlib.pyplot as plt
# import numpy as np
# def imshow(img):
#     img = img * 0.5 + 0.5
#     npimg = img.numpy()
#
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#
#
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
#
# plt.figure()
# imshow(torchvision.utils.make_grid(images))
# plt.show()