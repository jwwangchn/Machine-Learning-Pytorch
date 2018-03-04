import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import lenet as lenet
import argparse

import matplotlib.pyplot as plt
import random
import os

parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
args = parser.parse_args()


args.cuda = not args.disable_cuda and torch.cuda.is_available()

BATCH_SIZE = 8
LEARN_RATE = 0.01
MOMENTUM = 0.9
EPOCHS = 2


if args.cuda:
    print("Cuda is available!")
    torch.cuda.set_device(1)
else:
    print("Cuda is not available!")

def data_loader():
    # Define transform
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Set trainset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Set dataloader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Set classes name
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainset, testset, trainloader, testloader, classes


def show_image(trainset, testset):
    X, y = trainset.train_data, trainset.train_labels
    idx = random.randint(0, 19)
    plt.imshow(X[idx])
    plt.title('Class: %i' % y[idx])
    plt.show()


def train(trainloader, net, criterion, optimizer):
    # Train the network
    net.train()
    train_loss = 0.0
    train_acc = 0.0
    train_total = 0.0

    for i, (inputs, labels) in enumerate(trainloader):
        if args.cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        # zero the parameter gradients
        inputs_var, labels_var = Variable(inputs), Variable(labels)
        optimizer.zero_grad()

        # use the network to generate output
        outputs = net(inputs_var)
        loss = criterion(outputs, labels_var)
        # use loss to backward
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        if args.cuda:
            predicted = predicted.cuda()
            # Calculator total performance

        train_loss += loss.data[0]
        train_acc += (predicted == labels).cpu().sum()  # 将 GPU 上的运算移动到 CPU 上
        train_total += labels.size(0)

    train_loss = train_loss / len(trainloader)
    train_acc = train_acc / train_total

    return train_loss, train_acc, train_total


def test(testloader, net, criterion):
    net.eval()
    # evaluate the model
    test_loss = 0.0
    test_acc = 0.0
    test_total = 0.0
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))

    for i, (inputs, labels) in enumerate(testloader):
        if args.cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        inputs_var, labels_var = Variable(inputs), Variable(labels)

        outputs = net(inputs_var)
        loss = criterion(outputs, labels_var)

        _, predicted = torch.max(outputs.data, 1)
        if args.cuda:
            predicted = predicted.cuda()

        # Calculator class performance

        test_loss += loss.data[0]
        test_acc += (predicted == labels).cpu().sum()  # 将 GPU 上的运算移动到 CPU 上
        test_total += labels.size(0)

        c = (predicted == labels).cpu().squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    test_loss = test_loss / len(testloader)
    test_acc = test_acc / test_total
    return test_loss, test_acc, test_total, class_correct, class_total

if __name__ == '__main__':

    # Define the convolution neural network
    net = lenet.LeNet5(in_channels=3, model_name='LeNet5_v1', nclasses=10)

    if args.cuda:
        net.cuda()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Cross Entropy loss
    optimizer = optim.SGD(net.parameters(), lr=LEARN_RATE, momentum=MOMENTUM)

    trainset, testset, trainloader, testloader, classes = data_loader()
    show_image(trainset, testset)

    for epoch in range(EPOCHS):

        train_loss, train_acc, train_total = train(trainloader, net, criterion, optimizer)
        print('Epoch: %3d/%3d, Train Loss: %5.5f, Train Acc: %5.5f' % (epoch, EPOCHS, train_loss, 100*train_acc))

        test_loss, test_acc, test_total, class_correct, class_total = test(testloader, net, criterion)
        print('Epoch: %3d/%3d, Test Loss: %5.5f, Test Acc: %5.5f' % (epoch, EPOCHS, test_loss, 100*test_acc))
        # optimizer, learning_rate = adjust_learning_rate(optimizer, learning_rate, epoch)

        if epoch == EPOCHS - 1:
            print("Accuracy of the network on the 10000 test images: %d %%" % (100 * test_acc))
            for i in range(10):
                print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))


    print('Finish training and testing')
    # save the model's parameter
    torch.save(net.state_dict(), './weights/LeNet5_CIFAR10_Params.pkl')

    # import matplotlib
    # matplotlib.use('GTKAgg')
    # import matplotlib.pyplot as plt
    # import numpy as np
    # plt.plot(np.arange(100))
    # plt.show()


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
