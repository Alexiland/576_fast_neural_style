from __future__ import print_function
import os
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


from vgg_quant import VGG

DATASET = 'cifar10'
cuda = True


def updateBN(model, sparsity):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(sparsity * torch.sign(m.weight.data))  # L1


def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))


def DataPreprocess(batchsize=64, test_batch_size=256, usegpu=True, seed=1, DATASET='cifar10'):
    torch.manual_seed(seed)
    memorysetting = False
    cuda = usegpu and torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed(seed)
        memorysetting = True
    DATASET = DATASET
    # if not os.path.exists(savepath):
    #     os.makedirs(savepath)

    if DATASET == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data.cifar10', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.Pad(4),
                                 transforms.RandomCrop(32),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                             ])),
            batch_size=batchsize, shuffle=True, pin_memory=memorysetting)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])),
            batch_size=test_batch_size, shuffle=True, pin_memory=memorysetting)
    else:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.Pad(4),
                                  transforms.RandomCrop(32),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                              ])),
            batch_size=batchsize, shuffle=True, pin_memory=memorysetting)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])),
            batch_size=test_batch_size, shuffle=True, pin_memory=memorysetting)
    return train_loader, test_loader


def PreTrain(save_path, usegpu=True,
             start_epoch=0, epochs=90, lr=0.1, momentum=0.9, weightdecay=1e-4):

    model = VGG('VGG16')
    cuda = usegpu and torch.cuda.is_available()

    if cuda:
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weightdecay)

    # start Training
    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            pred = output.data.max(1, keepdim=True)[1]
            loss.backward()

            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data))

    def test():
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                if cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').data  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        return correct / float(len(test_loader.dataset))

    best_prec1 = 0.
    for epoch in range(start_epoch, epochs):
        if epoch in [epochs * 0.5, epochs * 0.75]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        train(epoch)
        prec1 = test()
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, filepath=save_path)

    print("Best accuracy: " + str(best_prec1))


train_loader, test_loader = DataPreprocess()

if __name__ == "__main__":
    PreTrain("./logs")