# -*- coding: utf-8 -*-
from __future__ import print_function, division
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from net_definition import Net


torch.cuda.manual_seed(1)


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=64, shuffle=False, num_workers=1, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=64, shuffle=False, num_workers=1, pin_memory=True)


def train(epoch):
    model.train()
    for batch, (data, target) in enumerate(train_loader):
        data, target = autograd.Variable(data.cuda()), autograd.Variable(target.cuda())
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch % 10 == 0:
            msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch / len(train_loader),
                                                                           loss.data[0])
            print(msg)


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = autograd.Variable(data.cuda(), volatile=True), autograd.Variable(target.cuda())
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        predict = output.data.max(1)[1]
        correct += predict.eq(target.data).cpu().sum()
    test_loss /= len(test_loader)
    msg = 'Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss,
                                                                           correct,
                                                                           len(test_loader.dataset),
                                                                           100. * correct / len(test_loader.dataset))
    print(msg)


model = Net()
model.cuda()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)


for i in range(1):
    train(i)
    test()
    torch.save(model, '../model/torch_%03d.model' % i)


weights = model.state_dict()
print(weights.keys())
print(weights['fc2.bias'])


# load saved model
# model2 = torch.load('../model/torch_000.model')
# weights2 = model2.state_dict()
# print(weights2.keys())
# print(weights2['fc2.bias'])






















