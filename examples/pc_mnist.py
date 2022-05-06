#!/usr/bin/env python

import torch
import sys
import os
from torchvision import datasets, transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.node import *
from modules.edge import *
from modules.network import *
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

batch_size = 64

# Construct data loader
def _one_hot_ten(label):
    return torch.nn.functional.one_hot(torch.tensor(label), num_classes=10)

mnist_train = datasets.MNIST('data', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,)),
                                ]),
                                target_transform=_one_hot_ten
                                )

mnist_test = datasets.MNIST('data', train=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,)),
                            ]),
                            target_transform=_one_hot_ten
                            )


cifar10_train = datasets.CIFAR10('/data/215/Program/Equilibrium-Propagation-master/cifar10_pytorch', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,)),
                            ]),
                            target_transform=_one_hot_ten
                            )

cifar10_test = datasets.CIFAR10('/data/215/Program/Equilibrium-Propagation-master/cifar10_pytorch', train=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,)),
                            ]),
                            target_transform=_one_hot_ten
                            )


kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
trainldr=torch.utils.data.DataLoader(
    cifar10_train, batch_size=batch_size, drop_last=True, shuffle=True, **kwargs)

testldr = torch.utils.data.DataLoader(
    cifar10_test, batch_size=batch_size, drop_last=True, shuffle=False, **kwargs)
trainiter = iter(trainldr)
testiter = iter(testldr)

# Initialize EP network
network = PC(etol=1e-3, beta=1., max_iter=2000)
network.addlayerednodes(4, True,data_init=torch.nn.init.zeros_)
# network.connect(0, 1, Conv2d(3, 10,3,pos_call=(lambda x: torch.flatten(x,start_dim=1)),padding=1,bias=False))
# network.connect(1, 2, Linear(10240, 10))

network.connect(0,1,Sequential(Conv2d(3, 32, kernel_size=(3, 3),bias=True),torch.nn.MaxPool2d(2),torch.nn.ReLU()))
network.connect(1,2,Sequential(Conv2d(32, 64, kernel_size=(3, 3),bias=True),torch.nn.MaxPool2d(2),torch.nn.ReLU()))
network.connect(2,3,Sequential(Flatten(),Linear(2304,10,bias=True)))


network.to(device)
network.initall(input_shape=torch.Size([batch_size,3,32,32]),device=device)

network.node_optim = torch.optim.SGD(network.nodes.parameters(),lr=0.05)
network.edge_optim = torch.optim.ASGD(network.edges.parameters(),lr=0.01)
epoches = 10

accuracy = 0.

x,y = next(trainiter)
x=x.cuda()
y=y.float().cuda()

network.innode = Node(x)
network.outnode = Node(state=y)
network.feedforward(network.innode)
pre_cost = torch.mean(network.cost())
print(f'pre cost is {pre_cost}')
for epoch in range(epoches):
    trainiter = iter(trainldr)
    print(len(trainiter))
    for i in range(500):
        x,y = next(trainiter)
        x=x.cuda()
        y=y.float().cuda()
        network.outnode = Node(state=y)
        if i%20==0:
            network.innode = Node(state=x)
            network.feedforward(network.innode)
            pre_cost = torch.mean(network.cost())
            print(f'\nStep:{i}\nout 1st:{network.nodes[-1]()[0]}\ny 1st:{y[0]}\n infer_cost:{pre_cost.item()}')
            network.edge_optim.zero_grad()
            network.node_optim.zero_grad()
        network.one_phase_update(x,y)
    correct = 0
    total = 0
    t_x,t_y = next(testiter)
    t_x=t_x.cuda()
    t_y=t_y.float().cuda()
    network.innode = Node(state=t_x)
    network.feedforward(network.innode)
    out = network.nodes[-1]()
    # Compute test batch accuracy, energy and store number of seen batches
    correct += float(torch.sum(torch.argmax(out,1) == t_y.argmax(dim=1)))
    total += t_x.size(0)
    print(f'\n\nEpoch {epoch} test batch accuracy: {correct/total}')
    pre_cost = torch.mean(network.cost())
    network.edge_optim.zero_grad()
    network.node_optim.zero_grad()
    accuracy = correct/total
