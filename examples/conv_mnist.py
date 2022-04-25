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
network = EP(costfunc=None,
            etol=1e-3, beta=0.1, max_iter=2000)
network.addlayerednodes(4, True,data_init=torch.nn.init.zeros_)
# network.connect(0, 1, Conv2d(3, 10,3,pos_call=(lambda x: torch.flatten(x,start_dim=1)),padding=1,bias=False))
# network.connect(1, 2, Linear(10240, 10))

network.connect(0,1,Conv2d(3, 32, kernel_size=(3, 3),pos_call=(lambda x: F.max_pool2d(x,2)),bias=True))
network.connect(1,2,Conv2d(32, 64, kernel_size=(3, 3),pos_call=(lambda x: F.max_pool2d(x,2)),bias=True))
network.connect(2,3,Conv2d(64, 128, kernel_size=(3, 3),pos_call=(lambda x: torch.flatten(F.max_pool2d(x,2),start_dim=1)),bias=True))
# network.connect(2,3,Conv2d(64, 10,pos_call=(lambda x: torch.flatten(x,start_dim=1)),kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=True))
#network.connect(3,4,Conv2d(512, 512,pos_call=(lambda x: torch.flatten(x,start_dim=1)), kernel_size=(3, 3), stride=(1, 1)))
out_linear=Linear(in_features=512, out_features=10, bias=True).to(device)
out_linear.pre = network.nodes[-1]
network.costfunc = lambda x,y : torch.nn.CrossEntropyLoss(reduction='none')(out_linear(x),y)


network.to(device)
network.initall(input_shape=torch.Size([batch_size,3,32,32]),device=device)

network.node_optim = torch.optim.SGD(network.nodes.parameters(),lr=0.05)
network.edge_optim = torch.optim.ASGD(torch.nn.ModuleList([*network.edges,out_linear]).parameters(),lr=0.01)
epoches = 10

accuracy = 0.

x,y = next(trainiter)
x=x.cuda()
y=y.float().cuda()

network.outnode = Node(state=y)
out,_,_ = network.infer(x,reset=True,beta=0)
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
            out,e_last,e_diff = network.infer(x,reset=True,beta=0)
            pre_cost = torch.mean(network.cost())
            print(f'\nStep:{i}\nout 1st:{out_linear()[0]}\ny 1st:{y[0]}\n elast:{e_last}, e_diff:{e_diff},infer_cost:{pre_cost.item()}')
            network.edge_optim.zero_grad()
            network.node_optim.zero_grad()
        network.three_phase_update(x,y)
    correct = 0
    total = 0
    t_x,t_y = next(testiter)
    t_x=t_x.cuda()
    t_y=t_y.float().cuda()
    out,e_last,e_diff = network.infer(t_x,reset=True,beta=0)
    # Compute test batch accuracy, energy and store number of seen batches
    correct += float(torch.sum(torch.argmax(out,1) == t_y.argmax(dim=1)))
    total += x.size(0)
    print(f'\n\nEpoch {epoch} test batch accuracy: {correct/total}')
    pre_cost = torch.mean(network.cost())
    network.edge_optim.zero_grad()
    network.node_optim.zero_grad()
    accuracy = correct/total
