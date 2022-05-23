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
import time
import pickle
from datetime import datetime
# from einops import rearrange, reduce, repeat # from einops.layers.torch import Rearrange, Reduce

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

batch_size = 512
test_batch_size = 512

class PatchEmbedding(torch.nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 4, emb_size: int = 48):
        self.patch_size = patch_size
        super().__init__()
        self.projection = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),

        )
        #self.cls_token = torch.nn.Parameter(torch.randn(1, 1, emb_size))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, _, _ = x.shape

        x = self.projection(x)
        #cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        #x = torch.cat([cls_tokens, x], dim=1)
        return x

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
                                transforms.Normalize((0.,), (0.4081,)),
                            ]),
                            target_transform=_one_hot_ten
                            )

cifar10_test = datasets.CIFAR10('/data/215/Program/Equilibrium-Propagation-master/cifar10_pytorch', train=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.,), (0.4081,)),
                            ]),
                            target_transform=_one_hot_ten
                            )


kwargs = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}
trainldr=torch.utils.data.DataLoader(
    cifar10_train, batch_size=batch_size, drop_last=True, shuffle=True, **kwargs)

testldr = torch.utils.data.DataLoader(
    cifar10_test, batch_size=test_batch_size, drop_last=True, shuffle=False, **kwargs)
trainiter = iter(trainldr)
testiter = iter(testldr)

# Initialize EP network
network = EP(etol=1e-1, beta=1, max_iter=200)
# network.addlayerednodes(5, True,data_init=lambda x: torch.nn.init.normal_(x,std=0.1),activation=torch.tanh)
network.addlayerednodes(5, True,data_init=torch.nn.init.zeros_,activation=torch.tanh)

network.connect(0,1,Sequential(Conv2d(3, 128, kernel_size=(3, 3),padding=1,stride=1),BatchNorm2d(128)))
network.connect(1,2,Sequential(Conv2d(128, 256, kernel_size=(5, 5),padding=2,stride=1),BatchNorm2d(256),MaxPool2d(2)))
network.connect(2,3,Sequential(Conv2d(256, 128, kernel_size=(5, 5),padding=2),BatchNorm2d(128),MaxPool2d(2)))
network.connect(3,4,Sequential(Conv2d(128, 100, kernel_size=(3, 3),padding=1),BatchNorm2d(100),MaxPool2d(2)))
#patching
# network.connect(0,1,EdgeBuilder(PatchEmbedding,in_channels= 3, patch_size= 2, emb_size = 96))
# network.connect(1,2,EdgeBuilder(PatchEmbedding,in_channels= 96, patch_size= 2, emb_size = 200))
# network.connect(2,3,EdgeBuilder(PatchEmbedding,in_channels= 200, patch_size= 2, emb_size = 400))
# network.connect(3,4,EdgeBuilder(PatchEmbedding,in_channels= 400, patch_size= 2, emb_size = 1024))
# network.connect(4,5,EdgeBuilder(PatchEmbedding,in_channels= 1024, patch_size= 2, emb_size = 512))
# network.connect(5,6,Sequential(Flatten(),Linear(512,10)))
# network.connect(4,6,Sequential(MaxPool2d(2),Flatten(),Linear(256,10)))
external_link = Sequential(Flatten(),Linear(1600,10)).to(device)
external_link.pre = network.nodes[-1]
del network.costfunc
celoss = torch.nn.CrossEntropyLoss(reduction='none')

def costfunc(x,y):
    return celoss(external_link(x),y)

network.costfunc = costfunc
network.extra_edges.append(external_link)

# network.connect(2,3,Conv2d(64, 10,pos_call=(lambda x: torch.flatten(x,start_dim=1)),kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=True))
#network.connect(3,4,Conv2d(512, 512,pos_call=(lambda x: torch.flatten(x,start_dim=1)), kernel_size=(3, 3), stride=(1, 1)))

network.to(device)
network.initall(input_shape=torch.Size([batch_size,3,32,32]),device=device)

network.node_optim = torch.optim.Adam(network.nodes.parameters(),lr=0.001)
network.edge_optim = torch.optim.ASGD(torch.nn.ModuleList([*network.edges,external_link]).parameters(),lr=1e-5)
epoches = 50

accuracy = 0.
repeat = 3

x,y = next(trainiter)
x=x.cuda()
y=y.float().cuda()

network.outnode = Node(state=y)
out,_,_ = network.infer(x,reset=True,beta=0)
pre_cost = torch.mean(network.cost())
print(f'pre cost is {pre_cost}')
F = open(f'/data/locallearn/examples/lr1e-5/cifar_cifar10_norm_{datetime.now().date()}_{datetime.now().time()}.log','a+')
F.write(f'{network} with beta {network.beta}')
network.epoches = 0

for epoch in range(epoches):
    start_time = time.time()
    trainldr=torch.utils.data.DataLoader(
        cifar10_train, batch_size=batch_size, drop_last=True, shuffle=True, **kwargs)
    trainiter = iter(trainldr)
    print(f'One epoch has {len(trainiter)} batches')
    for i in range(len(trainiter)-1):
    # for i in range(2):
        #x,y = next(trainiter)
        x, y = next(trainiter)
        x=x.cuda()
        y=y.float().cuda()
        network.outnode = Node(state=y)
        if i%20==0:
            out,e_last,e_diff = network.infer(x,reset=True,beta=0)
            pre_cost = torch.mean(network.cost())
            print(f'\nStep:{i}\nout 1st:{external_link(network.nodes[-1]())[0]}\ny 1st:{y[0]}\n elast:{e_last}, e_diff:{e_diff},infer_cost:{pre_cost.item()}')
            F.write(f'\nStep:{i}\nout 1st:{external_link(network.nodes[-1]())[0]}\ny 1st:{y[0]}\n elast:{e_last}, e_diff:{e_diff},infer_cost:{pre_cost.item()}')
            network.edge_optim.zero_grad()
            network.node_optim.zero_grad()
        network.three_phase_update(x,y,repeat=repeat,beta=network.beta)
        # network.edge_optim.zero_grad()
        # network.node_optim.zero_grad()
        print(f'complete the {i} step')
    correct = 0
    total = 0
    t_x,t_y = next(testiter)
    t_x=t_x.cuda()
    t_y=t_y.float().cuda()
    out,e_last,e_diff = network.infer(t_x,reset=True,beta=0)
    output = external_link(network.nodes[-1]())
    # Compute test batch accuracy, energy and store number of seen batches
    correct += float(torch.sum(torch.argmax(output,1) == t_y.argmax(dim=1)))
    total += x.size(0)
    print(f'\n\nEpoch {epoch} test batch accuracy: {correct/total}')
    F.write(f'\n\nEpoch {epoch} test batch accuracy: {correct/total}')
    pre_cost = torch.mean(network.cost())
    network.edge_optim.zero_grad()
    network.node_optim.zero_grad()
    accuracy = correct/total
    F.write(f'\ntime{time.time()-start_time}')
    pickle.dump(network,open(f'/data/locallearn/examples/lr1e-5/cifar_cifar10_norm_epoch{epoch}_{datetime.now().date()}_{datetime.now().time()}.pth',"wb"))
F.close()

