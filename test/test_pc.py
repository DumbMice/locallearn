#!/usr/bin/env python

import pytest
from modules.node import *
from modules.edge import *
from .fixtures import *
from itertools import cycle

# @pytest.mark.skip
def test_one_phase_update_lower_crossentropy_loss_MNISTPC(MNISTCudaPC,MNISTLoader100):

    MNISTCudaPC.initall(input_shape=torch.Size([100,784]),device=torch.device('cuda'))
    trainldr,testldr = MNISTLoader100
    trainiter = iter(trainldr)

    MNISTCudaPC.node_optim = torch.optim.SGD(MNISTCudaPC.nodes.parameters(),lr=0.05)
    MNISTCudaPC.edge_optim = torch.optim.SGD(MNISTCudaPC.edges.parameters(),lr=0.01)
    MNISTCudaPC.etol=1e-4
    MNISTCudaPC.beta = 50.
    MNISTCudaPC.max_iter = 100
    epoches = 10

    x,y = next(trainiter)
    x=x.cuda().flatten(start_dim=1)
    y=y.float().cuda()

    MNISTCudaPC.outnode = Node(state=y)
    MNISTCudaPC.innode = Node(x)
    MNISTCudaPC.feedforward(MNISTCudaPC.innode)
    initial_cost = torch.mean(MNISTCudaPC.cost())
    print(f'initial cost is {initial_cost}')
    for epoch in range(epoches):
        trainiter = iter(trainldr)
        for i in range(599):
            x,y = next(trainiter)
            x=x.cuda().flatten(start_dim=1)
            y=y.float().cuda()
            MNISTCudaPC.outnode = Node(state=y)
            if i%20==0:
                MNISTCudaPC.innode = Node(state=x)
                MNISTCudaPC.feedforward(MNISTCudaPC.innode)
                pre_cost = torch.mean(MNISTCudaPC.cost())
                out= MNISTCudaPC.nodes[-1].state
                print(f'step:{i},out 1st:{out[0]},y 1st:{y[0]},infer_cost:{pre_cost.item()}')
                MNISTCudaPC.edge_optim.zero_grad()
                MNISTCudaPC.node_optim.zero_grad()
            MNISTCudaPC.one_phase_update(x,y)
        correct = 0
        total = 0
        out,e_last,e_diff = MNISTCudaPC.infer(x,reset=True,beta=0)
        # Compute test batch accuracy, energy and store number of seen batches
        correct += float(torch.sum(torch.argmax(out,1) == y.argmax(dim=1)))
        total += x.size(0)
        print(f'epoch {epoch} accuracy: {correct/total}')
        current_cost = torch.mean(MNISTCudaPC.cost())
        MNISTCudaPC.edge_optim.zero_grad()
        MNISTCudaPC.node_optim.zero_grad()
    assert torch.mean(current_cost-initial_cost)<0.
