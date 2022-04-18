#!/usr/bin/env python


import pytest
from modules.node import *
from modules.edge import *
from .fixtures import *

def test_energy_gold_EP(MNISTCudaEP,MNISTgold,MNISTLoader):
    MNISTCudaEP.initall(input_shape=torch.Size([10,784]),device=torch.device('cuda'))
    MNISTCudaEP.activation = MNISTgold.phi[1]
    MNISTgold.to(torch.device('cuda'))
    MNISTgold.reset_state()
    trainldr,testldr = MNISTLoader
    trainiter = iter(trainldr)
    x,y = next(trainiter)
    x=x.view(-1,MNISTgold.dimensions[0]).cuda()
    y=y.cuda()
    MNISTgold.clamp_layer(0,x)
    MNISTgold.set_C_target(None)

    for i, u in enumerate(MNISTgold.u):
        MNISTCudaEP.nodes[i]().data=u.data.detach().clone()
    for i, W in enumerate(MNISTgold.W):
        MNISTCudaEP.edges[i].weight.data = W.weight.data.detach().clone()
        MNISTCudaEP.edges[i].bias.data = W.bias.data.detach().clone()
        print(MNISTCudaEP.edges[i].weight.data-W.weight.data)
        print(MNISTCudaEP.edges[i].bias.data-W.bias.data)
    MNISTCudaEP.clamp(0)
    MNISTgold.update_energy()
    print(f'gold is {MNISTgold.E} while mine is {MNISTCudaEP.energy()}')
    print(f'Initial states are:{MNISTCudaEP.nodes[0]()},{MNISTgold.u[0]}')
    MNISTCudaEP.beta=None
    assert AllEqual(MNISTgold.E,MNISTCudaEP.energy())<1e-3


# @pytest.mark.skip
def test_relax_gold_EP(MNISTCudaEP,MNISTgold,MNISTLoader):
    MNISTCudaEP.initall(input_shape=torch.Size([10,784]),device=torch.device('cuda'))
    MNISTgold.to(torch.device('cuda'))
    MNISTgold.reset_state()
    trainldr,testldr = MNISTLoader
    trainiter = iter(trainldr)
    x,y = next(trainiter)
    x=x.view(-1,MNISTgold.dimensions[0]).cuda()
    y=y.cuda()
    MNISTgold.clamp_layer(0,x)
    MNISTgold.set_C_target(None)

    for i, u in enumerate(MNISTgold.u):
        MNISTCudaEP.nodes[i].state=u.data.detach().detach().clone()
    for i, W in enumerate(MNISTgold.W):
        MNISTCudaEP.edges[i].weight.data = W.weight.data.detach().clone()
        MNISTCudaEP.edges[i].bias.data = W.bias.data.detach().clone()
    # MNISTCudaEP.clamp(0)
    MNISTgold.update_energy()
    print(f'gold is {MNISTgold.E} while mine is {MNISTCudaEP.energy()}')
    print(f'Initial states are:{MNISTCudaEP.nodes[0]()},{MNISTgold.u[0]}')
    MNISTCudaEP.beta=None
    MNISTgold.clamp_layer(0,x)
    print(f'Initial states 1 are:{MNISTCudaEP.nodes[1]()},{MNISTgold.u[1]}')
    print(f'Initial states are:{MNISTCudaEP.nodes[0]()},{MNISTgold.u[0]}')
    MNISTgold.u_relax(dt=0.1,n_relax=101,tol=0,tau=1.)
    MNISTCudaEP.node_optim = torch.optim.SGD(MNISTCudaEP.nodes.parameters(),lr=0.1)
    MNISTCudaEP.etol=0
    MNISTCudaEP.clamp(0)
    MNISTCudaEP.node_relax(lambda : torch.sum(MNISTCudaEP.energy()),100)
    print(f'Ending states are:{MNISTCudaEP.nodes[-1]()},{MNISTgold.u[-1]}')
    assert L2diff(MNISTCudaEP.nodes[-1](),MNISTgold.u[-1])<1e-3
